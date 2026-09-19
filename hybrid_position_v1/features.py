"""Leak-aware daily features for the USD/JPY hybrid shadow model.

The functions in this module are deliberately stateless: their outputs depend only
on the supplied frame.  Every predictor at date ``t`` is calculated with values at
or before ``t``.  Columns named ``target_z_*`` are forward-looking *labels* for
training/evaluation and must never be included in a model's predictor matrix.

Only pandas and NumPy are required.  The defaults mirror the frozen
``hybrid_position_v1/protocol.json``; callers may pass that decoded mapping to
``build_features`` to make the correspondence explicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "price",
    "high",
    "low",
    "vix",
    "sp500",
    "nikkei",
    "audjpy",
)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)
DEFAULT_MA_WINDOWS: tuple[int, ...] = (20, 50, 125, 200)
DEFAULT_FNG_WEIGHTS: Mapping[str, float] = {
    "momentum": 0.18,
    "strength": 0.12,
    "realized_vol": 0.18,
    "risk_regime": 0.14,
}


def _numeric(series: pd.Series) -> pd.Series:
    """Return a float Series while converting non-numeric observations to NaN."""

    return pd.to_numeric(series, errors="coerce").astype(float)


def _positive_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute log(numerator / denominator), returning NaN for invalid levels."""

    numerator = _numeric(numerator).where(lambda value: value > 0.0)
    denominator = _numeric(denominator).where(lambda value: value > 0.0)
    return np.log(numerator / denominator)


def validate_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a chronological, uniquely indexed daily input frame.

    Parameters
    ----------
    frame:
        DataFrame indexed by dates and containing ``REQUIRED_COLUMNS``.  Missing
        observations within those columns are allowed; missing columns are not.

    Returns
    -------
    pandas.DataFrame
        A defensive copy whose index is a ``DatetimeIndex`` and whose required
        market columns are floats.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame or its index cannot be parsed as dates.
    KeyError
        If a required column is absent.
    ValueError
        If dates are duplicated/not increasing, prices are non-positive, or a
        non-missing daily high is below the corresponding low.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    out = frame.copy(deep=True)
    try:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise TypeError("frame index must contain parseable dates") from exc
    if out.index.hasnans:
        raise ValueError("frame index must not contain NaT")
    if out.index.has_duplicates:
        raise ValueError("frame index must contain unique dates")
    if not out.index.is_monotonic_increasing:
        raise ValueError("frame index must be increasing")

    for column in REQUIRED_COLUMNS:
        out[column] = _numeric(out[column])

    for column in ("price", "high", "low"):
        invalid = out[column].notna() & (out[column] <= 0.0)
        if bool(invalid.any()):
            raise ValueError(f"{column} must be positive where present")
    invalid_range = out["high"].notna() & out["low"].notna() & (out["high"] < out["low"])
    if bool(invalid_range.any()):
        raise ValueError("high must be greater than or equal to low")
    return out


def rolling_percentile(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """Return the current observation's trailing mid-rank percentile (0--100).

    The rolling sample includes the current observation and never includes a
    future one.  Ties receive their mid-rank: values below the current value plus
    half of equal values, divided by the number of valid observations.
    """

    if window <= 0:
        raise ValueError("window must be positive")
    if min_periods <= 0 or min_periods > window:
        raise ValueError("min_periods must be in [1, window]")
    values = _numeric(series)

    def _midrank(sample: np.ndarray) -> float:
        current = sample[-1]
        if np.isnan(current):
            return np.nan
        valid = sample[~np.isnan(sample)]
        if valid.size < min_periods:
            return np.nan
        below = np.count_nonzero(valid < current)
        equal = np.count_nonzero(valid == current)
        return float(100.0 * (below + 0.5 * equal) / valid.size)

    result = values.rolling(window=window, min_periods=min_periods).apply(
        _midrank, raw=True
    )
    result.name = series.name
    return result


def daily_realized_volatility(
    price: pd.Series,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute trailing daily volatility from close-to-close log returns.

    Population standard deviation (``ddof=0``) is used.  The result is in daily,
    not annualized, units so it can scale an ``h``-day return by ``sqrt(h)``.
    """

    if window <= 1:
        raise ValueError("window must be greater than one")
    required = window if min_periods is None else min_periods
    if required <= 1 or required > window:
        raise ValueError("min_periods must be in [2, window]")
    close = _numeric(price).where(lambda value: value > 0.0)
    log_return = np.log(close / close.shift(1))
    result = log_return.rolling(window, min_periods=required).std(ddof=0)
    result.name = "rv20" if window == 20 else f"rv{window}"
    return result


def volatility_features(
    price: pd.Series,
    *,
    vol_window: int = 20,
    floor_window: int = 252,
    floor_quantile: float = 0.20,
    floor_min_periods: int = 60,
) -> pd.DataFrame:
    """Build RV, its trailing floor/median, effective scale, and vol ratio.

    ``vol_floor`` is the trailing 20th percentile by default and protects all
    normalized distances and targets from division by an unusually small RV.
    ``vol_ratio`` follows the frozen regime definition: annualized RV20 divided
    by its trailing-252 median.  Annualization cancels in the ratio, so daily RV
    is used on both sides for numerical simplicity.
    """

    if floor_window <= 0:
        raise ValueError("floor_window must be positive")
    if not 0.0 <= floor_quantile <= 1.0:
        raise ValueError("floor_quantile must be in [0, 1]")
    if floor_min_periods <= 0 or floor_min_periods > floor_window:
        raise ValueError("floor_min_periods must be in [1, floor_window]")

    rv = daily_realized_volatility(price, window=vol_window)
    floor = rv.rolling(floor_window, min_periods=floor_min_periods).quantile(
        floor_quantile
    )
    median = rv.rolling(floor_window, min_periods=floor_min_periods).median()
    # Before the floor has enough history, RV itself is a safe denominator.  The
    # floor is never backfilled, preserving the as-of property.
    effective = pd.concat([rv.rename("rv"), floor.rename("floor")], axis=1).max(
        axis=1, skipna=True
    )
    effective = effective.where(rv.notna()).replace(0.0, np.nan)

    out = pd.DataFrame(index=price.index)
    out["rv20"] = rv
    out["rv20_annualized"] = rv * np.sqrt(252.0)
    out["vol_floor"] = floor
    out["vol_floor_annualized"] = floor * np.sqrt(252.0)
    out["effective_daily_vol"] = effective
    out["vol_median"] = median
    out["vol_ratio"] = rv / median.replace(0.0, np.nan)
    return out


def _moving_average(series: pd.Series, window: int) -> pd.Series:
    """Moving average convention used by the audited upstream F&G project."""

    return _numeric(series).rolling(window, min_periods=max(2, window // 2)).mean()


def audited_market_fng(
    frame: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    normalization_window: int = 252,
    normalization_min_periods: int = 60,
) -> pd.DataFrame:
    """Compute the point-in-time-audited four-component market F&G index.

    The four admitted components are momentum, strength, realized volatility,
    and the cross-asset risk regime.  Rate differential, COT, and breadth are
    intentionally excluded.  Every component is oriented so a higher value means
    greed/risk-on/USDJPY-up pressure.  Available components are reweighted at
    each date, while ``fear_coverage`` reports their available protocol-weight
    share in [0, 1].
    """

    data = validate_daily_frame(frame)
    component_weights = dict(DEFAULT_FNG_WEIGHTS if weights is None else weights)
    expected = set(DEFAULT_FNG_WEIGHTS)
    if set(component_weights) != expected:
        raise ValueError(f"weights must define exactly {sorted(expected)}")
    if any((not np.isfinite(value)) or value < 0.0 for value in component_weights.values()):
        raise ValueError("F&G weights must be finite and nonnegative")
    total_weight = float(sum(component_weights.values()))
    if total_weight <= 0.0:
        raise ValueError("F&G weights must have a positive sum")

    price = data["price"]
    components = pd.DataFrame(index=data.index)

    ma125 = _moving_average(price, 125)
    momentum_raw = (price - ma125) / ma125.replace(0.0, np.nan)
    components["momentum"] = rolling_percentile(
        momentum_raw, normalization_window, normalization_min_periods
    )

    rolling_high = price.rolling(
        252, min_periods=normalization_min_periods
    ).max()
    rolling_low = price.rolling(
        252, min_periods=normalization_min_periods
    ).min()
    components["strength"] = (
        100.0
        * (price - rolling_low)
        / (rolling_high - rolling_low).replace(0.0, np.nan)
    ).clip(0.0, 100.0)

    rv20_annualized = daily_realized_volatility(price, 20, min_periods=10) * np.sqrt(
        252.0
    )
    components["realized_vol"] = 100.0 - rolling_percentile(
        rv20_annualized, normalization_window, normalization_min_periods
    )

    risk_parts: list[pd.Series] = []
    for column, ma_window in (("audjpy", 60), ("nikkei", 125), ("sp500", 125)):
        level = data[column]
        average = _moving_average(level, ma_window)
        raw = (level - average) / average.replace(0.0, np.nan)
        risk_parts.append(
            rolling_percentile(raw, normalization_window, normalization_min_periods)
        )
    risk_parts.append(
        100.0
        - rolling_percentile(
            data["vix"], normalization_window, normalization_min_periods
        )
    )
    components["risk_regime"] = pd.concat(risk_parts, axis=1).mean(
        axis=1, skipna=True
    )

    weight_vector = pd.Series(component_weights, dtype=float)
    available_weights = components.notna().mul(weight_vector, axis=1)
    denominator = available_weights.sum(axis=1)
    numerator = (components.fillna(0.0) * available_weights).sum(axis=1)
    level = numerator / denominator.replace(0.0, np.nan)

    out = components.rename(
        columns={column: f"fng_{column}" for column in components.columns}
    )
    out["audited_market_fng"] = level.clip(0.0, 100.0)
    out["fear_level"] = out["audited_market_fng"]
    out["fear_change_5d"] = out["fear_level"] - out["fear_level"].shift(5)
    out["fear_coverage"] = (denominator / total_weight).clip(0.0, 1.0)
    return out


def moving_average_features(
    price: pd.Series,
    daily_vol_scale: pd.Series,
    *,
    windows: Sequence[int] = DEFAULT_MA_WINDOWS,
    clip: float = 3.0,
) -> pd.DataFrame:
    """Return volatility-scaled MA-chain distances and a discrete trend score.

    The registered representation is ``price/MA20``, ``MA20/MA50``,
    ``MA50/MA125``, and ``MA125/MA200`` in log-distance units.  Distances are
    divided by effective daily volatility and clipped to ``[-clip, clip]``.
    ``trend_score`` is the mean of the four corresponding signs and therefore
    lies in ``[-1, 1]``.
    """

    if tuple(windows) != DEFAULT_MA_WINDOWS:
        raise ValueError(f"windows must be {DEFAULT_MA_WINDOWS}")
    if clip <= 0.0:
        raise ValueError("clip must be positive")

    close = _numeric(price)
    scale = _numeric(daily_vol_scale).replace(0.0, np.nan)
    averages = {
        window: close.rolling(window, min_periods=window).mean() for window in windows
    }
    raw_distances = {
        "log_price_ma20": _positive_log_ratio(close, averages[20]),
        "log_ma20_ma50": _positive_log_ratio(averages[20], averages[50]),
        "log_ma50_ma125": _positive_log_ratio(averages[50], averages[125]),
        "log_ma125_ma200": _positive_log_ratio(averages[125], averages[200]),
    }

    out = pd.DataFrame(index=price.index)
    for window, average in averages.items():
        out[f"ma{window}"] = average
    for name, raw in raw_distances.items():
        out[name] = (raw / scale).clip(-clip, clip)

    signs = pd.DataFrame(
        {
            name: np.sign(raw).where(raw.notna())
            for name, raw in raw_distances.items()
        },
        index=price.index,
    )
    # A regime is only declared after all four pieces of the chain are known.
    out["trend_score"] = signs.mean(axis=1, skipna=False)
    return out


def prior_year_pivots(frame: pd.DataFrame) -> pd.DataFrame:
    """Map classic prior-calendar-year pivot levels onto each current-year row.

    For year ``Y``, levels use only the high, low, and final valid close from
    ``Y-1``.  Consequently every level is fixed throughout ``Y`` and no value
    from the current year can affect it.
    """

    data = validate_daily_frame(frame)
    years = pd.Series(data.index.year, index=data.index, dtype=int)
    annual = data[["high", "low", "price"]].groupby(years).agg(
        high=("high", "max"), low=("low", "min"), close=("price", "last")
    )
    center = (annual["high"] + annual["low"] + annual["close"]) / 3.0
    annual_levels = pd.DataFrame(index=annual.index)
    annual_levels["pivot_p"] = center
    annual_levels["pivot_r1"] = 2.0 * center - annual["low"]
    annual_levels["pivot_s1"] = 2.0 * center - annual["high"]
    annual_levels["pivot_r2"] = center + annual["high"] - annual["low"]
    annual_levels["pivot_s2"] = center - annual["high"] + annual["low"]

    source_year = years - 1
    out = pd.DataFrame(index=data.index)
    out["pivot_source_year"] = source_year.where(
        source_year.isin(annual_levels.index)
    ).astype(float)
    for column in annual_levels.columns:
        out[column] = source_year.map(annual_levels[column]).to_numpy(dtype=float)
    return out


def pivot_distance_features(
    price: pd.Series,
    pivots: pd.DataFrame,
    daily_vol_scale: pd.Series,
    *,
    clip: float = 3.0,
) -> pd.DataFrame:
    """Build clipped distances to prior-year center/support/resistance levels.

    Center distance is signed.  Support and resistance distances are nonnegative
    proximity measures to the nearer of S1/S2 and R1/R2 respectively.  All three
    use the same effective daily-volatility denominator.
    """

    required = {"pivot_p", "pivot_r1", "pivot_s1", "pivot_r2", "pivot_s2"}
    missing = required.difference(pivots.columns)
    if missing:
        raise KeyError(f"missing pivot columns: {sorted(missing)}")
    if clip <= 0.0:
        raise ValueError("clip must be positive")

    scale = _numeric(daily_vol_scale).replace(0.0, np.nan)
    center = _positive_log_ratio(price, pivots["pivot_p"]) / scale
    support = pd.concat(
        [
            _positive_log_ratio(price, pivots["pivot_s1"]).abs(),
            _positive_log_ratio(price, pivots["pivot_s2"]).abs(),
        ],
        axis=1,
    ).min(axis=1, skipna=False) / scale
    resistance = pd.concat(
        [
            _positive_log_ratio(pivots["pivot_r1"], price).abs(),
            _positive_log_ratio(pivots["pivot_r2"], price).abs(),
        ],
        axis=1,
    ).min(axis=1, skipna=False) / scale

    return pd.DataFrame(
        {
            "pivot_center_distance": center.clip(-clip, clip),
            "pivot_support_distance": support.clip(0.0, clip),
            "pivot_resistance_distance": resistance.clip(0.0, clip),
        },
        index=price.index,
    )


def _remaining_business_days(
    dates: pd.DatetimeIndex, target_dates: pd.Series
) -> pd.Series:
    """Count weekdays from each observation date up to its target date."""

    naive_dates = dates.tz_localize(None) if dates.tz is not None else dates
    targets = pd.DatetimeIndex(pd.to_datetime(target_dates, errors="coerce"))
    if targets.tz is not None:
        targets = targets.tz_localize(None)
    valid = ~targets.isna()
    result = np.full(len(dates), np.nan, dtype=float)
    if bool(valid.any()):
        starts = naive_dates.values.astype("datetime64[D]")[valid]
        ends = targets.values.astype("datetime64[D]")[valid]
        result[valid] = np.busday_count(starts, ends).astype(float)
    return pd.Series(result, index=dates, name="macro_remaining_business_days")


def macro_features(
    frame: pd.DataFrame,
    daily_vol_scale: pd.Series,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    clip: float = 3.0,
) -> pd.DataFrame:
    """Create horizon-aware gaps from the as-of macro anchor to spot.

    Sparse macro observations are forward-filled only (never backfilled).  A
    target is available while its target date is not earlier than the market
    date.  For horizon ``h`` the full log gap is multiplied by
    ``min(1, h / max(remaining_business_days, h))`` and divided by
    ``effective_daily_vol * sqrt(h)`` before clipping.
    """

    if tuple(horizons) != DEFAULT_HORIZONS:
        raise ValueError(f"horizons must be {DEFAULT_HORIZONS}")
    if clip <= 0.0:
        raise ValueError("clip must be positive")

    out = pd.DataFrame(index=frame.index)
    for horizon in horizons:
        out[f"macro_gap_{horizon}"] = np.nan
    out["macro_available"] = 0.0
    out["macro_remaining_business_days"] = np.nan
    if "macro_anchor" not in frame.columns:
        return out

    anchor = _numeric(frame["macro_anchor"]).ffill()
    if "macro_target_date" in frame.columns:
        raw_target = pd.to_datetime(frame["macro_target_date"], errors="coerce")
        target = raw_target.ffill()
        remaining = _remaining_business_days(frame.index, target)
        target_valid = target.notna() & (remaining >= 0.0)
    else:
        remaining = pd.Series(np.nan, index=frame.index, dtype=float)
        target_valid = pd.Series(True, index=frame.index, dtype=bool)

    available = anchor.gt(0.0) & target_valid
    out["macro_available"] = available.astype(float)
    out["macro_remaining_business_days"] = remaining
    full_gap = _positive_log_ratio(anchor, frame["price"])
    scale = _numeric(daily_vol_scale).replace(0.0, np.nan)

    for horizon in horizons:
        if "macro_target_date" in frame.columns:
            denominator_days = remaining.clip(lower=float(horizon))
            convergence_fraction = (float(horizon) / denominator_days).clip(upper=1.0)
        else:
            # Without a dated macro target, retain the anchor as a horizon-level
            # fair-value gap rather than inventing a convergence speed.
            convergence_fraction = pd.Series(1.0, index=frame.index)
        gap = (
            full_gap
            * convergence_fraction
            / (scale * np.sqrt(float(horizon)))
        ).where(available)
        out[f"macro_gap_{horizon}"] = gap.clip(-clip, clip)
    return out


def forward_targets(
    price: pd.Series,
    daily_vol_scale: pd.Series,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Return normalized future-return labels for model training/evaluation.

    ``target_z_h[t] = log(price[t+h] / price[t]) / (scale[t] * sqrt(h))``.
    These columns intentionally use future prices and are labels, not predictors.
    A walk-forward trainer must purge rows whose target has not resolved by the
    training cutoff, as required by the frozen protocol.
    """

    close = _numeric(price).where(lambda value: value > 0.0)
    scale = _numeric(daily_vol_scale).replace(0.0, np.nan)
    out = pd.DataFrame(index=price.index)
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        future_return = np.log(close.shift(-horizon) / close)
        out[f"target_z_{horizon}"] = future_return / (
            scale * np.sqrt(float(horizon))
        )
    return out


def _protocol_settings(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract registered feature settings without mutating the mapping."""

    if protocol is None:
        return {
            "weights": dict(DEFAULT_FNG_WEIGHTS),
            "normalization_window": 252,
            "normalization_min_periods": 60,
            "ma_windows": DEFAULT_MA_WINDOWS,
            "horizons": DEFAULT_HORIZONS,
            "clip": 3.0,
            "vol_window": 20,
            "floor_window": 252,
            "floor_quantile": 0.20,
        }

    fear = protocol["fear_and_greed"]
    features = protocol["features"]
    prediction = protocol["prediction"]
    return {
        "weights": dict(fear["strict_component_weights"]),
        "normalization_window": int(fear["normalization_window"]),
        "normalization_min_periods": int(fear["normalization_min_periods"]),
        "ma_windows": tuple(int(value) for value in features["moving_average_windows"]),
        "horizons": tuple(int(value) for value in prediction["horizons_business_days"]),
        "clip": float(features["feature_clip"]),
        "vol_window": int(features["annualized_vol_window"]),
        "floor_window": int(features["vol_floor_window"]),
        "floor_quantile": float(features["vol_floor_quantile"]),
    }


def build_features(
    frame: pd.DataFrame,
    protocol: Mapping[str, Any] | None = None,
    *,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Build the complete registered daily feature/label table.

    All predictor columns are as-of the completed close at their row date.  Set
    ``include_targets=False`` for live inference; when true, ``target_z_1/5/20``
    are appended strictly as supervised labels.
    """

    data = validate_daily_frame(frame)
    settings = _protocol_settings(protocol)

    vol = volatility_features(
        data["price"],
        vol_window=settings["vol_window"],
        floor_window=settings["floor_window"],
        floor_quantile=settings["floor_quantile"],
        floor_min_periods=settings["normalization_min_periods"],
    )
    fng = audited_market_fng(
        data,
        weights=settings["weights"],
        normalization_window=settings["normalization_window"],
        normalization_min_periods=settings["normalization_min_periods"],
    )
    moving_average = moving_average_features(
        data["price"],
        vol["effective_daily_vol"],
        windows=settings["ma_windows"],
        clip=settings["clip"],
    )
    pivots = prior_year_pivots(data)
    pivot_distances = pivot_distance_features(
        data["price"], pivots, vol["effective_daily_vol"], clip=settings["clip"]
    )
    macro = macro_features(
        data,
        vol["effective_daily_vol"],
        horizons=settings["horizons"],
        clip=settings["clip"],
    )

    pieces = [vol, fng, moving_average, pivots, pivot_distances, macro]
    if include_targets:
        pieces.append(
            forward_targets(
                data["price"],
                vol["effective_daily_vol"],
                horizons=settings["horizons"],
            )
        )
    return pd.concat(pieces, axis=1)


__all__ = [
    "REQUIRED_COLUMNS",
    "rolling_percentile",
    "daily_realized_volatility",
    "volatility_features",
    "audited_market_fng",
    "moving_average_features",
    "prior_year_pivots",
    "pivot_distance_features",
    "macro_features",
    "forward_targets",
    "validate_daily_frame",
    "build_features",
]
