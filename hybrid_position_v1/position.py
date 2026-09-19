"""Position construction and semi-strict close-to-close backtest.

The module deliberately has no dependency other than pandas and NumPy.  A
forecast observed after the completed close on day ``t`` becomes the position
labelled ``t`` and earns the close-to-close simple return from ``t`` to ``t+1``.
Consequently, the backtest must *not* shift ``position`` again when computing
PnL.

Required forecast inputs are ``pred_1``, ``pred_5`` and ``pred_20``.  They are
standardised forecasts::

    future log return / (daily_vol * sqrt(horizon))

The public functions accept explicit column mappings, while a small set of
unambiguous aliases is supported for convenience.  ``backtest_positions`` is
the primary entry point; ``backtest`` and ``make_positions`` are compatibility
aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252.0


@dataclass(frozen=True)
class PositionConfig:
    """Frozen defaults registered in ``hybrid_position_v1/protocol.json``."""

    horizon_weights: tuple[tuple[int, float], ...] = (
        (1, 0.20),
        (5, 0.40),
        (20, 0.40),
    )
    minimum_agreeing_horizons: int = 2
    entry_band_sigma: float = 0.20
    exit_band_sigma: float = 0.10
    strength_full_scale_sigma: float = 0.80
    annual_vol_target: float = 0.05
    maximum_absolute_position: float = 1.00
    maximum_daily_position_change: float = 0.25
    trend_cap: float = 1.00
    range_cap: float = 0.60
    overheat_cap: float = 0.50
    stress_cap: float = 0.35
    stress_vol_ratio: float = 1.50
    fear_threshold: float = 24.0
    greed_threshold: float = 76.0
    trend_score_threshold: float = 0.50
    pivot_proximity_5d_vol_fraction: float = 0.25
    pivot_cap_when_blocked: float = 0.50
    one_way_cost_bps: float = 2.50
    stress_one_way_cost_bps: float = 5.00
    drawdown_half_threshold: float = 0.04
    drawdown_flat_threshold: float = 0.06
    daily_loss_cap: float = 0.01
    post_loss_cap: float = 0.25
    post_loss_days: int = 2

    @property
    def weights(self) -> dict[int, float]:
        return dict(self.horizon_weights)

    @property
    def regime_caps(self) -> dict[str, float]:
        return {
            "trend": self.trend_cap,
            "range": self.range_cap,
            "overheat": self.overheat_cap,
            "stress": self.stress_cap,
        }


DEFAULT_CONFIG = PositionConfig()


_FORECAST_ALIASES: dict[int, tuple[str, ...]] = {
    1: ("pred_1", "pred_1d", "prediction_1", "prediction_1d", "forecast_1", "forecast_1d"),
    5: ("pred_5", "pred_5d", "prediction_5", "prediction_5d", "forecast_5", "forecast_5d"),
    20: ("pred_20", "pred_20d", "prediction_20", "prediction_20d", "forecast_20", "forecast_20d"),
}

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "price": ("price", "close", "Close", "usdjpy", "spot"),
    "daily_vol": ("daily_vol", "realized_vol", "sigma_daily", "vol_20d"),
    "annualized_vol": (
        "annualized_vol",
        "annualised_vol",
        "ann_vol",
        "realized_vol_annualized",
        "vol_20d_annualized",
    ),
    "fear": ("audited_market_fng", "fear_level", "fng", "fear_and_greed"),
    "trend_score": ("trend_score",),
    "vol_ratio": ("vol_ratio",),
    "regime": ("regime",),
}

_PIVOT_ALIASES: dict[str, tuple[str, ...]] = {
    "p": ("pivot_p", "p", "pivot"),
    "r1": ("pivot_r1", "r1"),
    "s1": ("pivot_s1", "s1"),
    "r2": ("pivot_r2", "r2"),
    "s2": ("pivot_s2", "s2"),
}


def _coerce_config(config: PositionConfig | Mapping[str, object] | None) -> PositionConfig:
    if config is None:
        return DEFAULT_CONFIG
    if isinstance(config, PositionConfig):
        return config
    if isinstance(config, Mapping):
        # Accept either the position block from protocol.json or the complete
        # decoded protocol.  The JSON names remain the source of truth; the
        # shorter dataclass names are only an internal convenience.
        if "position" in config and isinstance(config["position"], Mapping):
            config = config["position"]  # type: ignore[assignment]
        aliases = {
            "entry_band_sigma_after_cost": "entry_band_sigma",
            "exit_band_sigma_after_cost": "exit_band_sigma",
            "base_one_way_cost_bps": "one_way_cost_bps",
            "drawdown_half_cap": "drawdown_half_threshold",
            "drawdown_flat_and_review": "drawdown_flat_threshold",
        }
        values = {name: getattr(DEFAULT_CONFIG, name) for name in DEFAULT_CONFIG.__dataclass_fields__}
        for original_name, value in config.items():
            name = aliases.get(str(original_name), str(original_name))
            if name == "horizon_weights" and isinstance(value, Mapping):
                value = tuple(
                    (
                        horizon,
                        float(
                            value[str(horizon)]
                            if str(horizon) in value
                            else value[horizon]
                        ),
                    )
                    for horizon in (1, 5, 20)
                )
            elif name == "regime_caps" and isinstance(value, Mapping):
                for regime in ("trend", "range", "overheat", "stress"):
                    if regime in value:
                        values[f"{regime}_cap"] = float(value[regime])
                continue
            # Daily loss/post-loss fields are part of the registered emergency
            # overlay and map directly because their protocol names match.
            if name in values:
                values[name] = value
        return PositionConfig(**values)
    raise TypeError("config must be PositionConfig, a mapping, or None")


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _resolve_forecast_columns(
    frame: pd.DataFrame,
    forecast_columns: Mapping[int | str, str] | None,
) -> dict[int, str]:
    explicit = {int(key): value for key, value in (forecast_columns or {}).items()}
    resolved: dict[int, str] = {}
    for horizon in (1, 5, 20):
        column = explicit.get(horizon) or _first_column(frame, _FORECAST_ALIASES[horizon])
        if column is None:
            aliases = ", ".join(_FORECAST_ALIASES[horizon])
            raise KeyError(f"missing {horizon}-day forecast; expected one of: {aliases}")
        if column not in frame.columns:
            raise KeyError(f"forecast column does not exist: {column}")
        resolved[horizon] = column
    return resolved


def _resolve_column(frame: pd.DataFrame, role: str, explicit: str | None = None) -> str | None:
    if explicit is not None:
        if explicit not in frame.columns:
            raise KeyError(f"{role} column does not exist: {explicit}")
        return explicit
    return _first_column(frame, _COLUMN_ALIASES[role])


def _direction(value: float, *, tolerance: float = 1e-15) -> int:
    if not np.isfinite(value) or abs(value) <= tolerance:
        return 0
    return 1 if value > 0.0 else -1


def classify_regime(
    trend_score: float,
    vol_ratio: float,
    fear_level: float,
    config: PositionConfig | Mapping[str, object] | None = None,
) -> str:
    """Classify one observation, applying the safest regime first.

    Priority is stress, overheat, trend, then range.  Missing inputs do not by
    themselves create a risky regime; in particular, a missing trend score is
    classified as range.
    """

    cfg = _coerce_config(config)
    if np.isfinite(vol_ratio) and vol_ratio >= cfg.stress_vol_ratio:
        return "stress"
    if np.isfinite(fear_level) and (
        fear_level <= cfg.fear_threshold or fear_level >= cfg.greed_threshold
    ):
        return "overheat"
    if np.isfinite(trend_score) and abs(trend_score) >= cfg.trend_score_threshold:
        return "trend"
    return "range"


def combine_horizon_forecasts(
    forecasts: Mapping[int | str, float] | Sequence[float],
    daily_vol: float,
    *,
    one_way_cost_bps: float = 2.5,
    config: PositionConfig | Mapping[str, object] | None = None,
) -> dict[str, float | int]:
    """Combine 1/5/20-day standardised forecasts and deduct one-way cost.

    Cost is translated into each forecast's z unit as
    ``cost / (daily_vol * sqrt(h))`` before the registered 20/40/40 blend.
    Missing forecasts contribute zero and cannot count toward agreement; the
    registered weights are never re-normalised around missing values.
    """

    cfg = _coerce_config(config)
    if isinstance(forecasts, Mapping):
        values = {
            horizon: float(forecasts.get(horizon, forecasts.get(str(horizon), np.nan)))
            for horizon in (1, 5, 20)
        }
    else:
        seq = list(forecasts)
        if len(seq) != 3:
            raise ValueError("forecasts must contain exactly 1, 5 and 20-day values")
        values = dict(zip((1, 5, 20), (float(value) for value in seq)))

    raw = 0.0
    after_cost = 0.0
    cost = max(float(one_way_cost_bps), 0.0) / 10_000.0
    valid_daily_vol = np.isfinite(daily_vol) and daily_vol > 0.0
    effective_cost_sigma = 0.0
    valid_horizons = 0

    for horizon, weight in cfg.weights.items():
        prediction = values[horizon]
        if not np.isfinite(prediction):
            continue
        valid_horizons += 1
        raw += weight * prediction
        horizon_cost_sigma = cost / (daily_vol * np.sqrt(float(horizon))) if valid_daily_vol else 0.0
        effective_cost_sigma += weight * horizon_cost_sigma
        net_magnitude = max(abs(prediction) - horizon_cost_sigma, 0.0)
        after_cost += weight * _direction(prediction) * net_magnitude

    signal_direction = _direction(after_cost)
    positive = sum(np.isfinite(value) and value > 0.0 for value in values.values())
    negative = sum(np.isfinite(value) and value < 0.0 for value in values.values())
    agreeing = positive if signal_direction > 0 else negative if signal_direction < 0 else 0

    return {
        "raw_signal_sigma": float(raw),
        "after_cost_signal_sigma": float(after_cost),
        "effective_cost_sigma": float(effective_cost_sigma),
        "signal_direction": int(signal_direction),
        "agreeing_horizons": int(agreeing),
        "valid_horizons": int(valid_horizons),
    }


def _pivot_block(
    price: float,
    daily_vol: float,
    direction: int,
    pivots: Mapping[str, float],
    cfg: PositionConfig,
) -> tuple[bool, float, str]:
    """Return whether a prior-year pivot blocks the intended direction."""

    if direction == 0 or not (np.isfinite(price) and price > 0.0):
        return False, np.nan, ""
    if not (np.isfinite(daily_vol) and daily_vol > 0.0):
        return False, np.nan, ""

    # Both forecasts and realised volatility use log-return units, so pivot
    # proximity is measured in log distance rather than a price-point
    # approximation.
    threshold = daily_vol * np.sqrt(5.0) * cfg.pivot_proximity_5d_vol_fraction
    candidates: list[tuple[float, str]] = []
    for name, value in pivots.items():
        if not np.isfinite(value):
            continue
        if float(value) <= 0.0:
            continue
        signed_distance = float(np.log(float(value) / price))
        if direction > 0 and signed_distance >= 0.0:
            candidates.append((signed_distance, name))
        elif direction < 0 and signed_distance <= 0.0:
            candidates.append((-signed_distance, name))

    if not candidates:
        return False, np.nan, ""
    distance, name = min(candidates, key=lambda item: item[0])
    return bool(distance <= threshold), float(distance), name


def _prepare_inputs(
    data: pd.DataFrame,
    forecast_columns: Mapping[int | str, str] | None,
    price_col: str | None,
    daily_vol_col: str | None,
    annualized_vol_col: str | None,
    regime_col: str | None,
    fear_col: str | None,
    trend_score_col: str | None,
    vol_ratio_col: str | None,
    pivot_columns: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, dict[int, str], dict[str, str | None], dict[str, str | None]]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    frame = data.copy()
    forecast_map = _resolve_forecast_columns(frame, forecast_columns)
    columns = {
        "price": _resolve_column(frame, "price", price_col),
        "daily_vol": _resolve_column(frame, "daily_vol", daily_vol_col),
        "annualized_vol": _resolve_column(frame, "annualized_vol", annualized_vol_col),
        "regime": _resolve_column(frame, "regime", regime_col),
        "fear": _resolve_column(frame, "fear", fear_col),
        "trend_score": _resolve_column(frame, "trend_score", trend_score_col),
        "vol_ratio": _resolve_column(frame, "vol_ratio", vol_ratio_col),
    }
    explicit_pivots = pivot_columns or {}
    pivots: dict[str, str | None] = {}
    for level, aliases in _PIVOT_ALIASES.items():
        column = explicit_pivots.get(level)
        if column is not None and column not in frame.columns:
            raise KeyError(f"pivot column does not exist: {column}")
        pivots[level] = column or _first_column(frame, aliases)
    return frame, forecast_map, columns, pivots


def _forward_returns(
    frame: pd.DataFrame,
    price: pd.Series,
    forward_return_col: str | None,
) -> pd.Series:
    if forward_return_col is not None:
        if forward_return_col not in frame.columns:
            raise KeyError(f"forward return column does not exist: {forward_return_col}")
        return pd.to_numeric(frame[forward_return_col], errors="coerce").astype(float)
    if price.notna().sum() < 2:
        raise KeyError("backtest requires price/close or an explicit forward_return_col")
    valid_price = price.where(price > 0.0)
    return (valid_price.shift(-1) / valid_price - 1.0).astype(float)


def _construct(
    data: pd.DataFrame,
    *,
    config: PositionConfig | Mapping[str, object] | None = None,
    forecast_columns: Mapping[int | str, str] | None = None,
    price_col: str | None = None,
    daily_vol_col: str | None = None,
    annualized_vol_col: str | None = None,
    regime_col: str | None = None,
    fear_col: str | None = None,
    trend_score_col: str | None = None,
    vol_ratio_col: str | None = None,
    pivot_columns: Mapping[str, str] | None = None,
    forward_return_col: str | None = None,
    require_returns: bool,
) -> pd.DataFrame:
    cfg = _coerce_config(config)
    frame, forecast_map, columns, pivot_map = _prepare_inputs(
        data,
        forecast_columns,
        price_col,
        daily_vol_col,
        annualized_vol_col,
        regime_col,
        fear_col,
        trend_score_col,
        vol_ratio_col,
        pivot_columns,
    )

    price = _numeric_series(frame, columns["price"])
    daily_vol = _numeric_series(frame, columns["daily_vol"])
    annualized_vol = _numeric_series(frame, columns["annualized_vol"])
    annualized_vol = annualized_vol.where(annualized_vol > 0.0, daily_vol * np.sqrt(TRADING_DAYS))
    fear = _numeric_series(frame, columns["fear"])
    trend_score = _numeric_series(frame, columns["trend_score"])
    vol_ratio = _numeric_series(frame, columns["vol_ratio"])
    predictions = {
        horizon: _numeric_series(frame, column) for horizon, column in forecast_map.items()
    }
    pivots = {
        level: _numeric_series(frame, column) for level, column in pivot_map.items()
    }

    if require_returns:
        forward_return = _forward_returns(frame, price, forward_return_col)
    elif forward_return_col is not None:
        forward_return = _forward_returns(frame, price, forward_return_col)
    elif price.notna().sum() >= 2:
        valid_price = price.where(price > 0.0)
        forward_return = valid_price.shift(-1) / valid_price - 1.0
    else:
        forward_return = pd.Series(np.nan, index=frame.index, dtype=float)

    # Preallocate plain NumPy arrays so stateful risk controls are explicit and
    # cannot accidentally use a pandas lead/forward-fill operation.
    n_rows = len(frame)
    output_arrays: dict[str, np.ndarray] = {
        "raw_signal_sigma": np.full(n_rows, np.nan),
        "after_cost_signal_sigma": np.full(n_rows, np.nan),
        "effective_cost_sigma": np.full(n_rows, np.nan),
        "agreeing_horizons": np.zeros(n_rows, dtype=int),
        "valid_horizons": np.zeros(n_rows, dtype=int),
        "signal_direction": np.zeros(n_rows, dtype=int),
        "annualized_vol_used": annualized_vol.to_numpy(dtype=float),
        "vol_target_multiplier": np.zeros(n_rows),
        "regime_cap": np.zeros(n_rows),
        "pivot_cap": np.ones(n_rows),
        "pivot_blocked": np.zeros(n_rows, dtype=bool),
        "pivot_distance": np.full(n_rows, np.nan),
        "drawdown_before_trade": np.zeros(n_rows),
        "drawdown_multiplier": np.ones(n_rows),
        "post_loss_days_remaining": np.zeros(n_rows, dtype=int),
        "post_loss_cap": np.ones(n_rows),
        "target_position_unconstrained": np.zeros(n_rows),
        "target_position": np.zeros(n_rows),
        "position": np.zeros(n_rows),
        "turnover": np.zeros(n_rows),
        "cost_bps": np.full(n_rows, cfg.one_way_cost_bps),
        "transaction_cost": np.zeros(n_rows),
        "forward_return": forward_return.to_numpy(dtype=float),
        "gross_pnl": np.full(n_rows, np.nan),
        "net_pnl": np.full(n_rows, np.nan),
        "equity": np.ones(n_rows),
    }
    regimes = np.empty(n_rows, dtype=object)
    pivot_levels = np.empty(n_rows, dtype=object)
    cap_reasons = np.empty(n_rows, dtype=object)

    previous_position = 0.0
    held_direction = 0
    equity = 1.0
    peak_equity = 1.0
    previous_pending_net = np.nan
    post_loss_days_remaining = 0

    for i in range(n_rows):
        # At close i the i-1 -> i result has become observable.  It is applied
        # before today's drawdown control and never before the i-1 decision.
        if i > 0 and np.isfinite(previous_pending_net):
            equity *= max(1.0 + previous_pending_net, 1e-12)
            peak_equity = max(peak_equity, equity)
            if previous_pending_net <= -cfg.daily_loss_cap:
                post_loss_days_remaining = int(cfg.post_loss_days)
        drawdown = max(0.0, 1.0 - equity / peak_equity) if peak_equity > 0.0 else 1.0
        output_arrays["drawdown_before_trade"][i] = drawdown

        explicit_regime = None
        if columns["regime"] is not None:
            value = frame.iloc[i][columns["regime"]]
            if pd.notna(value):
                candidate = str(value).strip().lower()
                if candidate in cfg.regime_caps:
                    explicit_regime = candidate
        regime = explicit_regime or classify_regime(
            trend_score.iloc[i], vol_ratio.iloc[i], fear.iloc[i], cfg
        )
        regimes[i] = regime
        regime_cap = cfg.regime_caps[regime]
        output_arrays["regime_cap"][i] = regime_cap
        cost_bps = cfg.stress_one_way_cost_bps if regime == "stress" else cfg.one_way_cost_bps
        output_arrays["cost_bps"][i] = cost_bps

        row_forecasts = {horizon: series.iloc[i] for horizon, series in predictions.items()}
        combination = combine_horizon_forecasts(
            row_forecasts,
            daily_vol.iloc[i],
            one_way_cost_bps=cost_bps,
            config=cfg,
        )
        for key in (
            "raw_signal_sigma",
            "after_cost_signal_sigma",
            "effective_cost_sigma",
            "agreeing_horizons",
            "valid_horizons",
            "signal_direction",
        ):
            output_arrays[key][i] = combination[key]

        signal = float(combination["after_cost_signal_sigma"])
        signal_direction = int(combination["signal_direction"])
        agreeing = int(combination["agreeing_horizons"])
        has_agreement = agreeing >= cfg.minimum_agreeing_horizons
        signal_magnitude = abs(signal)
        reasons: list[str] = []

        # Hysteresis is stateful.  Losing agreement closes an active intent;
        # reversal requires satisfying the entry band in the new direction.
        if held_direction == 0:
            if has_agreement and signal_magnitude >= cfg.entry_band_sigma:
                held_direction = signal_direction
            else:
                reasons.append("no_agreement" if not has_agreement else "entry_band")
        elif signal_direction == held_direction and has_agreement:
            if signal_magnitude < cfg.exit_band_sigma:
                held_direction = 0
                reasons.append("exit_band")
        elif (
            signal_direction == -held_direction
            and has_agreement
            and signal_magnitude >= cfg.entry_band_sigma
        ):
            held_direction = signal_direction
            reasons.append("signal_reversal")
        else:
            held_direction = 0
            reasons.append("lost_direction")

        ann_vol = annualized_vol.iloc[i]
        if np.isfinite(ann_vol) and ann_vol > 0.0:
            vol_multiplier = min(cfg.annual_vol_target / ann_vol, 1.0)
        else:
            vol_multiplier = 0.0
            held_direction = 0
            reasons.append("missing_vol")
        output_arrays["vol_target_multiplier"][i] = vol_multiplier

        strength = min(signal_magnitude / cfg.strength_full_scale_sigma, 1.0)
        unconstrained = held_direction * strength * vol_multiplier
        output_arrays["target_position_unconstrained"][i] = unconstrained

        target = float(np.clip(unconstrained, -cfg.maximum_absolute_position, cfg.maximum_absolute_position))
        if not np.isclose(target, unconstrained):
            reasons.append("absolute_cap")
        if abs(target) > regime_cap:
            target = float(np.sign(target) * regime_cap)
            reasons.append(f"regime:{regime}")

        pivot_values = {name: series.iloc[i] for name, series in pivots.items()}
        pivot_blocked, pivot_distance, pivot_level = _pivot_block(
            price.iloc[i], daily_vol.iloc[i], _direction(target), pivot_values, cfg
        )
        pivot_levels[i] = pivot_level
        output_arrays["pivot_blocked"][i] = pivot_blocked
        output_arrays["pivot_distance"][i] = pivot_distance
        if pivot_blocked:
            output_arrays["pivot_cap"][i] = cfg.pivot_cap_when_blocked
            if abs(target) > cfg.pivot_cap_when_blocked:
                target = float(np.sign(target) * cfg.pivot_cap_when_blocked)
                reasons.append(f"pivot:{pivot_level}")

        if drawdown >= cfg.drawdown_flat_threshold:
            target = 0.0
            held_direction = 0
            output_arrays["drawdown_multiplier"][i] = 0.0
            reasons.append("drawdown_flat")
        elif drawdown >= cfg.drawdown_half_threshold:
            target *= 0.5
            output_arrays["drawdown_multiplier"][i] = 0.5
            reasons.append("drawdown_half")

        output_arrays["post_loss_days_remaining"][i] = post_loss_days_remaining
        if post_loss_days_remaining > 0:
            output_arrays["post_loss_cap"][i] = cfg.post_loss_cap
            if abs(target) > cfg.post_loss_cap:
                target = float(np.sign(target) * cfg.post_loss_cap)
                reasons.append("post_loss_cap")

        output_arrays["target_position"][i] = target

        # A hard risk reduction (6% stop, or a newly tighter cap) may exceed the
        # normal 0.25/day speed limit.  Risk-increasing moves never do.
        lower = previous_position - cfg.maximum_daily_position_change
        upper = previous_position + cfg.maximum_daily_position_change
        position = float(np.clip(target, lower, upper))
        if not np.isclose(position, target):
            reasons.append("daily_change")
        if drawdown >= cfg.drawdown_flat_threshold:
            position = 0.0
        else:
            hard_cap = min(
                cfg.maximum_absolute_position,
                regime_cap,
                output_arrays["pivot_cap"][i],
                output_arrays["post_loss_cap"][i],
            )
            if abs(position) > hard_cap:
                position = float(np.sign(position) * hard_cap)
                reasons.append("risk_cap_override")

        turnover = abs(position - previous_position)
        transaction_cost = turnover * cost_bps / 10_000.0
        fwd = forward_return.iloc[i]
        gross = position * fwd if np.isfinite(fwd) else np.nan
        net = gross - transaction_cost if np.isfinite(gross) else np.nan

        output_arrays["position"][i] = position
        output_arrays["turnover"][i] = turnover
        output_arrays["transaction_cost"][i] = transaction_cost
        output_arrays["gross_pnl"][i] = gross
        output_arrays["net_pnl"][i] = net
        output_arrays["equity"][i] = (
            equity * max(1.0 + net, 1e-12) if np.isfinite(net) else equity
        )
        cap_reasons[i] = "|".join(dict.fromkeys(reasons)) if reasons else "none"

        previous_position = position
        previous_pending_net = net
        if post_loss_days_remaining > 0:
            post_loss_days_remaining -= 1

    result = frame.copy()
    for horizon in (1, 5, 20):
        result[f"pred_return_{horizon}"] = (
            predictions[horizon] * daily_vol * np.sqrt(float(horizon))
        )
    for name, values in output_arrays.items():
        result[name] = values
    result["regime"] = regimes
    result["pivot_level"] = pivot_levels
    result["cap_reason"] = cap_reasons
    # Concise aliases used by reports and downstream notebooks.
    result["signal_sigma"] = result["after_cost_signal_sigma"]
    result["drawdown"] = result["drawdown_before_trade"]
    result["gross_return"] = result["gross_pnl"]
    result["net_return"] = result["net_pnl"]
    return result


def build_positions(
    data: pd.DataFrame,
    *,
    config: PositionConfig | Mapping[str, object] | None = None,
    forecast_columns: Mapping[int | str, str] | None = None,
    price_col: str | None = None,
    daily_vol_col: str | None = None,
    annualized_vol_col: str | None = None,
    regime_col: str | None = None,
    fear_col: str | None = None,
    trend_score_col: str | None = None,
    vol_ratio_col: str | None = None,
    pivot_columns: Mapping[str, str] | None = None,
    forward_return_col: str | None = None,
) -> pd.DataFrame:
    """Construct positions; returns are optional but enable drawdown controls.

    If ``price``/``close`` or ``forward_return_col`` is present, the same
    stateful drawdown logic as the backtest is applied.  Without returns,
    drawdown remains zero and all other controls still operate.
    """

    return _construct(
        data,
        config=config,
        forecast_columns=forecast_columns,
        price_col=price_col,
        daily_vol_col=daily_vol_col,
        annualized_vol_col=annualized_vol_col,
        regime_col=regime_col,
        fear_col=fear_col,
        trend_score_col=trend_score_col,
        vol_ratio_col=vol_ratio_col,
        pivot_columns=pivot_columns,
        forward_return_col=forward_return_col,
        require_returns=False,
    )


def backtest_positions(
    data: pd.DataFrame,
    *,
    config: PositionConfig | Mapping[str, object] | None = None,
    forecast_columns: Mapping[int | str, str] | None = None,
    price_col: str | None = None,
    daily_vol_col: str | None = None,
    annualized_vol_col: str | None = None,
    regime_col: str | None = None,
    fear_col: str | None = None,
    trend_score_col: str | None = None,
    vol_ratio_col: str | None = None,
    pivot_columns: Mapping[str, str] | None = None,
    forward_return_col: str | None = None,
) -> pd.DataFrame:
    """Run the registered close-t signal -> t-to-t+1 return backtest.

    The returned frame includes at least ``gross_pnl``, ``net_pnl``,
    ``turnover``, ``position``, ``regime`` and ``cap_reason``.  The last row's
    PnL is NaN when it has no future close; this avoids inventing a realised
    outcome for an open shadow position.
    """

    return _construct(
        data,
        config=config,
        forecast_columns=forecast_columns,
        price_col=price_col,
        daily_vol_col=daily_vol_col,
        annualized_vol_col=annualized_vol_col,
        regime_col=regime_col,
        fear_col=fear_col,
        trend_score_col=trend_score_col,
        vol_ratio_col=vol_ratio_col,
        pivot_columns=pivot_columns,
        forward_return_col=forward_return_col,
        require_returns=True,
    )


# Stable, discoverable aliases for callers with different naming conventions.
make_positions = build_positions
generate_positions = build_positions
backtest = backtest_positions


__all__ = [
    "DEFAULT_CONFIG",
    "PositionConfig",
    "backtest",
    "backtest_positions",
    "build_positions",
    "classify_regime",
    "combine_horizon_forecasts",
    "generate_positions",
    "make_positions",
]
