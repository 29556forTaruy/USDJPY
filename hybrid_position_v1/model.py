"""Leakage-controlled daily forecast models for ``hybrid_position_v1``.

The module deliberately has no dependency on scikit-learn.  Ridge fitting,
pre-processing, time-series cross-validation, monthly refitting, and ensemble
weighting are implemented with NumPy and pandas so that every data cut is
auditable.

Forecasts are expressed in the target unit registered in ``protocol.json``::

    log(close[t + h] / close[t]) / (daily_vol[t] * sqrt(h))

Consequently the random-walk forecast is zero.  Converting a forecast back to
a price is intentionally kept as a separate helper, ``forecast_to_price``.
No function in this file reads or changes the registered protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


HORIZONS: tuple[int, ...] = (1, 5, 20)
RIDGE_ALPHAS: tuple[float, ...] = (1.0, 10.0, 100.0)
MODEL_NAMES: tuple[str, ...] = (
    "random_walk",
    "macro_ridge",
    "technical_ridge",
    "hybrid_ridge",
)
PRIOR_WEIGHTS: Mapping[str, float] = {
    "random_walk": 0.40,
    "macro_ridge": 0.20,
    "technical_ridge": 0.20,
    "hybrid_ridge": 0.20,
}

MINIMUM_TRAINING_ROWS = 756
MAXIMUM_TRAINING_ROWS = 1260
INNER_FOLDS = 3
EMBARGO_ROWS = 5
PREDICTION_CLIP_SIGMA = 2.5
ENSEMBLE_ERROR_WINDOW = 252
ENSEMBLE_SHRINKAGE_TO_PRIOR = 0.50
RANDOM_WALK_WEIGHT_FLOOR = 0.35


@dataclass(frozen=True)
class RidgeModel:
    """A fitted ridge model and its training-only transformations."""

    feature_names: tuple[str, ...]
    alpha: float
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    intercept: float
    coefficients: np.ndarray
    target_mean: float
    target_scale: float
    training_rows: int


def _two_dimensional(values: object, *, columns: Sequence[str] | None = None) -> np.ndarray:
    """Return a numeric two-dimensional array without mutating the input."""

    if isinstance(values, pd.DataFrame):
        frame = values if columns is None else values.loc[:, list(columns)]
        array = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("predictors must be a two-dimensional matrix")
    if array.shape[1] == 0:
        raise ValueError("at least one predictor is required")
    return array


def _one_dimensional(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array


def _training_medians(x: np.ndarray) -> np.ndarray:
    """Column medians, using zero only for a wholly missing training column."""

    medians = np.empty(x.shape[1], dtype=float)
    for column in range(x.shape[1]):
        finite = x[np.isfinite(x[:, column]), column]
        medians[column] = float(np.median(finite)) if len(finite) else 0.0
    return medians


def _impute(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, medians.reshape(1, -1))


def fit_ridge(
    x: object,
    y: object,
    alpha: float,
    *,
    feature_names: Sequence[str] | None = None,
) -> RidgeModel:
    """Fit standardized ridge with median imputation and a free intercept.

    Every transformation is estimated from ``x`` only.  The intercept is not
    penalized; this is equivalent to centering both predictors and target and
    applying the ridge penalty only to slope coefficients.
    """

    inferred_names = (
        tuple(str(column) for column in x.columns)
        if isinstance(x, pd.DataFrame)
        else None
    )
    matrix = _two_dimensional(x)
    target = _one_dimensional(y)
    if len(matrix) != len(target):
        raise ValueError("predictors and target have different row counts")
    if not np.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError("alpha must be finite and nonnegative")

    usable = np.isfinite(target)
    matrix = matrix[usable]
    target = target[usable]
    if len(target) == 0:
        raise ValueError("no finite target rows are available")

    medians = _training_medians(matrix)
    imputed = _impute(matrix, medians)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    standardized = (imputed - means) / scales

    target_mean = float(target.mean())
    centered_target = target - target_mean
    penalty = np.eye(standardized.shape[1], dtype=float) * float(alpha)
    gram = standardized.T @ standardized + penalty
    rhs = standardized.T @ centered_target
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs

    target_scale = float(target.std(ddof=0))
    if not np.isfinite(target_scale):
        target_scale = 0.0
    names = tuple(
        feature_names
        or inferred_names
        or (f"x{index}" for index in range(matrix.shape[1]))
    )
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match predictor columns")
    return RidgeModel(
        feature_names=names,
        alpha=float(alpha),
        medians=medians,
        means=means,
        scales=scales,
        intercept=target_mean,
        coefficients=np.asarray(coefficients, dtype=float),
        target_mean=target_mean,
        target_scale=target_scale,
        training_rows=int(len(target)),
    )


def predict_ridge(
    model: RidgeModel,
    x: object,
    *,
    clip_sigma: float | None = PREDICTION_CLIP_SIGMA,
) -> np.ndarray:
    """Predict and clip in the registered sigma-normalized target units.

    The target itself is already divided by contemporaneous volatility times
    ``sqrt(h)``.  Thus a value of 2.5 means an absolute range of ``[-2.5, 2.5]``;
    estimating another volatility from the training target would scale twice.
    """

    if isinstance(x, pd.DataFrame):
        matrix = _two_dimensional(x, columns=model.feature_names)
    elif isinstance(x, pd.Series) and set(model.feature_names).issubset(x.index):
        matrix = _two_dimensional(
            x.loc[list(model.feature_names)].to_numpy(dtype=float).reshape(1, -1)
        )
    else:
        raw = np.asarray(x, dtype=float)
        if raw.ndim == 1 and len(model.feature_names) > 1:
            raw = raw.reshape(1, -1)
        matrix = _two_dimensional(raw)
    if matrix.shape[1] != len(model.feature_names):
        raise ValueError("prediction matrix has the wrong number of columns")
    standardized = (_impute(matrix, model.medians) - model.means) / model.scales
    prediction = model.intercept + standardized @ model.coefficients
    prediction = np.asarray(prediction, dtype=float)
    if clip_sigma is not None:
        clip_sigma = float(clip_sigma)
        if not np.isfinite(clip_sigma) or clip_sigma < 0:
            raise ValueError("clip_sigma must be finite and nonnegative")
        prediction = np.clip(prediction, -clip_sigma, clip_sigma)
    return prediction


def purged_expanding_splits(
    dates: Sequence[object],
    target_end: Sequence[object],
    *,
    n_splits: int = INNER_FOLDS,
    embargo: int = EMBARGO_ROWS,
    minimum_inner_training_rows: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct expanding time-series folds with a purge and row embargo.

    A training observation is included only when both conditions hold:

    * its origin precedes the validation block by at least ``embargo`` rows;
    * its target is realized no later than the first validation date.

    Validation blocks are consecutive and cover the latter half of the input.
    This retains a substantial initial estimation sample while producing the
    three pre-registered expanding folds.
    """

    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce"))
    target_index = pd.DatetimeIndex(pd.to_datetime(list(target_end), errors="coerce"))
    if len(date_index) != len(target_index):
        raise ValueError("dates and target_end have different row counts")
    if date_index.hasnans or target_index.hasnans:
        raise ValueError("dates and target_end must be non-missing")
    if not date_index.is_monotonic_increasing:
        raise ValueError("dates must be sorted in ascending order")
    if n_splits < 1 or embargo < 0:
        raise ValueError("n_splits must be positive and embargo nonnegative")

    count = len(date_index)
    if count < n_splits + embargo + 2:
        return []
    initial = max(1, count // 2)
    if minimum_inner_training_rows is not None:
        initial = max(initial, int(minimum_inner_training_rows) + int(embargo))
    if count - initial < n_splits:
        return []

    validation_blocks = np.array_split(np.arange(initial, count, dtype=int), n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for validation in validation_blocks:
        if len(validation) == 0:
            continue
        validation_start = int(validation[0])
        positional_stop = max(0, validation_start - int(embargo))
        candidates = np.arange(positional_stop, dtype=int)
        realized = target_index[candidates] <= date_index[validation_start]
        training = candidates[np.asarray(realized, dtype=bool)]
        if minimum_inner_training_rows is not None and len(training) < int(
            minimum_inner_training_rows
        ):
            continue
        if len(training):
            splits.append((training, validation))
    return splits


def select_ridge_alpha(
    x: object,
    y: object,
    dates: Sequence[object],
    target_end: Sequence[object],
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    n_splits: int = INNER_FOLDS,
    embargo: int = EMBARGO_ROWS,
    clip_sigma: float | None = PREDICTION_CLIP_SIGMA,
) -> tuple[float, dict[str, object]]:
    """Select alpha by purged expanding CV and the one-standard-error rule.

    The standard error is calculated across the three fold MSEs.  Among all
    alphas within one standard error of the minimum-CV alpha, the largest alpha
    is selected.  This makes the tie-break explicitly conservative.
    """

    matrix = _two_dimensional(x)
    target = _one_dimensional(y)
    if len(matrix) != len(target) or len(target) != len(dates) or len(target) != len(
        target_end
    ):
        raise ValueError("CV inputs have different row counts")
    alpha_grid = sorted({float(value) for value in alphas})
    if not alpha_grid or any((not np.isfinite(value) or value < 0) for value in alpha_grid):
        raise ValueError("alphas must be a nonempty set of finite nonnegative values")

    minimum_inner = max(20, matrix.shape[1] + 5)
    splits = purged_expanding_splits(
        dates,
        target_end,
        n_splits=n_splits,
        embargo=embargo,
        minimum_inner_training_rows=minimum_inner,
    )
    fold_losses: dict[float, list[float]] = {alpha: [] for alpha in alpha_grid}
    for training, validation in splits:
        finite_validation = validation[np.isfinite(target[validation])]
        finite_training = training[np.isfinite(target[training])]
        if len(finite_training) < minimum_inner or len(finite_validation) == 0:
            continue
        for alpha in alpha_grid:
            fitted = fit_ridge(matrix[finite_training], target[finite_training], alpha)
            prediction = predict_ridge(
                fitted,
                matrix[finite_validation],
                clip_sigma=clip_sigma,
            )
            fold_losses[alpha].append(
                float(np.mean(np.square(prediction - target[finite_validation])))
            )

    if not all(fold_losses[alpha] for alpha in alpha_grid):
        selected = float(max(alpha_grid))
        return selected, {
            "selected_alpha": selected,
            "folds": 0,
            "mean_mse": {alpha: float("nan") for alpha in alpha_grid},
            "standard_error": float("nan"),
            "threshold": float("nan"),
            "fold_mse": fold_losses,
        }

    mean_mse = {
        alpha: float(np.mean(np.asarray(losses, dtype=float)))
        for alpha, losses in fold_losses.items()
    }
    best_alpha = min(alpha_grid, key=lambda value: (mean_mse[value], -value))
    best_losses = np.asarray(fold_losses[best_alpha], dtype=float)
    standard_error = (
        float(best_losses.std(ddof=1) / sqrt(len(best_losses)))
        if len(best_losses) > 1
        else 0.0
    )
    threshold = float(mean_mse[best_alpha] + standard_error)
    eligible = [alpha for alpha in alpha_grid if mean_mse[alpha] <= threshold + 1e-15]
    selected = float(max(eligible))
    return selected, {
        "selected_alpha": selected,
        "best_alpha": float(best_alpha),
        "folds": int(len(best_losses)),
        "mean_mse": mean_mse,
        "standard_error": standard_error,
        "threshold": threshold,
        "fold_mse": fold_losses,
    }


# Familiar name for callers coming from accuracy_v2.
choose_ridge_alpha = select_ridge_alpha


def enforce_random_walk_floor(
    weights: Mapping[str, float],
    *,
    floor: float = RANDOM_WALK_WEIGHT_FLOOR,
    models: Sequence[str] = MODEL_NAMES,
) -> dict[str, float]:
    """Normalize nonnegative weights, then impose the random-walk floor."""

    if not 0 <= float(floor) <= 1:
        raise ValueError("random-walk floor must lie in [0, 1]")
    names = tuple(models)
    if "random_walk" not in names:
        raise ValueError("models must include random_walk")
    values = np.asarray(
        [max(0.0, float(weights.get(name, 0.0))) for name in names], dtype=float
    )
    if not np.all(np.isfinite(values)) or values.sum() <= 0:
        values = np.zeros(len(names), dtype=float)
        values[names.index("random_walk")] = 1.0
    else:
        values /= values.sum()

    rw_index = names.index("random_walk")
    if values[rw_index] < floor:
        other_mask = np.arange(len(names)) != rw_index
        other_total = float(values[other_mask].sum())
        values[rw_index] = float(floor)
        if other_total > 0:
            values[other_mask] *= (1.0 - float(floor)) / other_total
        elif len(names) > 1:
            values[other_mask] = (1.0 - float(floor)) / (len(names) - 1)
    values /= values.sum()
    return {name: float(value) for name, value in zip(names, values)}


def ensemble_weights(
    scored_predictions: pd.DataFrame,
    *,
    available_models: Sequence[str] = MODEL_NAMES,
    actual_col: str = "actual",
    prior_weights: Mapping[str, float] = PRIOR_WEIGHTS,
    window: int = ENSEMBLE_ERROR_WINDOW,
    minimum_scored: int = ENSEMBLE_ERROR_WINDOW,
    shrinkage_to_prior: float = ENSEMBLE_SHRINKAGE_TO_PRIOR,
    random_walk_floor: float = RANDOM_WALK_WEIGHT_FLOOR,
) -> dict[str, float]:
    """Return prior/inverse-MSE weights using only already scored forecasts."""

    available = tuple(name for name in MODEL_NAMES if name in set(available_models))
    if "random_walk" not in available:
        available = ("random_walk",) + available
    prior = {
        name: (prior_weights.get(name, 0.0) if name in available else 0.0)
        for name in MODEL_NAMES
    }
    prior = enforce_random_walk_floor(prior, floor=random_walk_floor)

    required = [actual_col, *available]
    if scored_predictions.empty or any(column not in scored_predictions for column in required):
        return prior
    usable = scored_predictions.loc[:, required].apply(pd.to_numeric, errors="coerce")
    usable = usable.replace([np.inf, -np.inf], np.nan).dropna().tail(int(window))
    if len(usable) < int(minimum_scored):
        return prior
    if not 0 <= float(shrinkage_to_prior) <= 1:
        raise ValueError("shrinkage_to_prior must lie in [0, 1]")

    actual = usable[actual_col].to_numpy(dtype=float)
    mse = np.asarray(
        [
            float(np.mean(np.square(usable[name].to_numpy(dtype=float) - actual)))
            for name in available
        ],
        dtype=float,
    )
    inverse = 1.0 / np.maximum(mse, 1e-12)
    performance = inverse / inverse.sum()
    prior_available = np.asarray([prior_weights.get(name, 0.0) for name in available], dtype=float)
    prior_available = np.maximum(prior_available, 0.0)
    if prior_available.sum() <= 0:
        prior_available[:] = 1.0
    prior_available /= prior_available.sum()
    blended = float(shrinkage_to_prior) * prior_available + (
        1.0 - float(shrinkage_to_prior)
    ) * performance
    result = {name: 0.0 for name in MODEL_NAMES}
    result.update({name: float(weight) for name, weight in zip(available, blended)})
    return enforce_random_walk_floor(result, floor=random_walk_floor)


compute_ensemble_weights = ensemble_weights


def add_normalized_targets(
    data: pd.DataFrame,
    *,
    date_col: str = "date",
    price_col: str = "close",
    volatility_col: str = "daily_vol",
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Add ``_target_H`` and ``_target_end_H`` columns to a copy of data."""

    required = [date_col, price_col, volatility_col]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"missing target input columns: {missing}")
    result = data.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    if result[date_col].isna().any():
        raise ValueError("date column contains invalid or missing values")
    if result[date_col].duplicated().any():
        raise ValueError("date column contains duplicates")
    result = result.sort_values(date_col, kind="stable").reset_index(drop=True)
    price = pd.to_numeric(result[price_col], errors="coerce")
    volatility = pd.to_numeric(result[volatility_col], errors="coerce")
    valid_origin = (price > 0) & (volatility > 0) & np.isfinite(price) & np.isfinite(volatility)
    for horizon_value in horizons:
        horizon = int(horizon_value)
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        future_price = price.shift(-horizon)
        future_date = result[date_col].shift(-horizon)
        target = np.log(future_price / price) / (volatility * sqrt(horizon))
        target = target.where(valid_origin & (future_price > 0) & np.isfinite(future_price))
        result[f"_target_{horizon}"] = target.astype(float)
        result[f"_target_end_{horizon}"] = future_date
    return result


def forecast_to_price(
    origin_price: object,
    normalized_forecast: object,
    daily_volatility: object,
    horizon: int,
) -> np.ndarray:
    """Convert the registered normalized-return forecast back to spot units."""

    price = np.asarray(origin_price, dtype=float)
    prediction = np.asarray(normalized_forecast, dtype=float)
    volatility = np.asarray(daily_volatility, dtype=float)
    return price * np.exp(prediction * volatility * sqrt(int(horizon)))


def _default_macro_features(frame: pd.DataFrame, horizon: int) -> list[str]:
    preferred = [f"macro_gap_{horizon}", "macro_gap_h", "macro_available"]
    selected = [column for column in preferred if column in frame]
    if not selected:
        selected = [
            column
            for column in frame.columns
            if str(column).startswith("macro_")
            and not str(column).startswith(("macro_target", "macro_actual"))
        ]
    return list(dict.fromkeys(selected))


def _default_technical_features(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "fear_level",
        "fear_change_5d",
        "fear_coverage",
        "log_price_ma20",
        "log_ma20_ma50",
        "log_ma50_ma125",
        "log_ma125_ma200",
        "pivot_center_distance",
        "pivot_support_distance",
        "pivot_resistance_distance",
        "trend_score",
        "vol_ratio",
    ]
    return [column for column in preferred if column in frame]


def _features_for_horizon(
    specification: Sequence[str] | Mapping[int | str, Sequence[str]] | None,
    frame: pd.DataFrame,
    horizon: int,
    *,
    kind: str,
) -> list[str]:
    if specification is None:
        return (
            _default_macro_features(frame, horizon)
            if kind == "macro"
            else _default_technical_features(frame)
        )
    if isinstance(specification, Mapping):
        chosen = specification.get(horizon, specification.get(str(horizon), ()))
    else:
        chosen = specification
    names = list(dict.fromkeys(str(value) for value in chosen))
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"missing {kind} feature columns for horizon {horizon}: {missing}")
    return names


def _target_column(
    mapping: Mapping[int | str, str] | None,
    horizon: int,
    prefix: str,
) -> str:
    if mapping is None:
        return f"{prefix}{horizon}"
    value = mapping.get(horizon, mapping.get(str(horizon)))
    if value is None:
        raise ValueError(f"no column registered for horizon {horizon}")
    return str(value)


def walk_forward_forecasts(
    data: pd.DataFrame,
    *,
    macro_features: Sequence[str] | Mapping[int | str, Sequence[str]] | None = None,
    technical_features: Sequence[str] | Mapping[int | str, Sequence[str]] | None = None,
    date_col: str = "date",
    price_col: str = "close",
    volatility_col: str = "daily_vol",
    target_columns: Mapping[int | str, str] | None = None,
    target_end_columns: Mapping[int | str, str] | None = None,
    horizons: Sequence[int] = HORIZONS,
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
    maximum_training_rows: int = MAXIMUM_TRAINING_ROWS,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    inner_folds: int = INNER_FOLDS,
    embargo: int = EMBARGO_ROWS,
    prediction_clip_sigma: float = PREDICTION_CLIP_SIGMA,
) -> pd.DataFrame:
    """Run the registered monthly-refit, daily-prediction walk-forward.

    If target column mappings are omitted, targets are generated from ``close``
    and ``daily_vol``.  The return value is one row per date/horizon with wide
    candidate forecasts, ensemble forecast, selected alphas, and model weights.
    A ridge model never sees a row whose ``target_end`` is after its prediction
    date, and its rolling estimation window is capped at 1,260 rows.
    """

    if int(minimum_training_rows) < 1:
        raise ValueError("minimum_training_rows must be positive")
    if int(maximum_training_rows) < int(minimum_training_rows):
        raise ValueError("maximum_training_rows must be at least the minimum")

    frame = data.copy()
    if date_col not in frame:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"missing date column: {date_col}")
        frame.insert(0, date_col, frame.index)
    # A DatetimeIndex commonly carries the same name (``date``) as the column
    # inserted above.  Drop the index before sorting to avoid pandas treating
    # the label as ambiguously both an index level and a column.
    frame = frame.reset_index(drop=True)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    if frame[date_col].isna().any():
        raise ValueError("date column contains invalid or missing values")
    if frame[date_col].duplicated().any():
        raise ValueError("date column contains duplicates")
    frame = frame.sort_values(date_col, kind="stable").reset_index(drop=True)

    resolved_targets = target_columns
    resolved_target_ends = target_end_columns
    if resolved_targets is None:
        inferred_targets: dict[int, str] = {}
        for horizon_value in horizons:
            horizon = int(horizon_value)
            aliases = (f"target_z_{horizon}", f"_target_{horizon}", f"target_{horizon}")
            match = next((name for name in aliases if name in frame), None)
            if match is not None:
                inferred_targets[horizon] = match
        if len(inferred_targets) == len(tuple(horizons)):
            resolved_targets = inferred_targets

    if resolved_targets is not None and resolved_target_ends is None:
        generated_ends: dict[int, str] = {}
        for horizon_value in horizons:
            horizon = int(horizon_value)
            name = f"_target_end_{horizon}"
            frame[name] = frame[date_col].shift(-horizon)
            generated_ends[horizon] = name
        resolved_target_ends = generated_ends
    elif resolved_targets is None and resolved_target_ends is not None:
        raise ValueError("target_end_columns cannot be supplied without target_columns")

    if resolved_targets is None:
        resolved_price = price_col
        resolved_volatility = volatility_col
        if resolved_price not in frame and "price" in frame:
            resolved_price = "price"
        if resolved_volatility not in frame and "effective_daily_vol" in frame:
            resolved_volatility = "effective_daily_vol"
        frame = add_normalized_targets(
            frame,
            date_col=date_col,
            price_col=resolved_price,
            volatility_col=resolved_volatility,
            horizons=horizons,
        )
        resolved_targets = {int(value): f"_target_{int(value)}" for value in horizons}
        resolved_target_ends = {
            int(value): f"_target_end_{int(value)}" for value in horizons
        }

    output_rows: list[dict[str, object]] = []
    for horizon_value in horizons:
        horizon = int(horizon_value)
        target_col = _target_column(resolved_targets, horizon, "_target_")
        target_end_col = _target_column(resolved_target_ends, horizon, "_target_end_")
        missing_targets = [column for column in (target_col, target_end_col) if column not in frame]
        if missing_targets:
            raise ValueError(f"missing target columns for horizon {horizon}: {missing_targets}")

        macro_names = _features_for_horizon(
            macro_features, frame, horizon, kind="macro"
        )
        technical_names = _features_for_horizon(
            technical_features, frame, horizon, kind="technical"
        )
        feature_sets = {
            "macro_ridge": macro_names,
            "technical_ridge": technical_names,
            "hybrid_ridge": list(dict.fromkeys([*macro_names, *technical_names])),
        }
        target = pd.to_numeric(frame[target_col], errors="coerce")
        target_end = pd.to_datetime(frame[target_end_col], errors="coerce")

        active_models: dict[str, RidgeModel] = {}
        active_alphas: dict[str, float] = {}
        active_training_rows: dict[str, int] = {}
        active_cv: dict[str, dict[str, object]] = {}
        previous_month: tuple[int, int] | None = None
        horizon_history: list[dict[str, object]] = []

        for row_index, row in frame.iterrows():
            prediction_date = pd.Timestamp(row[date_col])
            month = (prediction_date.year, prediction_date.month)
            refitted = month != previous_month
            if refitted:
                previous_month = month
                active_models = {}
                active_alphas = {}
                active_training_rows = {}
                active_cv = {}
                finite_target = pd.Series(
                    np.isfinite(target.to_numpy(dtype=float)), index=target.index
                )
                eligible_mask = (
                    (frame[date_col] < prediction_date)
                    & finite_target
                    & target_end.notna()
                    & (target_end <= prediction_date)
                )
                eligible_indices = np.flatnonzero(eligible_mask.to_numpy())
                eligible_indices = eligible_indices[-int(maximum_training_rows) :]
                if len(eligible_indices) >= int(minimum_training_rows):
                    training_dates = frame.loc[eligible_indices, date_col].tolist()
                    training_target_end = target_end.iloc[eligible_indices].tolist()
                    training_y = target.iloc[eligible_indices].to_numpy(dtype=float)
                    for model_name, names in feature_sets.items():
                        if not names:
                            continue
                        training_x = (
                            frame.loc[eligible_indices, names]
                            .apply(pd.to_numeric, errors="coerce")
                            .to_numpy(dtype=float)
                        )
                        selected_alpha, diagnostics = select_ridge_alpha(
                            training_x,
                            training_y,
                            training_dates,
                            training_target_end,
                            alphas=alphas,
                            n_splits=inner_folds,
                            embargo=embargo,
                            clip_sigma=prediction_clip_sigma,
                        )
                        active_models[model_name] = fit_ridge(
                            training_x,
                            training_y,
                            selected_alpha,
                            feature_names=names,
                        )
                        active_alphas[model_name] = selected_alpha
                        active_training_rows[model_name] = len(eligible_indices)
                        active_cv[model_name] = diagnostics

            candidate_predictions: dict[str, float] = {"random_walk": 0.0}
            for model_name, names in feature_sets.items():
                fitted = active_models.get(model_name)
                if fitted is None:
                    candidate_predictions[model_name] = float("nan")
                    continue
                row_x = (
                    pd.DataFrame([row.loc[names]])
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=float)
                )
                candidate_predictions[model_name] = float(
                    predict_ridge(
                        fitted,
                        row_x,
                        clip_sigma=prediction_clip_sigma,
                    )[0]
                )

            available = [
                name for name in MODEL_NAMES if np.isfinite(candidate_predictions.get(name, np.nan))
            ]
            scored = pd.DataFrame(
                [
                    historical
                    for historical in horizon_history
                    if pd.notna(historical.get("actual"))
                    and pd.notna(historical.get("target_end"))
                    and pd.Timestamp(historical["target_end"]) <= prediction_date
                ]
            )
            weights = ensemble_weights(scored, available_models=available)
            ensemble = float(
                sum(
                    weights[name] * candidate_predictions[name]
                    for name in available
                )
            )
            output: dict[str, object] = {
                "date": prediction_date,
                "horizon": horizon,
                "target_end": target_end.iloc[row_index],
                "actual": float(target.iloc[row_index])
                if pd.notna(target.iloc[row_index])
                else float("nan"),
                **candidate_predictions,
                "ensemble": ensemble,
                "model_refitted": bool(refitted),
            }
            for model_name in MODEL_NAMES:
                output[f"weight_{model_name}"] = float(weights[model_name])
            for model_name in MODEL_NAMES[1:]:
                output[f"alpha_{model_name}"] = float(
                    active_alphas.get(model_name, np.nan)
                )
                output[f"training_rows_{model_name}"] = int(
                    active_training_rows.get(model_name, 0)
                )
                output[f"cv_folds_{model_name}"] = int(
                    active_cv.get(model_name, {}).get("folds", 0)
                )
            output_rows.append(output)
            horizon_history.append(output)

    columns = [
        "date",
        "horizon",
        "target_end",
        "actual",
        *MODEL_NAMES,
        "ensemble",
        *(f"alpha_{name}" for name in MODEL_NAMES[1:]),
        *(f"weight_{name}" for name in MODEL_NAMES),
        *(f"training_rows_{name}" for name in MODEL_NAMES[1:]),
        *(f"cv_folds_{name}" for name in MODEL_NAMES[1:]),
        "model_refitted",
    ]
    result = pd.DataFrame(output_rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    return result.loc[:, columns].sort_values(["date", "horizon"], kind="stable").reset_index(
        drop=True
    )


# Short aliases make the public entry point easy to discover in notebooks.
walk_forward_predict = walk_forward_forecasts
run_walk_forward = walk_forward_forecasts


__all__ = [
    "EMBARGO_ROWS",
    "ENSEMBLE_ERROR_WINDOW",
    "ENSEMBLE_SHRINKAGE_TO_PRIOR",
    "HORIZONS",
    "INNER_FOLDS",
    "MAXIMUM_TRAINING_ROWS",
    "MINIMUM_TRAINING_ROWS",
    "MODEL_NAMES",
    "PREDICTION_CLIP_SIGMA",
    "PRIOR_WEIGHTS",
    "RANDOM_WALK_WEIGHT_FLOOR",
    "RIDGE_ALPHAS",
    "RidgeModel",
    "add_normalized_targets",
    "choose_ridge_alpha",
    "compute_ensemble_weights",
    "enforce_random_walk_floor",
    "ensemble_weights",
    "fit_ridge",
    "forecast_to_price",
    "predict_ridge",
    "purged_expanding_splits",
    "run_walk_forward",
    "select_ridge_alpha",
    "walk_forward_forecasts",
    "walk_forward_predict",
]
