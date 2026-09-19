"""Evaluation metrics fixed by the hybrid-position protocol."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _finite_pairs(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def prediction_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int]:
    y, p = _finite_pairs(actual, predicted)
    if not len(y):
        return {"n": 0}
    error = p - y
    rw_error = -y
    rmse = float(np.sqrt(np.mean(error**2)))
    rw_rmse = float(np.sqrt(np.mean(rw_error**2)))
    mae = float(np.mean(np.abs(error)))
    rw_mae = float(np.mean(np.abs(rw_error)))
    nonzero = y != 0
    direction = float(np.mean(np.sign(p[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else math.nan
    return {
        "n": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "bias": float(np.mean(error)),
        "direction_accuracy": direction,
        "rw_rmse": rw_rmse,
        "rw_mae": rw_mae,
        "rmse_improvement_pct_vs_rw": 100.0 * (1.0 - rmse / rw_rmse) if rw_rmse else math.nan,
        "mae_improvement_pct_vs_rw": 100.0 * (1.0 - mae / rw_mae) if rw_mae else math.nan,
    }


def newey_west_dm_pvalue(
    actual: Iterable[float], predicted: Iterable[float], lag: int
) -> float:
    """Two-sided normal-approximation DM p-value versus zero-return RW.

    Positive loss differential means the candidate has lower squared error.
    """

    y, p = _finite_pairs(actual, predicted)
    if len(y) < max(20, lag + 5):
        return math.nan
    differential = y**2 - (y - p) ** 2
    centered = differential - differential.mean()
    n = len(centered)
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for k in range(1, min(lag, n - 2) + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        long_run += 2.0 * weight * gamma
    variance_mean = max(long_run / n, 0.0)
    if variance_mean <= 0:
        return math.nan
    statistic = float(differential.mean() / math.sqrt(variance_mean))
    return float(math.erfc(abs(statistic) / math.sqrt(2.0)))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    finite = sorted(
        ((name, float(value)) for name, value in pvalues.items() if math.isfinite(value)),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {name: math.nan for name in pvalues}
    running = 0.0
    total = len(finite)
    for rank, (name, value) in enumerate(finite):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def maximum_drawdown(net_returns: pd.Series) -> float:
    clean = pd.to_numeric(net_returns, errors="coerce").fillna(0.0)
    # Include the starting capital explicitly.  Without the leading 1.0, a
    # loss on the first observation incorrectly becomes the initial peak and
    # is reported as zero drawdown.
    equity = pd.concat(
        [pd.Series([1.0], dtype=float), (1.0 + clean).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity / equity.cummax() - 1.0
    return float(-drawdown.min()) if len(drawdown) else math.nan


def strategy_metrics(frame: pd.DataFrame, return_column: str = "net_return") -> dict[str, float | int]:
    returns = pd.to_numeric(frame[return_column], errors="coerce").dropna()
    if returns.empty:
        return {"n": 0}
    mean = float(returns.mean())
    vol = float(returns.std(ddof=0))
    annual_return = float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)
    annual_vol = vol * math.sqrt(252.0)
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0)) * math.sqrt(252.0) if len(downside) else 0.0
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    turnover = (
        pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0)
        if "turnover" in frame
        else pd.Series(0.0, index=frame.index, dtype=float)
    )
    position = (
        pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
        if "position" in frame
        else pd.Series(0.0, index=frame.index, dtype=float)
    )
    return {
        "n": int(len(returns)),
        "cumulative_return": float((1.0 + returns).prod() - 1.0),
        "annualized_return": annual_return,
        "annualized_vol": annual_vol,
        "sharpe": mean / vol * math.sqrt(252.0) if vol else math.nan,
        "sortino": annual_return / downside_vol if downside_vol else math.nan,
        "max_drawdown": maximum_drawdown(returns),
        "profit_factor": gains / losses if losses else math.inf,
        "positive_day_fraction": float((returns > 0).mean()),
        "turnover": float(turnover.sum()),
        "mean_abs_position": float(position.abs().mean()),
        "max_abs_position": float(position.abs().max()),
    }
