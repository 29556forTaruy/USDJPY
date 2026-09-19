#!/usr/bin/env python3
"""Run the pre-registered hybrid USD/JPY forecast and position study.

The registered protocol is read, never rewritten.  Results are walk-forward:
every ridge fit is limited to targets that had resolved by its prediction date.
Historical results can reject the design, but can never promote it to live use.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_position_v1.data import load_latest_market, sha256_file
from hybrid_position_v1.features import build_features
from hybrid_position_v1.metrics import (
    holm_adjust,
    newey_west_dm_pvalue,
    prediction_metrics,
    strategy_metrics,
)
from hybrid_position_v1.model import forecast_to_price, walk_forward_forecasts
from hybrid_position_v1.position import backtest_positions


MODULE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = MODULE_DIR / "protocol.json"
RESULTS_DIR = MODULE_DIR / "results"
MODEL_NAMES = (
    "random_walk",
    "macro_ridge",
    "technical_ridge",
    "hybrid_ridge",
    "ensemble",
)
TECHNICAL_FEATURES = (
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
)


def read_protocol() -> dict[str, Any]:
    with PROTOCOL_PATH.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_id") != "usdjpy-hybrid-position-v1":
        raise ValueError("unexpected hybrid protocol")
    if protocol.get("registered_before_first_backtest") is not True:
        raise ValueError("protocol was not registered before the first backtest")
    return protocol


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(_json_safe(dict(payload)), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _model_kwargs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    prediction = protocol["prediction"]
    return {
        "horizons": tuple(int(value) for value in prediction["horizons_business_days"]),
        "minimum_training_rows": int(prediction["minimum_training_rows"]),
        "maximum_training_rows": int(prediction["maximum_training_rows"]),
        "alphas": tuple(float(value) for value in prediction["ridge_alphas"]),
        "inner_folds": int(prediction["inner_folds"]),
        "embargo": int(prediction["embargo_business_days"]),
        "prediction_clip_sigma": float(prediction["prediction_clip_sigma"]),
    }


def run_forecasts(
    features: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    macro_features: Sequence[str] | Mapping[int | str, Sequence[str]] | None = None,
    technical_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    return walk_forward_forecasts(
        features,
        macro_features=macro_features,
        technical_features=technical_features,
        **_model_kwargs(protocol),
    )


def _wide_predictions(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    return (
        predictions.pivot(index="date", columns="horizon", values=model)
        .rename(columns=lambda value: f"pred_{int(value)}")
        .sort_index()
    )


def position_backtest(
    market: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    model: str = "ensemble",
) -> pd.DataFrame:
    inputs = market[["price"]].join(features).join(_wide_predictions(predictions, model))
    inputs["daily_vol"] = inputs["effective_daily_vol"]
    inputs["annualized_vol"] = inputs["rv20_annualized"]
    return backtest_positions(inputs, config=protocol)


def _common_sample(group: pd.DataFrame) -> pd.DataFrame:
    # All learned candidates begin together.  Excluding the pre-training period
    # prevents the ensemble's zero forecast from masquerading as an evaluated
    # model before any ridge estimate existed.
    return group.loc[group["hybrid_ridge"].notna() & group["actual"].notna()].copy()


def prediction_evaluation(
    predictions: pd.DataFrame, protocol: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lag_map = {
        int(key): int(value)
        for key, value in protocol["backtest"]["newey_west_lag_by_horizon"].items()
    }
    rows: list[dict[str, Any]] = []
    ensemble_pvalues: dict[str, float] = {}
    halves: dict[str, Any] = {}
    for horizon, raw_group in predictions.groupby("horizon", sort=True):
        horizon = int(horizon)
        group = _common_sample(raw_group)
        if group.empty:
            continue
        for model in MODEL_NAMES:
            metrics = prediction_metrics(group["actual"], group[model])
            pvalue = newey_west_dm_pvalue(
                group["actual"], group[model], lag_map[horizon]
            )
            rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "sample_start": str(pd.Timestamp(group["date"].min()).date()),
                    "sample_end": str(pd.Timestamp(group["date"].max()).date()),
                    **metrics,
                    "dm_pvalue_vs_rw": pvalue,
                }
            )
            if model == "ensemble":
                ensemble_pvalues[str(horizon)] = pvalue

        split = len(group) // 2
        halves[str(horizon)] = {}
        for label, half in (("first", group.iloc[:split]), ("second", group.iloc[split:])):
            halves[str(horizon)][label] = prediction_metrics(
                half["actual"], half["ensemble"]
            )
    adjusted = holm_adjust(ensemble_pvalues)
    for row in rows:
        if row["model"] == "ensemble":
            row["holm_adjusted_dm_pvalue"] = adjusted.get(str(row["horizon"]), math.nan)
        else:
            row["holm_adjusted_dm_pvalue"] = math.nan
    return pd.DataFrame(rows), {"halves": halves, "holm_adjusted": adjusted}


def _closed_trades(position: pd.Series) -> int:
    values = pd.to_numeric(position, errors="coerce").fillna(0.0)
    sign = np.sign(values)
    previous = sign.shift(1, fill_value=0.0)
    # A transition away from a non-zero sign closes the prior trade.  A final
    # open position is deliberately not counted as closed.
    return int(((previous != 0.0) & (sign != previous)).sum())


def _simple_comparator(
    market: pd.DataFrame,
    features: pd.DataFrame,
    raw_position: pd.Series,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    price = pd.to_numeric(market["price"], errors="coerce")
    desired = pd.to_numeric(raw_position, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    positions: list[float] = []
    previous = 0.0
    for target in desired:
        current = float(np.clip(float(target), previous - 0.25, previous + 0.25))
        positions.append(current)
        previous = current
    frame = pd.DataFrame(index=market.index)
    frame["position"] = positions
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    forward = price.shift(-1) / price - 1.0
    frame["gross_return"] = frame["position"] * forward
    frame["net_return"] = (
        frame["gross_return"] - frame["turnover"] * one_way_cost_bps / 10_000.0
    )
    return frame


def strategy_evaluation(
    market: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    first_dates = (
        predictions.loc[predictions["hybrid_ridge"].notna()]
        .groupby("horizon")["date"]
        .min()
    )
    evaluation_start = pd.Timestamp(first_dates.max())
    rows: list[dict[str, Any]] = []
    ensemble_positions = position_backtest(market, features, predictions, protocol)

    for model in MODEL_NAMES:
        model_positions = (
            ensemble_positions
            if model == "ensemble"
            else position_backtest(market, features, predictions, protocol, model=model)
        )
        sample = model_positions.loc[model_positions.index >= evaluation_start]
        rows.append(
            {
                "strategy": model,
                "sample_start": str(evaluation_start.date()),
                "sample_end": str(sample.index.max().date()),
                "closed_trades": _closed_trades(sample["position"]),
                **strategy_metrics(sample),
            }
        )

    annualized = features["rv20_annualized"].replace(0.0, np.nan)
    vol_scaled = (float(protocol["position"]["annual_vol_target"]) / annualized).clip(upper=1.0)
    ma_cross_direction = np.sign(features["ma20"] - features["ma125"])
    comparators = {
        "ma_cross": _simple_comparator(
            market,
            features,
            ma_cross_direction * vol_scaled,
            float(protocol["position"]["base_one_way_cost_bps"]),
        ),
        "risk_matched_always_long": _simple_comparator(
            market,
            features,
            vol_scaled,
            float(protocol["position"]["base_one_way_cost_bps"]),
        ),
    }
    for name, frame in comparators.items():
        sample = frame.loc[frame.index >= evaluation_start]
        rows.append(
            {
                "strategy": name,
                "sample_start": str(evaluation_start.date()),
                "sample_end": str(sample.index.max().date()),
                "closed_trades": _closed_trades(sample["position"]),
                **strategy_metrics(sample),
            }
        )

    stress_protocol = copy.deepcopy(dict(protocol))
    stress_protocol["position"] = dict(stress_protocol["position"])
    stress_protocol["position"]["base_one_way_cost_bps"] = float(
        protocol["position"]["stress_one_way_cost_bps"]
    )
    stress_positions = position_backtest(
        market, features, predictions, stress_protocol
    )
    stress_metrics = strategy_metrics(
        stress_positions.loc[stress_positions.index >= evaluation_start]
    )
    return (
        ensemble_positions,
        pd.DataFrame(rows),
        {
            "evaluation_start": str(evaluation_start.date()),
            "stress_cost_bps": float(protocol["position"]["stress_one_way_cost_bps"]),
            "stress_metrics": stress_metrics,
        },
    )


def run_ablations(
    market: pd.DataFrame,
    features: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    fear = {"fear_level", "fear_change_5d", "fear_coverage"}
    moving_average = {
        "log_price_ma20",
        "log_ma20_ma50",
        "log_ma50_ma125",
        "log_ma125_ma200",
        "trend_score",
    }
    pivot = {
        "pivot_center_distance",
        "pivot_support_distance",
        "pivot_resistance_distance",
    }
    variants: dict[str, tuple[list[str], Mapping[int, Sequence[str]] | None]] = {
        "without_fng": ([name for name in TECHNICAL_FEATURES if name not in fear], None),
        "without_ma": ([name for name in TECHNICAL_FEATURES if name not in moving_average], None),
        "without_pivot": ([name for name in TECHNICAL_FEATURES if name not in pivot], None),
        "without_macro": (list(TECHNICAL_FEATURES), {1: (), 5: (), 20: ()}),
    }
    rows: list[dict[str, Any]] = []
    lag_map = {
        int(key): int(value)
        for key, value in protocol["backtest"]["newey_west_lag_by_horizon"].items()
    }
    for name, (technical, macro) in variants.items():
        forecasts = run_forecasts(
            features,
            protocol,
            macro_features=macro,
            technical_features=technical,
        )
        for horizon, raw_group in forecasts.groupby("horizon", sort=True):
            group = _common_sample(raw_group)
            metrics = prediction_metrics(group["actual"], group["ensemble"])
            rows.append(
                {
                    "ablation": name,
                    "horizon": int(horizon),
                    **metrics,
                    "dm_pvalue_vs_rw": newey_west_dm_pvalue(
                        group["actual"], group["ensemble"], lag_map[int(horizon)]
                    ),
                }
            )
        positions = position_backtest(market, features, forecasts, protocol)
        start = forecasts.loc[forecasts["hybrid_ridge"].notna(), "date"].min()
        position_metrics = strategy_metrics(positions.loc[positions.index >= start])
        rows.append(
            {
                "ablation": name,
                "horizon": 0,
                "strategy_sharpe": position_metrics.get("sharpe"),
                "strategy_cumulative_return": position_metrics.get("cumulative_return"),
                "strategy_max_drawdown": position_metrics.get("max_drawdown"),
            }
        )
    return pd.DataFrame(rows)


def current_signal(
    market: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    positions: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    date = pd.Timestamp(market["price"].last_valid_index())
    spot = float(market.loc[date, "price"])
    vol = float(features.loc[date, "effective_daily_vol"])
    current_predictions = predictions.loc[predictions["date"].eq(date)].set_index("horizon")
    normalized = {
        str(horizon): float(current_predictions.loc[horizon, "ensemble"])
        for horizon in (1, 5, 20)
    }
    prices = {
        str(horizon): float(
            forecast_to_price(spot, normalized[str(horizon)], vol, horizon)
        )
        for horizon in (1, 5, 20)
    }
    latest_position = positions.loc[date]
    anchor = market.loc[date, "macro_anchor"]
    source_fng = market.loc[date, "source_fear_greed"]
    audited_fng = features.loc[date, "audited_market_fng"]
    return {
        "schema_version": 1,
        "market_date": str(date.date()),
        "earliest_execution": "next tradable time after the completed market-date close",
        "spot": spot,
        "effective_daily_vol": vol,
        "annualized_realized_vol": float(features.loc[date, "rv20_annualized"]),
        "normalized_forecasts": normalized,
        "forecast_prices": prices,
        "forecast_change_pct": {
            key: 100.0 * (value / spot - 1.0) for key, value in prices.items()
        },
        "target_position": float(latest_position["position"]),
        "signal_sigma_after_cost": float(latest_position["signal_sigma"]),
        "agreeing_horizons": int(latest_position["agreeing_horizons"]),
        "regime": str(latest_position["regime"]),
        "cap_reason": str(latest_position["cap_reason"]),
        "audited_market_fng": float(audited_fng),
        "source_headline_fng_diagnostic": float(source_fng) if pd.notna(source_fng) else None,
        "fear_coverage": float(features.loc[date, "fear_coverage"]),
        "macro_anchor": float(anchor) if pd.notna(anchor) else None,
        "macro_divergence_pct": 100.0 * (float(anchor) / spot - 1.0) if pd.notna(anchor) else None,
        "prior_year_pivots": {
            key: float(features.loc[date, f"pivot_{key}"])
            for key in ("p", "r1", "s1", "r2", "s2")
        },
        "source_bundle_sha256": manifest["bundle_sha256"],
        "source_commit": manifest["source_commit"],
        "warning": "research-only shadow signal; no order is authorized",
    }


def promotion_decision(
    prediction_table: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    strategy_table: pd.DataFrame,
    strategy_diagnostics: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rules = protocol["promotion"]
    ensemble = prediction_table.loc[prediction_table["model"].eq("ensemble")].set_index(
        "horizon"
    )
    strategy = strategy_table.loc[strategy_table["strategy"].eq("ensemble")].iloc[0]
    criteria: dict[str, bool] = {}
    for horizon in (5, 20):
        criteria[f"h{horizon}_rmse_improvement"] = bool(
            ensemble.loc[horizon, "rmse_improvement_pct_vs_rw"]
            >= float(rules["rmse_improvement_min_pct"][str(horizon)])
        )
        criteria[f"h{horizon}_mae_improvement"] = bool(
            ensemble.loc[horizon, "mae_improvement_pct_vs_rw"] > 0.0
        )
    criteria["h1_within_deterioration_limit"] = bool(
        ensemble.loc[1, "rmse_improvement_pct_vs_rw"]
        >= -float(rules["one_day_rmse_max_deterioration_pct"])
    )
    half_values = diagnostics["halves"]
    criteria["both_halves_nonnegative_improvement"] = all(
        float(half_values[str(horizon)][half]["rmse_improvement_pct_vs_rw"]) >= 0.0
        for horizon in (5, 20)
        for half in ("first", "second")
    )
    criteria["holm_dm_and_positive_effect"] = all(
        float(ensemble.loc[horizon, "holm_adjusted_dm_pvalue"])
        <= float(rules["holm_adjusted_dm_p_max"])
        and float(ensemble.loc[horizon, "rmse_improvement_pct_vs_rw"]) > 0.0
        for horizon in (5, 20)
    )
    criteria["net_sharpe"] = bool(strategy["sharpe"] >= float(rules["net_sharpe_min"]))
    criteria["profit_factor"] = bool(
        strategy["profit_factor"] >= float(rules["profit_factor_min"])
    )
    criteria["max_drawdown"] = bool(
        strategy["max_drawdown"] <= float(rules["max_drawdown_max"])
    )
    criteria["stress_cost_positive"] = bool(
        strategy_diagnostics["stress_metrics"]["cumulative_return"]
        >= float(rules["stress_cost_cumulative_pnl_min"])
    )
    criteria["minimum_closed_trades"] = bool(
        strategy["closed_trades"] >= int(rules["minimum_closed_trades"])
    )
    # Historical tests are not permitted to satisfy the forward requirement.
    criteria["minimum_forward_days"] = False
    criteria["minimum_target_quarters"] = False
    promoted = bool(all(criteria.values())) and bool(
        rules["historical_backtest_can_auto_promote"]
    )
    return {
        "promoted": promoted,
        "decision": "KEEP_SHADOW_ONLY" if not promoted else "ELIGIBLE_FOR_HUMAN_REVIEW",
        "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
        "historical_backtest_can_auto_promote": bool(
            rules["historical_backtest_can_auto_promote"]
        ),
        "human_review_required": bool(rules["human_review_required"]),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def render_report(summary: Mapping[str, Any], prediction_table: pd.DataFrame, strategy_table: pd.DataFrame) -> str:
    signal = summary["current_signal"]
    ensemble = prediction_table.loc[prediction_table["model"].eq("ensemble")].set_index("horizon")
    strategy = strategy_table.loc[strategy_table["strategy"].eq("ensemble")].iloc[0]
    pivot = signal["prior_year_pivots"]
    lines = [
        "# USD/JPY ハイブリッド予測・ポジションモデル V1",
        "",
        "## 結論",
        "",
        f"このモデルは**シャドー運用のまま維持**します。過去検証だけで実運用へ昇格する設計ではなく、今回も昇格条件を満たしませんでした。直近 {signal['market_date']} の目標ポジションは **{_fmt(signal['target_position'], 3)}**（+1が最大ロング、-1が最大ショート）です。",
        "",
        "## 直近シグナル",
        "",
        f"終値は {_fmt(signal['spot'], 3)} 円、監査版F&Gは {_fmt(signal['audited_market_fng'], 1)}、コスト控除後の合成シグナルは {_fmt(signal['signal_sigma_after_cost'], 3)}σ です。3期間中 {signal['agreeing_horizons']} 期間が同方向ですが、判定は `{signal['cap_reason']}`、相場状態は `{signal['regime']}` です。",
        "",
        f"予測レートは1日 {_fmt(signal['forecast_prices']['1'], 3)} 円、5日 {_fmt(signal['forecast_prices']['5'], 3)} 円、20日 {_fmt(signal['forecast_prices']['20'], 3)} 円です。四半期マクロ・アンカーは {_fmt(signal['macro_anchor'], 3)} 円、乖離は {_fmt(signal['macro_divergence_pct'], 2)}% です。",
        "",
        f"前年OHLCから固定した年足ピボットは P={_fmt(pivot['p'], 3)}、R1={_fmt(pivot['r1'], 3)}、S1={_fmt(pivot['s1'], 3)}、R2={_fmt(pivot['r2'], 3)}、S2={_fmt(pivot['s2'], 3)} です。",
        "",
        "## 予測精度（学習開始後の共通期間）",
        "",
        "| 期間 | 件数 | RMSE改善率 vs 変化なし | MAE改善率 | 方向一致率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in (1, 5, 20):
        row = ensemble.loc[horizon]
        lines.append(
            f"| {horizon}日 | {int(row['n'])} | {_fmt(row['rmse_improvement_pct_vs_rw'])}% | {_fmt(row['mae_improvement_pct_vs_rw'])}% | {_fmt(100.0 * row['direction_accuracy'], 1)}% |"
        )
    lines.extend(
        [
            "",
            "改善率がマイナスなら、単純な『変化なし』予測より悪かったことを意味します。この結果を見て係数や閾値を後付けで変更していません。",
            "",
            "## ポジション検証",
            "",
            f"評価開始日は {strategy['sample_start']}、累積損益は {_fmt(100.0 * strategy['cumulative_return'])}%、Sharpeは {_fmt(strategy['sharpe'])}、最大ドローダウンは {_fmt(100.0 * strategy['max_drawdown'])}%、Profit Factorは {_fmt(strategy['profit_factor'])}、クローズ済み取引数は {int(strategy['closed_trades'])} でした。取引コスト控除後です。",
            "",
            "## 初心者向けの読み方",
            "",
            "四半期モデルを『遠くの基準点』、F&Gと移動平均を『現在の風向き』、年足ピボットを『近くの壁』として扱います。1・5・20日の予測が2つ以上同じ方向を向き、コストを引いた強さが0.20σを超えたときだけ新規ポジションを検討します。その後も年率5%のボラ目標、相場状態、ピボット、日次変化幅、損失後・ドローダウン制限でサイズを縮めます。",
            "",
            "## 重要な制約",
            "",
            "- F&Gの公表ヘッドラインは診断表示だけです。COTは銘柄混在と公表日ラグ、金利差は改訂履歴とマクロ重複、breadthは欠損バイアスのため監査版から除外しました。",
            "- 履歴は当時入手可能だった全ビンテージではなく、現在ファイルから再構成した履歴です。厳密なリアルタイム成績ではありません。",
            "- マクロ値は四半期モデルの予測アンカーであり、裁定可能な公正価値ではありません。",
            "- シグナルは当日終値確定後に作り、最短でも次の取引可能時点から適用します。自動発注は実装していません。",
            "- 研究用途であり、投資助言ではありません。",
            "",
        ]
    )
    return "\n".join(lines)


def run(*, skip_ablations: bool = False) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    protocol = read_protocol()
    market, manifest = load_latest_market()
    features = build_features(market, protocol)
    predictions = run_forecasts(features, protocol)
    prediction_table, prediction_diagnostics = prediction_evaluation(predictions, protocol)
    positions, strategy_table, strategy_diagnostics = strategy_evaluation(
        market, features, predictions, protocol
    )
    signal = current_signal(market, features, predictions, positions, manifest)
    decision = promotion_decision(
        prediction_table,
        prediction_diagnostics,
        strategy_table,
        strategy_diagnostics,
        protocol,
    )

    ablations_path = RESULTS_DIR / "ablation_metrics.csv"
    if not skip_ablations:
        ablations = run_ablations(market, features, protocol)
        ablations.to_csv(ablations_path, index=False)
    elif not ablations_path.exists():
        pd.DataFrame(columns=["ablation", "horizon"]).to_csv(ablations_path, index=False)

    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    positions.to_csv(RESULTS_DIR / "positions.csv", index=True, index_label="date")
    prediction_table.to_csv(RESULTS_DIR / "prediction_metrics.csv", index=False)
    strategy_table.to_csv(RESULTS_DIR / "strategy_metrics.csv", index=False)
    features.to_csv(RESULTS_DIR / "features.csv", index=True, index_label="date")
    atomic_json(RESULTS_DIR / "current_signal.json", signal)

    core_files = [
        "data.py",
        "features.py",
        "model.py",
        "position.py",
        "metrics.py",
        "run_backtest.py",
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "implementation_sha256": {
            name: sha256_file(MODULE_DIR / name) for name in core_files
        },
        "source_manifest": manifest,
        "backtest": {
            "market_start": str(market.index.min().date()),
            "market_end": str(market.index.max().date()),
            "rows": int(len(market)),
            "prediction_diagnostics": prediction_diagnostics,
            "strategy_diagnostics": strategy_diagnostics,
            "ablations_executed": not skip_ablations,
        },
        "current_signal": signal,
        "promotion": decision,
        "warning": "shadow research only; no live order authorization",
    }
    atomic_json(RESULTS_DIR / "summary.json", summary)
    (RESULTS_DIR / "REPORT.md").write_text(
        render_report(summary, prediction_table, strategy_table), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="daily refresh mode; preserve an existing ablation table",
    )
    args = parser.parse_args()
    summary = run(skip_ablations=args.skip_ablations)
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
