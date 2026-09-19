#!/usr/bin/env python3
"""Register, issue, and verify append-only hybrid shadow signals.

This command never downloads data and never places an order.  It accepts only
the content-addressed source bundle and backtest artifacts produced locally.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_position_v1.data import (
    load_latest_market,
    read_json,
    sha256_file,
    source_freshness,
    verify_latest_source,
)
from hybrid_position_v1.ledger import (
    DEFAULT_DB,
    DEFAULT_HEAD,
    append_event,
    iter_events,
    latest_event,
    verify as verify_ledger,
)


MODULE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = MODULE_DIR / "protocol.json"
RESULTS_DIR = MODULE_DIR / "results"
STATUS_PATH = MODULE_DIR / "state" / "shadow_status.json"
TOKYO = ZoneInfo("Asia/Tokyo")
RESULT_FILES = (
    "predictions.csv",
    "positions.csv",
    "prediction_metrics.csv",
    "strategy_metrics.csv",
    "ablation_metrics.csv",
    "features.csv",
    "current_signal.json",
    "summary.json",
    "REPORT.md",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _protocol() -> dict[str, Any]:
    value = read_json(PROTOCOL_PATH)
    if value.get("protocol_id") != "usdjpy-hybrid-position-v1":
        raise ValueError("unexpected hybrid protocol")
    return value


def result_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in RESULT_FILES:
        path = RESULTS_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"backtest artifact is missing: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def verify_backtest_artifacts() -> dict[str, Any]:
    protocol = _protocol()
    source = verify_latest_source()
    summary = read_json(RESULTS_DIR / "summary.json")
    signal = read_json(RESULTS_DIR / "current_signal.json")
    current_protocol_hash = sha256_file(PROTOCOL_PATH)
    if summary.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("backtest protocol id mismatch")
    if summary.get("protocol_sha256") != current_protocol_hash:
        raise ValueError("protocol changed after the backtest")
    if summary.get("source_manifest", {}).get("bundle_sha256") != source["manifest"].get(
        "bundle_sha256"
    ):
        raise ValueError("backtest does not use the latest verified source bundle")
    if signal.get("source_bundle_sha256") != source["manifest"].get("bundle_sha256"):
        raise ValueError("current signal source bundle mismatch")
    if signal != summary.get("current_signal"):
        raise ValueError("standalone current signal differs from summary")
    for name, expected in summary.get("implementation_sha256", {}).items():
        path = MODULE_DIR / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"implementation changed after the backtest: {name}")
    return {
        "protocol": protocol,
        "protocol_sha256": current_protocol_hash,
        "source": source,
        "summary": summary,
        "signal": signal,
        "result_sha256": result_hashes(),
    }


def _events(event_type: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not DEFAULT_DB.exists():
        return output
    for row in iter_events():
        if row["event_type"] == event_type:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            output.append(item)
    return output


def register_artifacts() -> dict[str, Any]:
    verified = verify_backtest_artifacts()
    protocol = verified["protocol"]
    manifest = verified["source"]["manifest"]
    summary = verified["summary"]
    market_date = str(verified["signal"]["market_date"])

    protocol_event = append_event(
        "PROTOCOL_REGISTERED_HYBRID_V1",
        str(protocol["registered_date"]),
        {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": verified["protocol_sha256"],
            "registered_before_first_backtest": bool(
                protocol["registered_before_first_backtest"]
            ),
            "mode": protocol["execution"]["mode"],
            "automatic_orders": False,
        },
    )
    registrations = _events("PROTOCOL_REGISTERED_HYBRID_V1")
    registered_hashes = {item["payload"]["protocol_sha256"] for item in registrations}
    if registered_hashes != {verified["protocol_sha256"]}:
        raise ValueError("multiple or changed protocol registrations detected")

    source_event = append_event(
        "SOURCE_BUNDLE_REGISTERED_HYBRID_V1",
        str(manifest["last_date"]),
        {
            "bundle_sha256": manifest["bundle_sha256"],
            "source_remote": manifest["source_remote"],
            "source_commit": manifest["source_commit"],
            "market_csv_sha256": manifest["market_csv_sha256"],
            "protocol_sha256": manifest["protocol_sha256"],
            "historical_quality": manifest["historical_quality"],
        },
    )
    backtest_event = append_event(
        "BACKTEST_REGISTERED_HYBRID_V1",
        market_date,
        {
            "source_bundle_sha256": manifest["bundle_sha256"],
            "protocol_sha256": verified["protocol_sha256"],
            "result_sha256": verified["result_sha256"],
            "generated_at_utc": summary["generated_at_utc"],
            "promotion": summary["promotion"],
        },
    )
    return {
        "verified": verified,
        "protocol_event": protocol_event,
        "source_event": source_event,
        "backtest_event": backtest_event,
    }


def issue() -> dict[str, Any]:
    registered = register_artifacts()
    verified = registered["verified"]
    signal = verified["signal"]
    market_date = str(signal["market_date"])
    common = {
        "market_date": market_date,
        "earliest_execution": signal["earliest_execution"],
        "spot": signal["spot"],
        "protocol_sha256": verified["protocol_sha256"],
        "source_bundle_sha256": signal["source_bundle_sha256"],
        "results_summary_sha256": verified["result_sha256"]["summary.json"],
        "automatic_orders": False,
    }
    forecast_payload = {
        **common,
        "normalized_forecasts": signal["normalized_forecasts"],
        "forecast_prices": signal["forecast_prices"],
        "audited_market_fng": signal["audited_market_fng"],
        "macro_anchor": signal["macro_anchor"],
        "prior_year_pivots": signal["prior_year_pivots"],
    }
    position_payload = {
        **common,
        "target_position": signal["target_position"],
        "signal_sigma_after_cost": signal["signal_sigma_after_cost"],
        "agreeing_horizons": signal["agreeing_horizons"],
        "regime": signal["regime"],
        "cap_reason": signal["cap_reason"],
        "research_only": True,
    }

    existing_forecasts = [
        item
        for item in _events("FORECAST_ISSUED_HYBRID_V1")
        if item["event_date"] == market_date
    ]
    if len(existing_forecasts) > 1:
        raise ValueError("multiple forecasts already exist for this market date")
    if existing_forecasts:
        forecast_event = existing_forecasts[0]
        for key in ("normalized_forecasts", "forecast_prices"):
            if forecast_event["payload"].get(key) != forecast_payload[key]:
                raise ValueError("forecast already issued; revision is forbidden")
    else:
        forecast_event = append_event(
            "FORECAST_ISSUED_HYBRID_V1", market_date, forecast_payload
        )

    existing_positions = [
        item
        for item in _events("POSITION_TARGET_ISSUED_HYBRID_V1")
        if item["event_date"] == market_date
    ]
    if len(existing_positions) > 1:
        raise ValueError("multiple positions already exist for this market date")
    if existing_positions:
        position_event = existing_positions[0]
        for key in ("target_position", "signal_sigma_after_cost", "regime", "cap_reason"):
            if position_event["payload"].get(key) != position_payload[key]:
                raise ValueError("position already issued; revision is forbidden")
    else:
        position_event = append_event(
            "POSITION_TARGET_ISSUED_HYBRID_V1", market_date, position_payload
        )
    status = build_status(verified=verified)
    _atomic_json(STATUS_PATH, status)
    return {
        "status": status,
        "forecast_event": forecast_event,
        "position_event": position_event,
    }


def build_status(*, verified: Mapping[str, Any] | None = None) -> dict[str, Any]:
    checked = dict(verified or verify_backtest_artifacts())
    market, _ = load_latest_market()
    signal = checked["signal"]
    freshness = source_freshness(market, datetime.now(TOKYO).date())
    ledger = verify_ledger()
    warnings: list[str] = []
    if not freshness["fresh"]:
        warnings.append("market source is stale")
    if checked["summary"]["promotion"]["promoted"]:
        warnings.append("unexpected promotion flag; human review is still required")
    return {
        "schema_version": 1,
        "status": "ok" if freshness["fresh"] else "stale",
        "mode": "shadow_only",
        "automatic_orders": False,
        "protocol_sha256": checked["protocol_sha256"],
        "source_bundle_sha256": checked["source"]["manifest"]["bundle_sha256"],
        "result_summary_sha256": checked["result_sha256"]["summary.json"],
        "freshness": freshness,
        "latest_market_date": signal["market_date"],
        "latest_forecast": {
            "normalized": signal["normalized_forecasts"],
            "prices": signal["forecast_prices"],
        },
        "latest_position": signal["target_position"],
        "signal_sigma_after_cost": signal["signal_sigma_after_cost"],
        "regime": signal["regime"],
        "promotion": checked["summary"]["promotion"],
        "ledger": ledger,
        "warnings": warnings,
    }


def verify() -> dict[str, Any]:
    checked = verify_backtest_artifacts()
    ledger = verify_ledger()
    protocol_events = _events("PROTOCOL_REGISTERED_HYBRID_V1")
    if len(protocol_events) != 1:
        raise ValueError("protocol must be registered exactly once")
    if protocol_events[0]["payload"]["protocol_sha256"] != checked["protocol_sha256"]:
        raise ValueError("registered protocol hash mismatch")
    latest_source = latest_event("SOURCE_BUNDLE_REGISTERED_HYBRID_V1")
    latest_backtest = latest_event("BACKTEST_REGISTERED_HYBRID_V1")
    latest_forecast = latest_event("FORECAST_ISSUED_HYBRID_V1")
    latest_position = latest_event("POSITION_TARGET_ISSUED_HYBRID_V1")
    if any(item is None for item in (latest_source, latest_backtest, latest_forecast, latest_position)):
        raise ValueError("shadow ledger registrations are incomplete")
    assert latest_source and latest_backtest and latest_forecast and latest_position
    bundle = checked["source"]["manifest"]["bundle_sha256"]
    if latest_source["payload"]["bundle_sha256"] != bundle:
        raise ValueError("latest source event is stale")
    if latest_backtest["payload"]["result_sha256"] != checked["result_sha256"]:
        raise ValueError("latest backtest event is stale")
    signal = checked["signal"]
    if latest_forecast["event_date"] != signal["market_date"]:
        raise ValueError("latest forecast date is stale")
    if latest_forecast["payload"]["forecast_prices"] != signal["forecast_prices"]:
        raise ValueError("latest issued forecast differs from current result")
    if latest_position["payload"]["target_position"] != signal["target_position"]:
        raise ValueError("latest issued position differs from current result")
    status = build_status(verified=checked)
    if STATUS_PATH.exists():
        stored = read_json(STATUS_PATH)
        for key in (
            "protocol_sha256",
            "source_bundle_sha256",
            "result_summary_sha256",
            "latest_market_date",
            "latest_position",
        ):
            if stored.get(key) != status.get(key):
                raise ValueError(f"stored shadow status is stale: {key}")
    return {"ok": True, **status, "ledger": ledger}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "issue", "verify", "status"))
    args = parser.parse_args()
    if args.command == "initialize":
        result = register_artifacts()
        printable = {
            "protocol_event": result["protocol_event"],
            "source_event": result["source_event"],
            "backtest_event": result["backtest_event"],
        }
    elif args.command == "issue":
        result = issue()
        printable = result
    elif args.command == "verify":
        printable = verify()
    else:
        printable = read_json(STATUS_PATH) if STATUS_PATH.exists() else build_status()
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
