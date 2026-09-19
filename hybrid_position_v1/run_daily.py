#!/usr/bin/env python3
"""Daily guarded refresh for the hybrid USD/JPY shadow model.

Order is fixed: integrity preflight, allowlisted source fast-forward, immutable
source archive, model refresh, first-seen shadow issue, integrity postflight.
No command in this file can place an order.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_DEPENDENCIES = ROOT / "work" / "python_deps"
if LOCAL_DEPENDENCIES.is_dir():
    # Append so the bundled pandas/numpy remain authoritative while optional
    # pyarrow/yfinance can be resolved from the local dependency directory.
    sys.path.append(str(LOCAL_DEPENDENCIES))

from hybrid_position_v1 import run_backtest, run_shadow, sync_sources  # noqa: E402


NETWORK_ENV = "USDJPY_HYBRID_ALLOW_NETWORK"
DEFAULT_FNG_REPO = ROOT / "work" / "forex-fear-and-greed-source"
LOCK_PATH = MODULE_DIR / "state" / "daily_run.lock"


def update_allowlisted_source(repo: Path) -> dict[str, Any]:
    remote_before, commit_before = sync_sources.validate_source_repo(repo)
    result = subprocess.run(
        ["git", "-C", str(repo), "pull", "--ff-only", "origin", "main"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_after, commit_after = sync_sources.validate_source_repo(repo)
    if remote_after != sync_sources.ALLOWED_REMOTE or remote_before != remote_after:
        raise ValueError("source remote changed during refresh")
    return {
        "remote": remote_after,
        "commit_before": commit_before,
        "commit_after": commit_after,
        "changed": commit_before != commit_after,
        "git_result": (result.stdout or result.stderr).strip(),
    }


def run(repo: Path, *, skip_pull: bool = False) -> dict[str, Any]:
    if os.environ.get(NETWORK_ENV) != "1":
        raise RuntimeError(f"network is disabled; set {NETWORK_ENV}=1")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another hybrid daily run is active") from exc

        preflight = run_shadow.verify()
        source_update = (
            {"skipped": True, "reason": "--skip-pull"}
            if skip_pull
            else update_allowlisted_source(repo)
        )
        archive_args = argparse.Namespace(
            fng_repo=repo,
            macro_history=ROOT / "accuracy_v2" / "results" / "features_and_forecasts.csv",
            ohlc_csv=None,
            ohlc_start="2018-01-01",
        )
        archive = sync_sources.archive(archive_args)
        if archive["manifest"]["bundle_sha256"] == preflight["source_bundle_sha256"]:
            postflight = run_shadow.verify()
            return {
                "schema_version": 1,
                "status": postflight["status"],
                "updated": False,
                "reason": "verified source bundle is unchanged",
                "preflight": preflight,
                "source_update": source_update,
                "source_bundle_sha256": archive["manifest"]["bundle_sha256"],
                "market_date": postflight["latest_market_date"],
                "forecast_prices": postflight["latest_forecast"]["prices"],
                "target_position": postflight["latest_position"],
                "promotion": postflight["promotion"],
                "postflight": postflight,
            }
        backtest = run_backtest.run(skip_ablations=True)
        issued = run_shadow.issue()
        postflight = run_shadow.verify()
        return {
            "schema_version": 1,
            "status": postflight["status"],
            "updated": True,
            "preflight": preflight,
            "source_update": source_update,
            "source_bundle_sha256": archive["manifest"]["bundle_sha256"],
            "market_date": backtest["current_signal"]["market_date"],
            "forecast_prices": backtest["current_signal"]["forecast_prices"],
            "target_position": backtest["current_signal"]["target_position"],
            "promotion": backtest["promotion"],
            "issued_forecast_sequence": issued["forecast_event"]["sequence"],
            "issued_position_sequence": issued["position_event"]["sequence"],
            "postflight": postflight,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--fng-repo", type=Path, default=DEFAULT_FNG_REPO)
    parser.add_argument("--skip-pull", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        result = run_shadow.verify()
    else:
        result = run(args.fng_repo.resolve(), skip_pull=args.skip_pull)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status", "ok") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
