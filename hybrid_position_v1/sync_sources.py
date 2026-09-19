#!/usr/bin/env python3
"""Archive immutable inputs for the hybrid USD/JPY shadow model.

The script reads the public ``29556forTaruy/forex-fear-and-greed`` seed,
adds exact USD/JPY OHLC needed by the prior-year pivot, and joins only macro
forecasts that were available on each date.  It never mutates the existing
``accuracy_v2`` protocol or ledger.

Network access is opt-in through ``USDJPY_HYBRID_ALLOW_NETWORK=1``.  Without
it, callers must provide a previously saved OHLC CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
STATE_DIR = MODULE_DIR / "state"
RAW_DIR = STATE_DIR / "raw"
ALLOW_NETWORK_ENV = "USDJPY_HYBRID_ALLOW_NETWORK"
ALLOWED_REMOTE = "https://github.com/29556forTaruy/forex-fear-and-greed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_source_repo(repo: Path) -> tuple[str, str]:
    if not (repo / ".git").exists():
        raise ValueError(f"not a git repository: {repo}")
    remote = _git(repo, "remote", "get-url", "origin").removesuffix(".git")
    if remote != ALLOWED_REMOTE:
        raise ValueError(f"source remote is not allowlisted: {remote}")
    commit = _git(repo, "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("unexpected source commit SHA")
    return remote, commit


def quarter_end(value: str) -> pd.Timestamp:
    period = pd.Period(value, freq="Q")
    return period.end_time.normalize()


def load_macro_history(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {
        "release_date",
        "target_quarter",
        "rw_floor_ensemble",
        "origin_spot",
        "origin_quarter",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"macro history is missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=index)
    out["macro_anchor"] = pd.NA
    out["macro_origin_spot"] = pd.NA
    out["macro_release_date"] = pd.NaT
    out["macro_target_date"] = pd.NaT
    out["macro_origin_quarter"] = pd.NA
    out["macro_target_quarter"] = pd.NA

    usable = source.dropna(subset=["release_date", "target_quarter", "rw_floor_ensemble"])
    usable = usable.sort_values("release_date")
    for row in usable.itertuples(index=False):
        release = pd.Timestamp(row.release_date).normalize()
        target = quarter_end(str(row.target_quarter))
        mask = (out.index >= release) & (out.index <= target)
        # Later releases overwrite earlier anchors only from their own release date.
        out.loc[mask, "macro_anchor"] = float(row.rw_floor_ensemble)
        out.loc[mask, "macro_origin_spot"] = float(row.origin_spot)
        out.loc[mask, "macro_release_date"] = release
        out.loc[mask, "macro_target_date"] = target
        out.loc[mask, "macro_origin_quarter"] = str(row.origin_quarter)
        out.loc[mask, "macro_target_quarter"] = str(row.target_quarter)

    out["macro_anchor"] = pd.to_numeric(out["macro_anchor"], errors="coerce")
    out["macro_origin_spot"] = pd.to_numeric(out["macro_origin_spot"], errors="coerce")
    return out


def compute_source_diagnostics(repo: Path, seed: pd.DataFrame) -> pd.DataFrame:
    """Run the source repository's own index code for diagnostics only."""

    source_path = str(repo)
    sys.path.insert(0, source_path)
    previous_config = sys.modules.pop("config", None)
    try:
        from fng.index import compute_index  # type: ignore

        computed = compute_index(seed.copy())
    finally:
        sys.path.remove(source_path)
        for name in list(sys.modules):
            if name == "fng" or name.startswith("fng."):
                sys.modules.pop(name, None)
        sys.modules.pop("config", None)
        if previous_config is not None:
            sys.modules["config"] = previous_config
    return computed.add_prefix("source_")


def fetch_ohlc(start: str) -> pd.DataFrame:
    if os.environ.get(ALLOW_NETWORK_ENV) != "1":
        raise PermissionError(
            f"network disabled; set {ALLOW_NETWORK_ENV}=1 or provide --ohlc-csv"
        )
    import yfinance as yf

    history = yf.Ticker("USDJPY=X").history(start=start, auto_adjust=True)
    required = ["Open", "High", "Low", "Close"]
    if history.empty or any(column not in history for column in required):
        raise RuntimeError("failed to fetch USDJPY OHLC")
    frame = history[required].copy()
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    frame.columns = ["open", "high", "low", "close"]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def load_ohlc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OHLC CSV is missing columns: {sorted(missing)}")
    return frame[list(sorted(required))]


def normalize_market(seed: pd.DataFrame, diagnostics: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    seed = seed.copy()
    seed.index = pd.DatetimeIndex(seed.index).tz_localize(None).normalize()
    seed = seed[~seed.index.duplicated(keep="last")].sort_index()
    if "price" not in seed or seed["price"].dropna().empty:
        raise ValueError("fear seed has no valid price column")

    full_index = ohlc.index.union(seed.index).sort_values()
    market = ohlc.reindex(full_index)
    for column in seed.columns:
        market[column] = seed[column].reindex(full_index)
    market["price"] = market["price"].combine_first(market["close"])
    for column in diagnostics.columns:
        market[column] = diagnostics[column].reindex(full_index)
    market.index.name = "date"
    return market


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def archive(args: argparse.Namespace) -> dict[str, Any]:
    source_repo = args.fng_repo.resolve()
    remote, commit = validate_source_repo(source_repo)
    seed_path = source_repo / "data" / "seed" / "dataset_USDJPY.parquet"
    source_manifest_path = source_repo / "data" / "seed" / "manifest.json"
    if not seed_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError("USDJPY seed or source manifest is missing")

    seed = pd.read_parquet(seed_path)
    diagnostics = compute_source_diagnostics(source_repo, seed)
    if args.ohlc_csv:
        ohlc = load_ohlc(args.ohlc_csv.resolve())
    else:
        ohlc = fetch_ohlc(args.ohlc_start)
    market = normalize_market(seed, diagnostics, ohlc)

    macro_path = args.macro_history.resolve()
    macro = load_macro_history(macro_path, market.index)
    market = market.join(macro)

    # Hash normalized CSV bytes before choosing the content-addressed directory.
    csv_bytes = market.to_csv(date_format="%Y-%m-%d", float_format="%.12g").encode("utf-8")
    market_sha = hashlib.sha256(csv_bytes).hexdigest()
    seed_sha = sha256_file(seed_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    macro_sha = sha256_file(macro_path)
    protocol_sha = sha256_file(MODULE_DIR / "protocol.json")
    bundle_key_material = json.dumps(
        {
            "source_commit": commit,
            "seed_sha256": seed_sha,
            "source_manifest_sha256": source_manifest_sha,
            "ohlc_rows": len(ohlc),
            "market_sha256": market_sha,
            "macro_sha256": macro_sha,
            "protocol_sha256": protocol_sha,
        },
        sort_keys=True,
    ).encode("utf-8")
    bundle_sha = hashlib.sha256(bundle_key_material).hexdigest()
    destination = RAW_DIR / bundle_sha
    existing_manifest_path = destination / "manifest.json"
    if existing_manifest_path.exists():
        with existing_manifest_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = {
            "bundle_sha256": bundle_sha,
            "source_commit": commit,
            "seed_sha256": seed_sha,
            "source_manifest_sha256": source_manifest_sha,
            "macro_history_sha256": macro_sha,
            "protocol_sha256": protocol_sha,
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(f"immutable source bundle mismatch: {key}")
        artifacts = {
            "market.csv": "market_csv_sha256",
            "ohlc.csv": "ohlc_csv_sha256",
            "macro_history.csv": "macro_history_sha256",
        }
        for name, key in artifacts.items():
            path = destination / name
            if not path.is_file() or sha256_file(path) != existing.get(key):
                raise ValueError(f"immutable source bundle artifact mismatch: {name}")
        if existing.get("market_csv_sha256") != hashlib.sha256(csv_bytes).hexdigest():
            raise ValueError("existing bundle id collided with different market bytes")
        pointer = {
            "schema_version": 1,
            "bundle_sha256": bundle_sha,
            "bundle_dir": str(destination.relative_to(ROOT)),
            "manifest_sha256": sha256_file(existing_manifest_path),
        }
        atomic_json(STATE_DIR / "latest_source.json", pointer)
        return {"pointer": pointer, "manifest": existing}

    destination.mkdir(parents=True, exist_ok=False)

    market_path = destination / "market.csv"
    market_path.write_bytes(csv_bytes)
    ohlc_path = destination / "ohlc.csv"
    ohlc.to_csv(ohlc_path, date_format="%Y-%m-%d", float_format="%.12g")
    shutil.copy2(source_manifest_path, destination / "source_manifest.json")
    shutil.copy2(macro_path, destination / "macro_history.csv")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "bundle_sha256": bundle_sha,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_remote": remote,
        "source_commit": commit,
        "seed_path": str(seed_path),
        "seed_sha256": seed_sha,
        "source_manifest_sha256": source_manifest_sha,
        "macro_history_sha256": macro_sha,
        "protocol_sha256": protocol_sha,
        "market_csv_sha256": sha256_file(market_path),
        "ohlc_csv_sha256": sha256_file(ohlc_path),
        "first_date": str(market.index.min().date()),
        "last_date": str(market.index.max().date()),
        "rows": int(len(market)),
        "historical_quality": "current-file history; not point-in-time vintages",
        "strict_fng_note": "source headline is diagnostic only; model recomputes audited components",
    }
    atomic_json(destination / "manifest.json", manifest)
    pointer = {
        "schema_version": 1,
        "bundle_sha256": bundle_sha,
        "bundle_dir": str(destination.relative_to(ROOT)),
        "manifest_sha256": sha256_file(destination / "manifest.json"),
    }
    atomic_json(STATE_DIR / "latest_source.json", pointer)
    return {"pointer": pointer, "manifest": manifest}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fng-repo", type=Path, required=True)
    ap.add_argument(
        "--macro-history",
        type=Path,
        default=ROOT / "accuracy_v2" / "results" / "features_and_forecasts.csv",
    )
    ap.add_argument("--ohlc-csv", type=Path)
    ap.add_argument("--ohlc-start", default="2018-01-01")
    return ap


def main() -> int:
    result = archive(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
