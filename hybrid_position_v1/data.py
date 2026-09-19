"""Verified loading of content-addressed hybrid model source bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
LATEST_POINTER = MODULE_DIR / "state" / "latest_source.json"
ALLOWED_REMOTE = "https://github.com/29556forTaruy/forex-fear-and-greed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_latest_source(pointer_path: Path = LATEST_POINTER) -> dict[str, Any]:
    """Verify pointer, manifest, normalized market data, OHLC, and protocol hashes."""

    pointer = read_json(pointer_path)
    relative = Path(str(pointer["bundle_dir"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("bundle path must remain below the project root")
    bundle = ROOT / relative
    manifest_path = bundle / "manifest.json"
    if sha256_file(manifest_path) != pointer["manifest_sha256"]:
        raise ValueError("source manifest hash mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("bundle_sha256") != pointer.get("bundle_sha256"):
        raise ValueError("bundle identity mismatch")
    if manifest.get("source_remote") != ALLOWED_REMOTE:
        raise ValueError("source remote is outside the allowlist")
    checks = {
        "market.csv": "market_csv_sha256",
        "ohlc.csv": "ohlc_csv_sha256",
        "macro_history.csv": "macro_history_sha256",
    }
    for name, key in checks.items():
        path = bundle / name
        if not path.exists() or sha256_file(path) != manifest.get(key):
            raise ValueError(f"source artifact hash mismatch: {name}")
    protocol_path = MODULE_DIR / "protocol.json"
    if sha256_file(protocol_path) != manifest.get("protocol_sha256"):
        raise ValueError("protocol changed after source registration")
    return {"pointer": pointer, "manifest": manifest, "bundle": bundle}


def load_latest_market(pointer_path: Path = LATEST_POINTER) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the verified normalized daily market frame and provenance manifest."""

    verified = verify_latest_source(pointer_path)
    market_path = verified["bundle"] / "market.csv"
    frame = pd.read_csv(market_path, parse_dates=["date"]).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("market dates must be unique and sorted")
    if frame["price"].dropna().le(0).any():
        raise ValueError("USDJPY prices must be positive")
    return frame, verified["manifest"]


def source_freshness(frame: pd.DataFrame, as_of: str | pd.Timestamp) -> dict[str, Any]:
    """Report calendar-day freshness without silently accepting a stale source."""

    latest = pd.Timestamp(frame["price"].dropna().index.max()).normalize()
    cutoff = pd.Timestamp(as_of).normalize()
    age = int((cutoff - latest).days)
    return {
        "as_of": str(cutoff.date()),
        "latest_market_date": str(latest.date()),
        "calendar_age_days": age,
        "fresh": 0 <= age <= 4,
    }
