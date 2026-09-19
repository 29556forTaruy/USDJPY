"""Append-only hash-chained event ledger for hybrid shadow forecasts."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = MODULE_DIR / "state" / "hybrid_v1.sqlite3"
DEFAULT_HEAD = MODULE_DIR / "state" / "hybrid_v1.sqlite3.head.json"
GENESIS = "0" * 64


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def event_digest(
    sequence: int,
    event_type: str,
    event_date: str,
    created_at_utc: str,
    previous_hash: str,
    payload_json: str,
) -> str:
    material = "\n".join(
        [str(sequence), event_type, event_date, created_at_utc, previous_hash, payload_json]
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def initialize(path: Path = DEFAULT_DB) -> None:
    with connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_date TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS events_idempotency
              ON events(event_type, event_date, payload_sha256);
            CREATE TRIGGER IF NOT EXISTS events_no_update
              BEFORE UPDATE ON events
              BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete
              BEFORE DELETE ON events
              BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            """
        )


def _atomic_head(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_event(
    event_type: str,
    event_date: str,
    payload: dict[str, Any],
    path: Path = DEFAULT_DB,
    head_path: Path = DEFAULT_HEAD,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Append one idempotent event and atomically refresh the external head."""

    initialize(path)
    payload_json = canonical_json(payload)
    payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    created = created_at_utc or datetime.now(timezone.utc).isoformat()
    connection = connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        duplicate = connection.execute(
            "SELECT * FROM events WHERE event_type=? AND event_date=? AND payload_sha256=?",
            (event_type, event_date, payload_sha),
        ).fetchone()
        if duplicate is not None:
            connection.rollback()
            return dict(duplicate)
        last = connection.execute(
            "SELECT sequence,event_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if last is None else int(last["sequence"]) + 1
        previous = GENESIS if last is None else str(last["event_hash"])
        digest = event_digest(sequence, event_type, event_date, created, previous, payload_json)
        connection.execute(
            """INSERT INTO events
               (sequence,event_type,event_date,created_at_utc,previous_hash,payload_json,payload_sha256,event_hash)
               VALUES (?,?,?,?,?,?,?,?)""",
            (sequence, event_type, event_date, created, previous, payload_json, payload_sha, digest),
        )
        connection.commit()
    finally:
        connection.close()
    _atomic_head(
        head_path,
        {
            "schema_version": 1,
            "sequence": sequence,
            "event_hash": digest,
            "updated_at_utc": created,
        },
    )
    return {
        "sequence": sequence,
        "event_type": event_type,
        "event_date": event_date,
        "created_at_utc": created,
        "previous_hash": previous,
        "payload_json": payload_json,
        "payload_sha256": payload_sha,
        "event_hash": digest,
    }


def iter_events(path: Path = DEFAULT_DB) -> Iterable[sqlite3.Row]:
    connection = connect(path)
    try:
        rows = connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
    finally:
        connection.close()
    return rows


def verify(path: Path = DEFAULT_DB, head_path: Path = DEFAULT_HEAD) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ledger is not initialized: {path}")
    previous = GENESIS
    count = 0
    last_hash = GENESIS
    for expected, row in enumerate(iter_events(path), start=1):
        if int(row["sequence"]) != expected:
            raise ValueError("ledger sequence gap")
        if row["previous_hash"] != previous:
            raise ValueError("ledger previous-hash mismatch")
        if hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest() != row["payload_sha256"]:
            raise ValueError("ledger payload hash mismatch")
        calculated = event_digest(
            expected,
            row["event_type"],
            row["event_date"],
            row["created_at_utc"],
            row["previous_hash"],
            row["payload_json"],
        )
        if calculated != row["event_hash"]:
            raise ValueError("ledger event hash mismatch")
        previous = calculated
        last_hash = calculated
        count = expected
    if not head_path.exists():
        raise FileNotFoundError("external ledger head is missing")
    with head_path.open(encoding="utf-8") as handle:
        head = json.load(handle)
    if int(head.get("sequence", -1)) != count or head.get("event_hash") != last_hash:
        raise ValueError("external ledger head mismatch")
    return {"ok": True, "events": count, "head": last_hash}


def latest_event(event_type: str, path: Path = DEFAULT_DB) -> dict[str, Any] | None:
    if not path.exists():
        return None
    connection = connect(path)
    try:
        row = connection.execute(
            "SELECT * FROM events WHERE event_type=? ORDER BY sequence DESC LIMIT 1",
            (event_type,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json"))
    return result
