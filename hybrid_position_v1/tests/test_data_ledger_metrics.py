from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_position_v1 import data, ledger, metrics


class VerifiedDataBundleTests(unittest.TestCase):
    def _make_bundle(self, root: Path) -> tuple[Path, Path]:
        bundle = root / "bundle"
        bundle.mkdir()
        (bundle / "market.csv").write_text(
            "date,price\n2024-01-02,140.0\n2024-01-03,141.0\n", encoding="utf-8"
        )
        (bundle / "ohlc.csv").write_text(
            "date,open,high,low,close\n2024-01-02,139,141,138,140\n",
            encoding="utf-8",
        )
        (bundle / "macro_history.csv").write_text(
            "release_date,macro_anchor\n2024-01-02,145\n", encoding="utf-8"
        )
        identity = "a" * 64
        manifest = {
            "bundle_sha256": identity,
            "source_remote": data.ALLOWED_REMOTE,
            "market_csv_sha256": data.sha256_file(bundle / "market.csv"),
            "ohlc_csv_sha256": data.sha256_file(bundle / "ohlc.csv"),
            "macro_history_sha256": data.sha256_file(bundle / "macro_history.csv"),
            "protocol_sha256": data.sha256_file(data.MODULE_DIR / "protocol.json"),
        }
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        pointer = root / "latest_source.json"
        pointer.write_text(
            json.dumps(
                {
                    "bundle_dir": str(bundle.relative_to(data.ROOT)),
                    "bundle_sha256": identity,
                    "manifest_sha256": data.sha256_file(bundle / "manifest.json"),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return bundle, pointer

    def test_content_addressed_bundle_verifies_and_loads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hybrid-data-test-", dir=data.ROOT) as tmp:
            bundle, pointer = self._make_bundle(Path(tmp))
            verified = data.verify_latest_source(pointer)
            self.assertEqual(verified["bundle"], bundle)
            market, manifest = data.load_latest_market(pointer)
            self.assertEqual(list(market["price"]), [140.0, 141.0])
            self.assertEqual(manifest["source_remote"], data.ALLOWED_REMOTE)

    def test_modified_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hybrid-data-test-", dir=data.ROOT) as tmp:
            bundle, pointer = self._make_bundle(Path(tmp))
            with (bundle / "market.csv").open("a", encoding="utf-8") as handle:
                handle.write("2024-01-04,999.0\n")
            with self.assertRaisesRegex(ValueError, "market.csv"):
                data.verify_latest_source(pointer)


class AppendOnlyLedgerTests(unittest.TestCase):
    def test_hash_chain_idempotence_and_append_only_triggers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hybrid-ledger-test-") as tmp:
            db = Path(tmp) / "events.sqlite3"
            head = Path(tmp) / "events.head.json"
            first = ledger.append_event(
                "FORECAST",
                "2024-01-02",
                {"forecast": 0.25, "model": "hybrid"},
                db,
                head,
                created_at_utc="2024-01-02T12:00:00+00:00",
            )
            duplicate = ledger.append_event(
                "FORECAST",
                "2024-01-02",
                {"model": "hybrid", "forecast": 0.25},
                db,
                head,
                created_at_utc="2030-01-01T00:00:00+00:00",
            )
            second = ledger.append_event(
                "POSITION",
                "2024-01-02",
                {"position": 0.1},
                db,
                head,
                created_at_utc="2024-01-02T12:01:00+00:00",
            )

            self.assertEqual(first["event_hash"], duplicate["event_hash"])
            self.assertEqual(duplicate["created_at_utc"], first["created_at_utc"])
            self.assertEqual(second["previous_hash"], first["event_hash"])
            self.assertEqual(len(list(ledger.iter_events(db))), 2)
            self.assertEqual(ledger.verify(db, head)["events"], 2)

            connection = sqlite3.connect(db)
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute(
                        "UPDATE events SET event_date='2099-01-01' WHERE sequence=1"
                    )
                connection.rollback()
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute("DELETE FROM events WHERE sequence=1")
            finally:
                connection.close()

            self.assertEqual(ledger.verify(db, head)["head"], second["event_hash"])

    def test_external_head_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hybrid-ledger-test-") as tmp:
            db = Path(tmp) / "events.sqlite3"
            head = Path(tmp) / "events.head.json"
            ledger.append_event(
                "FORECAST",
                "2024-01-02",
                {"forecast": 0.25},
                db,
                head,
                created_at_utc="2024-01-02T12:00:00+00:00",
            )
            head.write_text(
                json.dumps({"sequence": 1, "event_hash": "f" * 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "head mismatch"):
                ledger.verify(db, head)


class MetricTests(unittest.TestCase):
    def test_prediction_metrics_and_finite_pair_filtering(self) -> None:
        result = metrics.prediction_metrics(
            [1.0, -1.0, 2.0, np.nan], [1.0, -1.0, 2.0, 999.0]
        )
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["direction_accuracy"], 1.0)
        self.assertEqual(result["rmse_improvement_pct_vs_rw"], 100.0)

    def test_newey_west_dm_and_holm_adjustment(self) -> None:
        actual = np.linspace(-2.0, 3.0, 80)
        predicted = 0.8 * actual
        pvalue = metrics.newey_west_dm_pvalue(actual, predicted, lag=4)
        self.assertTrue(math.isfinite(pvalue))
        self.assertGreaterEqual(pvalue, 0.0)
        self.assertLessEqual(pvalue, 1.0)

        adjusted = metrics.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "missing": math.nan})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertTrue(math.isnan(adjusted["missing"]))

    def test_maximum_drawdown_includes_initial_equity(self) -> None:
        # The initial capital is 1.0; an immediate -10% return is a 10% drawdown.
        self.assertAlmostEqual(metrics.maximum_drawdown(pd.Series([-0.10, 0.05])), 0.10)
        self.assertAlmostEqual(
            metrics.maximum_drawdown(pd.Series([0.10, -0.20, 0.05])), 0.20
        )

    def test_strategy_metrics_accepts_absent_optional_position_columns(self) -> None:
        result = metrics.strategy_metrics(pd.DataFrame({"net_return": [0.01, -0.005, 0.002]}))
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["turnover"], 0.0)
        self.assertEqual(result["mean_abs_position"], 0.0)
        self.assertEqual(result["max_abs_position"], 0.0)


if __name__ == "__main__":
    unittest.main()
