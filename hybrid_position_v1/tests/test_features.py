from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from hybrid_position_v1.features import (
    audited_market_fng,
    build_features,
    forward_targets,
    macro_features,
    prior_year_pivots,
)


def synthetic_daily_frame(
    *, start: str = "2018-01-02", periods: int = 900
) -> pd.DataFrame:
    """Make deterministic, strictly positive daily market inputs."""

    index = pd.bdate_range(start, periods=periods)
    step = np.arange(periods, dtype=float)
    price = 105.0 * np.exp(0.00015 * step + 0.012 * np.sin(step / 19.0))
    frame = pd.DataFrame(index=index)
    frame["price"] = price
    frame["high"] = price * (1.004 + 0.0005 * (1.0 + np.sin(step / 7.0)))
    frame["low"] = price * (0.996 - 0.0005 * (1.0 + np.cos(step / 11.0)))
    frame["vix"] = 18.0 + 2.5 * np.sin(step / 23.0) + 0.002 * step
    frame["sp500"] = 2400.0 * np.exp(0.0003 * step + 0.01 * np.sin(step / 31.0))
    frame["nikkei"] = 21000.0 * np.exp(0.0002 * step + 0.012 * np.cos(step / 29.0))
    frame["audjpy"] = 82.0 * np.exp(0.0001 * step + 0.015 * np.sin(step / 17.0))
    return frame


class FeatureLeakageTests(unittest.TestCase):
    def test_predictor_prefix_is_invariant_to_all_future_market_changes(self) -> None:
        frame = synthetic_daily_frame()
        cutoff = 640
        baseline = build_features(frame, include_targets=False)

        changed = frame.copy()
        future = changed.index[cutoff:]
        changed.loc[future, "price"] *= 1.35
        changed.loc[future, "high"] *= 1.55
        changed.loc[future, "low"] *= 1.20
        changed.loc[future, "vix"] *= 2.0
        changed.loc[future, "sp500"] *= 0.70
        changed.loc[future, "nikkei"] *= 1.25
        changed.loc[future, "audjpy"] *= 0.80
        perturbed = build_features(changed, include_targets=False)

        assert_frame_equal(
            baseline.iloc[:cutoff], perturbed.iloc[:cutoff], check_exact=True
        )

    def test_targets_are_opt_in_and_do_not_change_predictors(self) -> None:
        frame = synthetic_daily_frame(periods=520)
        predictors = build_features(frame, include_targets=False)
        labelled = build_features(frame, include_targets=True)
        target_columns = [column for column in labelled if column.startswith("target_z_")]

        self.assertEqual(target_columns, ["target_z_1", "target_z_5", "target_z_20"])
        self.assertFalse(any(column.startswith("target_z_") for column in predictors))
        assert_frame_equal(
            predictors,
            labelled.drop(columns=target_columns),
            check_exact=True,
        )

    def test_forward_target_uses_only_t_scale_and_t_plus_h_price(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=6)
        price = pd.Series([100.0, 101.0, 103.0, 102.0, 106.0, 108.0], index=index)
        scale = pd.Series([0.02, 0.03, 0.04, 0.05, 0.06, 0.07], index=index)
        result = forward_targets(price, scale, horizons=(1, 2))

        self.assertAlmostEqual(
            result.loc[index[0], "target_z_1"], np.log(101.0 / 100.0) / 0.02
        )
        self.assertAlmostEqual(
            result.loc[index[1], "target_z_2"],
            np.log(102.0 / 101.0) / (0.03 * np.sqrt(2.0)),
        )
        self.assertTrue(np.isnan(result.loc[index[-1], "target_z_1"]))
        self.assertTrue(result["target_z_2"].tail(2).isna().all())


class PriorYearPivotTests(unittest.TestCase):
    def test_pivots_are_fixed_within_year_and_use_prior_year_ohlc(self) -> None:
        frame = synthetic_daily_frame(start="2019-01-02", periods=700)
        pivots = prior_year_pivots(frame)
        prior = frame.loc["2019"]
        high = float(prior["high"].max())
        low = float(prior["low"].min())
        close = float(prior["price"].iloc[-1])
        center = (high + low + close) / 3.0

        rows_2020 = pivots.loc["2020"]
        self.assertTrue(rows_2020["pivot_p"].eq(center).all())
        self.assertTrue(rows_2020["pivot_r1"].eq(2.0 * center - low).all())
        self.assertTrue(rows_2020["pivot_s1"].eq(2.0 * center - high).all())
        self.assertTrue(rows_2020["pivot_r2"].eq(center + high - low).all())
        self.assertTrue(rows_2020["pivot_s2"].eq(center - high + low).all())
        self.assertTrue(rows_2020["pivot_source_year"].eq(2019.0).all())

    def test_current_year_ohlc_cannot_change_that_years_pivots(self) -> None:
        frame = synthetic_daily_frame(start="2019-01-02", periods=700)
        baseline = prior_year_pivots(frame).loc["2020"]
        changed = frame.copy()
        current = changed.index.year == 2020
        changed.loc[current, "price"] *= 1.4
        changed.loc[current, "high"] *= 2.0
        changed.loc[current, "low"] *= 0.5
        actual = prior_year_pivots(changed).loc["2020"]

        assert_frame_equal(baseline, actual, check_exact=True)


class AuditedFearGreedTests(unittest.TestCase):
    def test_excluded_rate_cot_breadth_and_published_headline_are_ignored(self) -> None:
        frame = synthetic_daily_frame(periods=520)
        frame["rate_diff"] = np.linspace(-5.0, 5.0, len(frame))
        frame["cot"] = np.arange(len(frame), dtype=float)
        frame["breadth"] = 1.0
        frame["source_fear_greed"] = 99.0
        baseline = audited_market_fng(frame)

        changed = frame.copy()
        changed["rate_diff"] = 1_000_000.0
        changed["cot"] = -1_000_000.0
        changed["breadth"] = 0.0
        changed["source_fear_greed"] = 1.0
        actual = audited_market_fng(changed)

        assert_frame_equal(baseline, actual, check_exact=True)
        valid = actual["audited_market_fng"].dropna()
        self.assertGreater(len(valid), 0)
        self.assertTrue(valid.between(0.0, 100.0).all())
        self.assertTrue(actual["fear_coverage"].dropna().between(0.0, 1.0).all())

    def test_weights_cannot_reintroduce_excluded_components(self) -> None:
        frame = synthetic_daily_frame(periods=100)
        with self.assertRaisesRegex(ValueError, "exactly"):
            audited_market_fng(
                frame,
                weights={
                    "momentum": 0.18,
                    "strength": 0.12,
                    "realized_vol": 0.18,
                    "risk_regime": 0.14,
                    "cot": 0.10,
                },
            )


class MacroAvailabilityTests(unittest.TestCase):
    def test_anchor_is_not_backfilled_and_expires_after_target_date(self) -> None:
        index = pd.bdate_range("2024-01-08", "2024-01-19")
        frame = pd.DataFrame({"price": 100.0}, index=index)
        release = pd.Timestamp("2024-01-10")
        target = pd.Timestamp("2024-01-17")
        frame["macro_anchor"] = np.nan
        frame["macro_target_date"] = pd.NaT
        frame.loc[release, "macro_anchor"] = 110.0
        frame.loc[release, "macro_target_date"] = target
        scale = pd.Series(0.01, index=index)

        result = macro_features(frame, scale)

        self.assertTrue(result.loc[: release - pd.offsets.BDay(1), "macro_available"].eq(0.0).all())
        self.assertTrue(result.loc[release:target, "macro_available"].eq(1.0).all())
        self.assertTrue(result.loc[target + pd.offsets.BDay(1) :, "macro_available"].eq(0.0).all())
        self.assertTrue(result.loc[: release - pd.offsets.BDay(1), "macro_gap_1"].isna().all())
        self.assertTrue(result.loc[target + pd.offsets.BDay(1) :, "macro_gap_1"].isna().all())

        remaining = float(np.busday_count(release.date(), target.date()))
        expected = np.log(110.0 / 100.0) * (1.0 / remaining) / 0.01
        self.assertAlmostEqual(result.loc[release, "macro_gap_1"], expected)
        self.assertEqual(result.loc[release, "macro_remaining_business_days"], remaining)

    def test_macro_gap_obeys_registered_horizon_formula(self) -> None:
        index = pd.bdate_range("2024-02-01", periods=4)
        frame = pd.DataFrame(
            {
                "price": 100.0,
                "macro_anchor": 101.0,
                "macro_target_date": pd.Timestamp("2024-02-15"),
            },
            index=index,
        )
        scale = pd.Series(0.02, index=index)
        result = macro_features(frame, scale)
        remaining = float(np.busday_count(index[0].date(), pd.Timestamp("2024-02-15").date()))
        expected = (
            np.log(101.0 / 100.0)
            * min(1.0, 5.0 / max(remaining, 5.0))
            / (0.02 * np.sqrt(5.0))
        )
        self.assertAlmostEqual(result.iloc[0]["macro_gap_5"], expected)


if __name__ == "__main__":
    unittest.main()
