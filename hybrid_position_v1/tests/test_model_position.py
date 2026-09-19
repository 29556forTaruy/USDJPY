from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hybrid_position_v1.model import (
    ensemble_weights,
    fit_ridge,
    predict_ridge,
    purged_expanding_splits,
    walk_forward_forecasts,
)
from hybrid_position_v1.position import PositionConfig, backtest_positions


def _forecast_frame(
    *, start: str = "2024-01-02", periods: int = 130
) -> pd.DataFrame:
    """Small deterministic input with a named DatetimeIndex."""

    index = pd.bdate_range(start, periods=periods, name="date")
    step = np.arange(periods, dtype=float)
    close = 100.0 * np.exp(0.0005 * step + 0.01 * np.sin(step / 7.0))
    return pd.DataFrame(
        {
            "close": close,
            "daily_vol": 0.01 + 0.001 * np.cos(step / 11.0),
            "macro_x": np.sin(step / 17.0),
            "technical_x": np.cos(step / 13.0),
        },
        index=index,
    )


def _position_frame(
    forward_returns: list[float],
    *,
    forecast: float = 2.0,
    vol_ratio: float = 1.0,
) -> pd.DataFrame:
    count = len(forward_returns)
    return pd.DataFrame(
        {
            "price": 100.0,
            "daily_vol": 0.01,
            "annualized_vol": 0.05,
            "pred_1": forecast,
            "pred_5": forecast,
            "pred_20": forecast,
            "fear_level": 50.0,
            "trend_score": 1.0,
            "vol_ratio": vol_ratio,
            "forward": forward_returns,
        },
        index=pd.bdate_range("2024-01-02", periods=count, name="date"),
    )


class RidgeLeakageControlTests(unittest.TestCase):
    def test_named_datetime_index_is_accepted_without_ambiguous_date_error(self) -> None:
        frame = _forecast_frame(periods=70)
        result = walk_forward_forecasts(
            frame,
            macro_features=["macro_x"],
            technical_features=["technical_x"],
            horizons=(1,),
            minimum_training_rows=20,
            maximum_training_rows=35,
        )

        self.assertEqual(len(result), len(frame))
        self.assertEqual(result["date"].iloc[0], frame.index[0])
        self.assertEqual(result["date"].iloc[-1], frame.index[-1])
        self.assertEqual(set(result["horizon"]), {1})

    def test_purged_splits_obey_target_resolution_and_embargo_boundaries(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=80)
        # Each label resolves seven business rows after its origin.
        target_end = pd.Series(dates, index=np.arange(len(dates))).shift(-7)
        target_end.iloc[-7:] = dates[-1] + pd.offsets.BDay(20)
        splits = purged_expanding_splits(
            dates,
            target_end,
            n_splits=3,
            embargo=5,
            minimum_inner_training_rows=20,
        )

        self.assertEqual(len(splits), 3)
        for training, validation in splits:
            validation_start = int(validation[0])
            self.assertTrue(np.all(training < validation_start - 5))
            self.assertTrue(
                np.all(
                    pd.DatetimeIndex(target_end.iloc[training])
                    <= dates[validation_start]
                )
            )

    def test_scaling_and_imputation_are_estimated_from_training_rows_only(self) -> None:
        training_x = pd.DataFrame(
            {"a": [0.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]}
        )
        training_y = np.array([-1.0, 0.0, 2.0])
        fitted = fit_ridge(training_x, training_y, alpha=1.0)

        np.testing.assert_allclose(fitted.medians, [1.0, 20.0])
        np.testing.assert_allclose(fitted.means, [1.0, 20.0])
        original_medians = fitted.medians.copy()
        original_means = fitted.means.copy()
        original_scales = fitted.scales.copy()

        # An extreme held-out row is transformed with the frozen training
        # statistics; prediction must not update them.
        prediction = predict_ridge(
            fitted,
            pd.DataFrame({"a": [1.0e12], "b": [np.nan]}),
            clip_sigma=None,
        )
        self.assertTrue(np.isfinite(prediction[0]))
        np.testing.assert_array_equal(fitted.medians, original_medians)
        np.testing.assert_array_equal(fitted.means, original_means)
        np.testing.assert_array_equal(fitted.scales, original_scales)

    def test_models_refit_only_at_month_boundary_and_rolling_window_is_capped(self) -> None:
        frame = _forecast_frame(periods=150)
        result = walk_forward_forecasts(
            frame,
            macro_features=["macro_x"],
            technical_features=["technical_x"],
            horizons=(1,),
            minimum_training_rows=20,
            maximum_training_rows=30,
        )
        month = result["date"].dt.to_period("M")
        expected_refit = month.ne(month.shift(1))

        np.testing.assert_array_equal(result["model_refitted"], expected_refit)
        learned = result.loc[result["training_rows_hybrid_ridge"] > 0]
        self.assertGreater(len(learned), 0)
        self.assertLessEqual(int(learned["training_rows_hybrid_ridge"].max()), 30)
        self.assertEqual(int(learned["training_rows_hybrid_ridge"].max()), 30)
        for _, group in result.groupby(month, sort=False):
            self.assertLessEqual(group["training_rows_hybrid_ridge"].nunique(), 1)

    def test_dynamic_ensemble_cannot_push_random_walk_below_floor(self) -> None:
        actual = np.linspace(-2.0, 2.0, 252)
        scored = pd.DataFrame(
            {
                "actual": actual,
                "random_walk": 0.0,
                "macro_ridge": actual * 0.5,
                "technical_ridge": actual * 0.8,
                "hybrid_ridge": actual,
            }
        )
        weights = ensemble_weights(scored)

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreaterEqual(weights["random_walk"], 0.35)
        self.assertTrue(all(value >= 0.0 for value in weights.values()))


class PositionTimingAndRiskTests(unittest.TestCase):
    def test_t_close_position_earns_t_to_t_plus_one_return_without_shift(self) -> None:
        frame = _position_frame([np.nan, np.nan, np.nan], forecast=1.0)
        frame["price"] = [100.0, 110.0, 110.0]
        result = backtest_positions(
            frame.drop(columns="forward"),
            config=PositionConfig(maximum_daily_position_change=1.0),
        )

        expected_return = 110.0 / 100.0 - 1.0
        self.assertGreater(result["position"].iloc[0], 0.0)
        self.assertAlmostEqual(
            result["gross_pnl"].iloc[0],
            result["position"].iloc[0] * expected_return,
        )
        self.assertAlmostEqual(result["gross_pnl"].iloc[1], 0.0)
        self.assertTrue(np.isnan(result["gross_pnl"].iloc[2]))

    def test_entry_threshold_regime_cap_and_transaction_cost(self) -> None:
        below = _position_frame([0.0], forecast=0.19)
        below_result = backtest_positions(
            below,
            forward_return_col="forward",
            config=PositionConfig(maximum_daily_position_change=1.0),
        )
        self.assertEqual(below_result["position"].iloc[0], 0.0)

        stress = _position_frame([0.0], forecast=2.0, vol_ratio=2.0)
        stress_result = backtest_positions(
            stress,
            forward_return_col="forward",
            config=PositionConfig(maximum_daily_position_change=1.0),
        )
        self.assertEqual(stress_result["regime"].iloc[0], "stress")
        self.assertAlmostEqual(stress_result["position"].iloc[0], 0.35)
        self.assertAlmostEqual(stress_result["turnover"].iloc[0], 0.35)
        self.assertAlmostEqual(
            stress_result["transaction_cost"].iloc[0], 0.35 * 5.0 / 10_000.0
        )
        self.assertAlmostEqual(
            stress_result["net_pnl"].iloc[0],
            stress_result["gross_pnl"].iloc[0]
            - stress_result["transaction_cost"].iloc[0],
        )

    def test_daily_loss_activates_cap_for_exactly_two_subsequent_decisions(self) -> None:
        frame = _position_frame([-0.05, 0.0, 0.0, 0.0, 0.0])
        result = backtest_positions(frame, forward_return_col="forward")

        self.assertLessEqual(result["net_pnl"].iloc[0], -0.01)
        self.assertEqual(list(result["post_loss_days_remaining"].iloc[1:4]), [2, 1, 0])
        self.assertLessEqual(abs(result["position"].iloc[1]), 0.25)
        self.assertLessEqual(abs(result["position"].iloc[2]), 0.25)
        self.assertGreater(abs(result["position"].iloc[3]), 0.25)

    def test_six_percent_drawdown_forces_immediate_flat_position(self) -> None:
        frame = _position_frame([-0.30, 0.0, 0.0])
        result = backtest_positions(frame, forward_return_col="forward")

        self.assertGreaterEqual(result["drawdown_before_trade"].iloc[1], 0.06)
        self.assertEqual(result["drawdown_multiplier"].iloc[1], 0.0)
        self.assertEqual(result["target_position"].iloc[1], 0.0)
        self.assertEqual(result["position"].iloc[1], 0.0)
        self.assertIn("drawdown_flat", result["cap_reason"].iloc[1])


if __name__ == "__main__":
    unittest.main()
