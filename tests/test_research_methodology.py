import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_master_session_dataset import (
    add_predictors,
    add_targets,
    aggregate_unique_headlines,
    build_master,
    normalize_title,
)
from build_timestamp_safe_alignment import month_ranges
from evaluate_oos_uncertainty import (
    circular_block_indices,
    delta_metrics,
)
from headline_utils import split_headlines
from run_garch_oos_simulation import simulate_position_path


ROOT = Path(__file__).resolve().parents[1]


class HistoricalHeadlineSerializationTests(unittest.TestCase):
    def test_json_headlines_preserve_literal_legacy_delimiter(self):
        headlines = [
            "Secure Their Future: Market || China Ping An Insurance",
            "China economy growth outlook improves",
        ]
        payload = json.dumps(headlines)
        parsed = split_headlines(
            "legacy text should not be used",
            payload,
        )
        self.assertEqual(parsed, headlines)


class SessionHeadlineDedupTests(unittest.TestCase):
    def test_normalized_title_collapses_case_and_whitespace(self):
        self.assertEqual(
            normalize_title("  China   MARKET Update  "),
            "china market update",
        )

    def test_session_level_repeated_titles_are_removed(self):
        rows = pd.DataFrame(
            {
                "assigned_session_date": [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-02",
                ],
                "gdelt_seen_time_shanghai": [
                    "2025-01-02 10:00:00+08:00",
                    "2025-01-02 11:00:00+08:00",
                    "2025-01-02 12:00:00+08:00",
                ],
                "title": [
                    "China Market Update",
                    " china   market update ",
                    "China Growth Outlook",
                ],
                "sentiment_score": [0.2, 0.8, -0.1],
                "predicted_label": [
                    "positive",
                    "positive",
                    "negative",
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aligned.csv"
            rows.to_csv(path, index=False)
            grouped, audit = aggregate_unique_headlines(path)

        self.assertEqual(
            audit["assigned_headline_observations"],
            3,
        )
        self.assertEqual(
            audit["session_unique_normalized_headlines"],
            2,
        )
        self.assertEqual(
            audit["within_session_repeated_headlines_removed"],
            1,
        )
        self.assertEqual(
            int(grouped["unique_headline_count_t"].iloc[0]),
            2,
        )
        # Earliest duplicate is retained deterministically: (0.2 + -0.1)/2
        self.assertAlmostEqual(
            float(grouped["sentiment_mean_t"].iloc[0]),
            0.05,
            places=12,
        )


class MasterFeatureTests(unittest.TestCase):
    def _base_predictor_frame(self, n=25):
        return pd.DataFrame(
            {
                "date": pd.date_range("2025-01-02", periods=n, freq="B"),
                "window_start_shanghai": ["x"] * n,
                "window_end_shanghai": ["y"] * n,
                "close": np.arange(n, dtype=float) + 100.0,
                "return_t": np.linspace(-0.01, 0.01, n),
                "volatility_20_t": np.full(n, 0.012),
                "momentum_5_20_t": np.full(n, 0.01),
                "momentum_acceleration_t": np.zeros(n),
                "regime_dummy_t": np.zeros(n),
                "unique_headline_count_t": np.full(n, 10),
                "headline_observation_count_t": np.full(n, 10),
                "sentiment_mean_t": np.linspace(-0.1, 0.1, n),
                "sentiment_std_t": np.full(n, 0.2),
                "positive_share_t": np.full(n, 0.4),
                "negative_share_t": np.full(n, 0.3),
                "neutral_share_t": np.full(n, 0.3),
                "has_news_t": np.ones(n, dtype=int),
            }
        )

    def test_news_intensity_uses_prior_20_sessions_only(self):
        data = self._base_predictor_frame(21)
        data.loc[20, "unique_headline_count_t"] = 20
        out = add_predictors(data)
        self.assertAlmostEqual(
            float(
                out.loc[
                    20,
                    "prior_20_session_mean_unique_headlines",
                ]
            ),
            10.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(out.loc[20, "news_intensity_20"]),
            2.0,
            places=12,
        )

    def test_strict_rolling_sentiment_propagates_missingness(self):
        data = self._base_predictor_frame(25)
        data.loc[10, "sentiment_mean_t"] = np.nan
        out = add_predictors(data)

        # 5-session window containing the missing observation stays missing.
        for idx in range(10, 15):
            self.assertTrue(
                pd.isna(out.loc[idx, "sentiment_roll_5"])
            )
        self.assertFalse(
            pd.isna(out.loc[15, "sentiment_roll_5"])
        )

    def test_next_session_target_is_exact_chronological_shift(self):
        data = self._base_predictor_frame(5)
        out = add_targets(data)
        self.assertEqual(
            out.loc[0, "target_session_date"],
            out.loc[1, "date"],
        )
        self.assertAlmostEqual(
            float(out.loc[0, "target_return_t_plus_1"]),
            float(out.loc[1, "return_t"]),
            places=12,
        )
        self.assertTrue(
            pd.isna(out.loc[4, "target_return_t_plus_1"])
        )
        self.assertEqual(
            int(out.loc[4, "target_available"]),
            0,
        )

    def test_no_news_count_is_zero_but_sentiment_remains_missing(self):
        dates = pd.date_range("2025-01-02", periods=25, freq="B")
        market = pd.DataFrame(
            {
                "date": dates,
                "window_start_shanghai": ["x"] * 25,
                "window_end_shanghai": ["y"] * 25,
                "close": np.arange(25, dtype=float) + 100.0,
                "return": np.linspace(-0.01, 0.01, 25),
                "volatility": np.full(25, 0.012),
                "momentum": np.full(25, 0.01),
                "momentum_acceleration": np.zeros(25),
                "regime_dummy": np.zeros(25),
            }
        )
        news_dates = dates.delete(12)
        news = pd.DataFrame(
            {
                "assigned_session_date": news_dates,
                "unique_headline_count_t": np.full(24, 10),
                "sentiment_mean_t": np.full(24, 0.1),
                "sentiment_std_t": np.full(24, 0.2),
                "positive_share_t": np.full(24, 0.4),
                "negative_share_t": np.full(24, 0.3),
                "neutral_share_t": np.full(24, 0.3),
                "headline_observation_count_t": np.full(24, 10),
            }
        )
        out = build_master(market, news)
        row = out.iloc[12]
        self.assertEqual(
            int(row["unique_headline_count_t"]),
            0,
        )
        self.assertEqual(int(row["has_news_t"]), 0)
        self.assertTrue(pd.isna(row["sentiment_mean_t"]))
        self.assertTrue(pd.isna(row["sentiment_std_t"]))


class TimestampWindowTests(unittest.TestCase):
    def test_month_ranges_respect_partial_boundary_months(self):
        ranges = list(
            month_ranges(
                dt.date(2026, 9, 28),
                dt.date(2026, 10, 3),
            )
        )
        self.assertEqual(
            ranges,
            [
                (
                    dt.date(2026, 9, 28),
                    dt.date(2026, 9, 30),
                ),
                (
                    dt.date(2026, 10, 1),
                    dt.date(2026, 10, 3),
                ),
            ],
        )


class FrozenSpecificationTests(unittest.TestCase):
    def _load(self, relative):
        return json.loads(
            (ROOT / relative).read_text(encoding="utf-8")
        )

    def test_directional_split_is_target_date_based_and_frozen(self):
        spec = self._load(
            "config/directional_experiment_2023_2026.json"
        )
        self.assertTrue(spec["frozen_before_model_fit"])
        self.assertEqual(
            spec["partition_key"],
            "target_session_date",
        )
        self.assertEqual(
            spec["split"]["untouched_final_holdout"][
                "target_year"
            ],
            2025,
        )
        self.assertEqual(
            spec["split"]["post_sample_robustness"][
                "target_year"
            ],
            2026,
        )
        self.assertEqual(
            len(spec["master_dataset_sha256"]),
            64,
        )

    def test_uncertainty_spec_is_paired_block_bootstrap(self):
        spec = self._load(
            "config/oos_uncertainty_spec.json"
        )
        boot = spec["bootstrap"]
        self.assertEqual(boot["replications"], 5000)
        self.assertEqual(
            boot["block_length_sessions"],
            10,
        )
        self.assertTrue(boot["paired_indices"])

    def test_garch_simulation_conventions_are_frozen(self):
        spec = self._load(
            "config/garch_simulation_spec.json"
        )
        signal = spec["directional_signal"]
        self.assertAlmostEqual(
            signal["long_probability_threshold"],
            0.55,
        )
        self.assertAlmostEqual(
            signal["short_probability_threshold"],
            0.45,
        )
        self.assertFalse(signal["thresholds_optimized"])
        self.assertAlmostEqual(
            spec["transaction_cost"][
                "rate_per_unit_turnover"
            ],
            0.001,
        )
        self.assertFalse(
            spec["transaction_cost"]["optimized"]
        )


class BootstrapMethodTests(unittest.TestCase):
    def test_circular_block_indices_are_bounded_and_blockwise_consecutive(self):
        rng = np.random.default_rng(123)
        idx = circular_block_indices(
            n=20,
            block_length=5,
            rng=rng,
        )
        self.assertEqual(len(idx), 20)
        self.assertTrue(((idx >= 0) & (idx < 20)).all())
        for start in range(0, 20, 5):
            block = idx[start : start + 5]
            expected = (
                block[0] + np.arange(5)
            ) % 20
            np.testing.assert_array_equal(
                block,
                expected,
            )

    def test_increment_signs_favor_better_sentiment_probabilities(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p_market = np.full(len(y), 0.5)
        p_sentiment = np.array(
            [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
        )
        result = delta_metrics(
            y,
            p_market,
            p_sentiment,
            bins=3,
        )
        self.assertGreater(
            result["balanced_accuracy_improvement"],
            0,
        )
        self.assertGreater(
            result["brier_improvement"],
            0,
        )
        self.assertGreater(
            result["log_loss_improvement"],
            0,
        )
        self.assertGreater(
            result["roc_auc_improvement"],
            0,
        )


class SimulationAccountingTests(unittest.TestCase):
    def test_turnover_cost_and_net_return_are_exact(self):
        frame = pd.DataFrame(
            {
                "signal": [1.0, -1.0, 0.0],
                "position": [1.0, -0.5, 0.0],
                "benchmark_return": [0.01, -0.02, 0.03],
            }
        )
        out, metrics = simulate_position_path(
            frame,
            position_col="position",
            cost_rate=0.001,
        )
        expected_turnover = np.array(
            [1.0, 1.5, 0.5]
        )
        np.testing.assert_allclose(
            out["turnover"].to_numpy(),
            expected_turnover,
        )
        np.testing.assert_allclose(
            out["transaction_cost"].to_numpy(),
            expected_turnover * 0.001,
        )
        np.testing.assert_allclose(
            out["net_strategy_return"].to_numpy(),
            out["gross_strategy_return"].to_numpy()
            - out["transaction_cost"].to_numpy(),
        )
        self.assertAlmostEqual(
            metrics["total_transaction_cost_rate"],
            float((expected_turnover * 0.001).sum()),
            places=12,
        )

    def test_simulation_period_starts_from_flat_position(self):
        frame = pd.DataFrame(
            {
                "signal": [1.0],
                "position": [0.7],
                "benchmark_return": [0.01],
            }
        )
        out, _ = simulate_position_path(
            frame,
            position_col="position",
            cost_rate=0.001,
        )
        self.assertAlmostEqual(
            float(out["turnover"].iloc[0]),
            0.7,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
