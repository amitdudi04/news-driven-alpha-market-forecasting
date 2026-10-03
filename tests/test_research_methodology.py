import unittest

import numpy as np
import pandas as pd

from descriptive_analysis import load_inputs, summarize
from module4_features import align_sentiment_to_trading_days
from module13_signal_engine import generate_signal
from module7_backtesting import build_backtest


class TradingCalendarTests(unittest.TestCase):
    def setUp(self):
        self.market = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-02", "2026-01-05", "2026-01-06"]
                ),
                "close": [100.0, 101.0, 99.0],
                "return": [0.01, 0.00995, -0.0200],
                "volatility": [0.01, 0.011, 0.012],
            }
        )
        self.sent = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                        "2026-01-05",
                        "2026-01-06",
                    ]
                ),
                "sentiment_mean": [0.1, 0.2, 0.4, 0.6, -0.2],
                "sentiment_std": [0.0, 0.0, 0.0, 0.0, 0.0],
                "article_count": [10, 10, 10, 10, 10],
            }
        )

    def test_weekend_news_is_aggregated_to_monday_information_set(self):
        aligned = align_sentiment_to_trading_days(self.sent, self.market)
        monday = aligned.loc[
            aligned["date"] == pd.Timestamp("2026-01-05")
        ].iloc[0]
        self.assertAlmostEqual(monday["sentiment_mean_t"], 0.4, places=8)
        self.assertEqual(int(monday["article_count_t"]), 30)

    def test_pre_news_market_history_is_not_treated_as_neutral_sentiment(self):
        earlier_market = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2025-12-31"]),
                        "close": [98.0],
                        "return": [0.002],
                        "volatility": [0.009],
                    }
                ),
                self.market,
            ],
            ignore_index=True,
        )
        aligned = align_sentiment_to_trading_days(self.sent, earlier_market)
        self.assertEqual(aligned["date"].min(), pd.Timestamp("2026-01-02"))

    def test_missing_news_interval_is_not_neutralized(self):
        incomplete_sent = self.sent[
            self.sent["date"] != pd.Timestamp("2026-01-06")
        ].copy()
        aligned = align_sentiment_to_trading_days(incomplete_sent, self.market)
        tuesday = aligned.loc[
            aligned["date"] == pd.Timestamp("2026-01-06")
        ].iloc[0]
        self.assertTrue(pd.isna(tuesday["sentiment_mean_t"]))

    def test_no_weekend_market_rows_are_created(self):
        aligned = align_sentiment_to_trading_days(self.sent, self.market)
        self.assertListEqual(list(aligned["date"]), list(self.market["date"]))


class SignalRuleTests(unittest.TestCase):
    def test_symmetric_probability_rule(self):
        self.assertEqual(generate_signal(0.60, 0.02, 0.02)["signal"], "LONG")
        self.assertEqual(generate_signal(0.40, 0.02, 0.02)["signal"], "SHORT")
        self.assertEqual(generate_signal(0.50, 0.02, 0.02)["signal"], "NO TRADE")

    def test_position_never_exceeds_one(self):
        result = generate_signal(
            0.60,
            pred_volatility=0.001,
            target_volatility=0.02,
        )
        self.assertLessEqual(abs(result["position"]), 1.0)


class BacktestTimingTests(unittest.TestCase):
    def _garch_history(self):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-06",
                        "2026-01-07",
                        "2026-01-08",
                    ]
                ),
                "pred_vol_t+1": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            }
        )

    def test_risk_target_uses_all_prior_garch_forecasts(self):
        predictions = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-08")],
                "actual_return_t+1": [0.01],
                "direction_probability_full": [0.60],
            }
        )
        result = build_backtest(predictions, self._garch_history())
        self.assertAlmostEqual(
            float(result["target_volatility"].iloc[0]),
            0.03,
            places=10,
        )

    def test_log_market_return_is_converted_before_strategy_compounding(self):
        predictions = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-08")],
                "actual_return_t+1": [0.01],
                "direction_probability_full": [0.60],
            }
        )
        result = build_backtest(predictions, self._garch_history())
        expected_simple = float(np.expm1(0.01))
        self.assertAlmostEqual(
            float(result["benchmark_return"].iloc[0]),
            expected_simple,
            places=12,
        )
        self.assertAlmostEqual(
            float(result["benchmark_wealth"].iloc[0]),
            1.0 + expected_simple,
            places=12,
        )


class DescriptiveResultsTests(unittest.TestCase):
    def test_public_descriptive_statistics_are_reproducible(self):
        news, sentiment, market = load_inputs()
        summary, terciles = summarize(news, sentiment, market)
        row = summary.iloc[0]

        self.assertEqual(int(row["news_days"]), 49)
        self.assertEqual(int(row["headline_title_observations"]), 6919)
        self.assertEqual(int(row["market_sessions_in_aligned_window"]), 49)
        self.assertEqual(int(row["sentiment_next_return_pairs"]), 31)
        self.assertEqual(int(row["labelled_model_rows"]), 8)
        self.assertAlmostEqual(
            float(row["spearman_time_series_rank_correlation"]),
            -0.3479838709677419,
            places=10,
        )
        self.assertAlmostEqual(
            float(row["pearson_correlation"]),
            -0.24574751260633484,
            places=10,
        )
        self.assertAlmostEqual(
            float(row["compounded_return_from_stored_session_log_returns"]),
            0.015555772205326897,
            places=10,
        )
        self.assertAlmostEqual(
            float(row["first_close_to_last_close_change"]),
            0.0088631915587567,
            places=10,
        )
        self.assertAlmostEqual(
            float(row["annualized_realized_volatility"]),
            0.20784906665718886,
            places=10,
        )

        low = terciles.loc[
            terciles["sentiment_group"] == "lowest"
        ].iloc[0]
        high = terciles.loc[
            terciles["sentiment_group"] == "highest"
        ].iloc[0]
        self.assertAlmostEqual(
            float(low["mean_next_session_return"]),
            0.0030498320398687786,
            places=10,
        )
        self.assertAlmostEqual(
            float(high["mean_next_session_return"]),
            -0.0016130031782305516,
            places=10,
        )


class CommittedDataTests(unittest.TestCase):
    def test_committed_news_sample_contains_no_known_seed_rows(self):
        news = pd.read_csv("data/news_daily.csv")
        text = news["raw_text"].astype(str)
        self.assertFalse(
            text.str.contains(
                "China PBOC announces new liquidity measures to stabilize markets on",
                regex=False,
            ).any()
        )
        self.assertFalse(
            text.str.contains(
                "China PBOC economy stock market financial markets regulation",
                regex=False,
            ).any()
        )
        self.assertGreaterEqual(
            pd.to_datetime(news["date"]).min(),
            pd.Timestamp("2026-04-22"),
        )

    def test_committed_market_schema_is_research_only(self):
        market = pd.read_csv("data/csi300_features.csv")
        self.assertListEqual(
            list(market.columns),
            ["date", "close", "return", "volatility"],
        )
        self.assertFalse(market["date"].duplicated().any())

    def test_sentiment_counts_match_committed_news_counts(self):
        news = pd.read_csv("data/news_daily.csv")[["date", "article_count"]]
        sentiment = pd.read_csv("data/sentiment_features.csv")[
            ["date", "article_count"]
        ]
        merged = news.merge(
            sentiment,
            on="date",
            how="inner",
            suffixes=("_news", "_sentiment"),
        )
        self.assertGreater(len(merged), 0)
        self.assertTrue(
            (
                merged["article_count_news"]
                == merged["article_count_sentiment"]
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
