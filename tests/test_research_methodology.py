import unittest

import pandas as pd

from module4_features import align_sentiment_to_trading_days
from module13_signal_engine import generate_signal


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
        self.assertAlmostEqual(
            monday["sentiment_mean_t"],
            0.4,
            places=8,
        )
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

    def test_no_weekend_market_rows_are_created(self):
        aligned = align_sentiment_to_trading_days(self.sent, self.market)
        self.assertListEqual(
            list(aligned["date"]),
            list(self.market["date"]),
        )


class SignalRuleTests(unittest.TestCase):
    def test_symmetric_probability_rule(self):
        self.assertEqual(
            generate_signal(0.60, 0.02, 0.02)["signal"],
            "LONG",
        )
        self.assertEqual(
            generate_signal(0.40, 0.02, 0.02)["signal"],
            "SHORT",
        )
        self.assertEqual(
            generate_signal(0.50, 0.02, 0.02)["signal"],
            "NO TRADE",
        )

    def test_position_never_exceeds_one(self):
        result = generate_signal(
            0.60,
            pred_volatility=0.001,
            target_volatility=0.02,
        )
        self.assertLessEqual(abs(result["position"]), 1.0)


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

    def test_sentiment_counts_match_committed_news_counts(self):
        news = pd.read_csv("data/news_daily.csv")[["date", "article_count"]]
        sentiment = pd.read_csv("data/sentiment_features.csv")[["date", "article_count"]]
        merged = news.merge(
            sentiment,
            on="date",
            how="inner",
            suffixes=("_news", "_sentiment"),
        )
        self.assertGreater(len(merged), 0)
        self.assertTrue(
            (merged["article_count_news"] == merged["article_count_sentiment"]).all()
        )


if __name__ == "__main__":
    unittest.main()
