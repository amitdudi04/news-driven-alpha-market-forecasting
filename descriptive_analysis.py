"""Descriptive finance analysis for the committed research sample.

This module reproduces the public descriptive statistics directly from the
committed news, sentiment and CSI 300 files. It does not fit a predictive model.
"""

import os

import numpy as np
import pandas as pd

from module4_features import align_sentiment_to_trading_days, engineer_features


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    news = pd.read_csv(os.path.join("data", "news_daily.csv"))
    sentiment = pd.read_csv(os.path.join("data", "sentiment_features.csv"))
    market = pd.read_csv(os.path.join("data", "csi300_features.csv"))

    news["date"] = pd.to_datetime(news["date"])
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    market["date"] = pd.to_datetime(market["date"])
    return news.sort_values("date"), sentiment.sort_values("date"), market.sort_values("date")


def build_pairs(sentiment: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = align_sentiment_to_trading_days(sentiment, market)
    aligned = aligned.copy()
    aligned["next_session_return"] = aligned["return_t"].shift(-1)
    pairs = aligned.dropna(
        subset=["sentiment_mean_t", "next_session_return"]
    ).reset_index(drop=True)
    return aligned, pairs


def _spearman_without_scipy(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average")))


def sentiment_terciles(pairs: pd.DataFrame) -> pd.DataFrame:
    ordered = pairs.sort_values("sentiment_mean_t").reset_index(drop=True)
    n = len(ordered)
    base = n // 3
    groups = [
        ("lowest", ordered.iloc[:base]),
        ("middle", ordered.iloc[base : 2 * base]),
        ("highest", ordered.iloc[2 * base :]),
    ]

    rows = []
    for name, group in groups:
        if group.empty:
            continue
        rows.append(
            {
                "sentiment_group": name,
                "observations": int(len(group)),
                "mean_sentiment": float(group["sentiment_mean_t"].mean()),
                "mean_next_session_return": float(group["next_session_return"].mean()),
                "next_session_positive_rate": float(
                    (group["next_session_return"] > 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    news: pd.DataFrame,
    sentiment: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned, pairs = build_pairs(sentiment, market)
    model_rows = engineer_features(
        sentiment,
        market,
        keep_latest_unlabeled=False,
    )

    x = pairs["sentiment_mean_t"]
    y = pairs["next_session_return"]

    compounded_log_return = float(np.expm1(aligned["return_t"].sum()))
    close_to_close_change = float(
        aligned["close"].iloc[-1] / aligned["close"].iloc[0] - 1.0
    )
    annualized_realized_volatility = float(
        aligned["return_t"].std(ddof=1) * np.sqrt(252)
    )
    naive_hit_rate = float((np.sign(x) == np.sign(y)).mean())

    terciles = sentiment_terciles(pairs)
    low_return = float(
        terciles.loc[
            terciles["sentiment_group"] == "lowest",
            "mean_next_session_return",
        ].iloc[0]
    )
    high_return = float(
        terciles.loc[
            terciles["sentiment_group"] == "highest",
            "mean_next_session_return",
        ].iloc[0]
    )

    summary = pd.DataFrame(
        [
            {
                "news_start": news["date"].min().strftime("%Y-%m-%d"),
                "news_end": news["date"].max().strftime("%Y-%m-%d"),
                "news_days": int(len(news)),
                "sentiment_days": int(len(sentiment)),
                "headline_title_observations": int(news["article_count"].sum()),
                "mean_headlines_per_news_day": float(news["article_count"].mean()),
                "median_headlines_per_news_day": float(news["article_count"].median()),
                "mean_daily_sentiment": float(sentiment["sentiment_mean"].mean()),
                "market_sessions_in_aligned_window": int(len(aligned)),
                "market_window_start": aligned["date"].min().strftime("%Y-%m-%d"),
                "market_window_end": aligned["date"].max().strftime("%Y-%m-%d"),
                "up_sessions": int((aligned["return_t"] > 0).sum()),
                "down_sessions": int((aligned["return_t"] < 0).sum()),
                "compounded_return_from_stored_session_log_returns": compounded_log_return,
                "first_close_to_last_close_change": close_to_close_change,
                "annualized_realized_volatility": annualized_realized_volatility,
                "mean_20d_daily_volatility": float(aligned["volatility"].mean()),
                "sessions_with_usable_news": int(aligned["sentiment_mean_t"].notna().sum()),
                "sessions_without_usable_news": int(aligned["sentiment_mean_t"].isna().sum()),
                "sentiment_next_return_pairs": int(len(pairs)),
                "spearman_time_series_rank_correlation": _spearman_without_scipy(x, y),
                "pearson_correlation": float(x.corr(y)),
                "naive_sentiment_sign_hit_rate": naive_hit_rate,
                "low_minus_high_sentiment_return_spread": low_return - high_return,
                "high_minus_low_sentiment_return_spread": high_return - low_return,
                "labelled_model_rows": int(len(model_rows)),
            }
        ]
    )
    return summary, terciles


def main():
    news, sentiment, market = load_inputs()
    summary, terciles = summarize(news, sentiment, market)

    os.makedirs("outputs", exist_ok=True)
    summary.to_csv("outputs/descriptive_summary.csv", index=False)
    terciles.to_csv("outputs/sentiment_terciles.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
