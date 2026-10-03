import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_FEATURES = [
    "volatility",
    "sentiment_roll_5",
    "sentiment_roll_10",
    "sentiment_zscore",
    "sentiment_momentum",
    "sentiment_volatility",
    "news_intensity",
    "momentum",
    "momentum_acceleration",
    "sentiment_x_volatility",
    "sentiment_zscore_x_volatility",
    "regime_dummy",
]

MARKET_ONLY_FEATURES = [
    "volatility",
    "momentum",
    "momentum_acceleration",
    "regime_dummy",
]


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    sent_path = os.path.join(os.getcwd(), "data", "sentiment_features.csv")
    market_path = os.path.join(os.getcwd(), "data", "csi300_features.csv")
    if not os.path.exists(sent_path) or not os.path.exists(market_path):
        raise FileNotFoundError("Run the sentiment and CSI 300 data modules first.")

    sent_df = pd.read_csv(sent_path)
    market_df = pd.read_csv(market_path)
    sent_df["date"] = pd.to_datetime(sent_df["date"])
    market_df["date"] = pd.to_datetime(market_df["date"])
    return sent_df.sort_values("date"), market_df.sort_values("date")


def _pool_sentiment(block: pd.DataFrame) -> tuple[float, float, int]:
    """Article-weighted aggregation of calendar-day sentiment observations."""
    if block.empty:
        return 0.0, 0.0, 0

    counts = pd.to_numeric(block["article_count"], errors="coerce").fillna(0).clip(lower=0)
    means = pd.to_numeric(block["sentiment_mean"], errors="coerce").fillna(0.0)
    stds = pd.to_numeric(block["sentiment_std"], errors="coerce").fillna(0.0).clip(lower=0)
    total = int(counts.sum())
    if total <= 0:
        return 0.0, 0.0, 0

    weights = counts / counts.sum()
    pooled_mean = float(np.sum(weights * means))
    pooled_var = float(np.sum(weights * (stds**2 + (means - pooled_mean) ** 2)))
    return pooled_mean, float(np.sqrt(max(pooled_var, 0.0))), total


def align_sentiment_to_trading_days(sent_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Map calendar-day news to CSI 300 trading sessions without creating weekend prices.

    For trading day t, sentiment is aggregated from the previous trading date
    (exclusive) through t (inclusive). The row is therefore interpreted as an
    end-of-day t information set used to forecast the next trading session.
    """
    sent = sent_df.copy().sort_values("date").reset_index(drop=True)
    market = market_df.copy().sort_values("date").reset_index(drop=True)

    if sent.empty:
        raise ValueError("Sentiment dataset is empty.")
    # Do not create a long pre-news training history by treating missing news as
    # neutral. The research sample begins when genuine sentiment observations begin.
    market = market[market["date"] >= sent["date"].min()].reset_index(drop=True)

    required_sent = {"date", "sentiment_mean", "sentiment_std", "article_count"}
    required_market = {"date", "close", "return", "volatility"}
    if not required_sent.issubset(sent.columns):
        raise ValueError(f"Sentiment data missing columns: {sorted(required_sent - set(sent.columns))}")
    if not required_market.issubset(market.columns):
        raise ValueError(f"Market data missing columns: {sorted(required_market - set(market.columns))}")

    rows = []
    for i, mrow in market.iterrows():
        current_date = mrow["date"]
        previous_date = market.loc[i - 1, "date"] if i > 0 else current_date - pd.Timedelta(days=1)
        block = sent[(sent["date"] > previous_date) & (sent["date"] <= current_date)]
        s_mean, s_std, article_count = _pool_sentiment(block)
        rows.append(
            {
                "date": current_date,
                "close": float(mrow["close"]),
                "return_t": float(mrow["return"]),
                "volatility": float(mrow["volatility"]),
                "sentiment_mean_t": s_mean,
                "sentiment_std_t": s_std,
                "article_count_t": article_count,
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def engineer_features(sent_df: pd.DataFrame, market_df: pd.DataFrame, keep_latest_unlabeled: bool = True) -> pd.DataFrame:
    df = align_sentiment_to_trading_days(sent_df, market_df)

    df["sentiment_roll_5"] = df["sentiment_mean_t"].rolling(5).mean()
    df["sentiment_roll_10"] = df["sentiment_mean_t"].rolling(10).mean()
    rolling_mean_20 = df["sentiment_mean_t"].rolling(20).mean()
    rolling_std_20 = df["sentiment_mean_t"].rolling(20).std()
    df["sentiment_zscore"] = (df["sentiment_mean_t"] - rolling_mean_20) / (rolling_std_20 + 1e-8)
    df["sentiment_momentum"] = df["sentiment_mean_t"].diff()
    df["sentiment_volatility"] = rolling_std_20
    df["news_intensity"] = df["article_count_t"] / (df["article_count_t"].rolling(7).mean() + 1e-8)

    ma_short = df["close"].rolling(5).mean()
    ma_long = df["close"].rolling(20).mean()
    df["momentum"] = (ma_short / ma_long) - 1.0
    df["momentum_acceleration"] = df["momentum"].diff()

    df["sentiment_x_volatility"] = df["sentiment_roll_5"] * df["volatility"]
    df["sentiment_zscore_x_volatility"] = df["sentiment_zscore"] * df["volatility"]

    prior_median_vol = df["volatility"].shift(1).expanding(min_periods=20).median()
    df["regime_dummy"] = np.where(
        prior_median_vol.notna(),
        (df["volatility"] > prior_median_vol).astype(int),
        np.nan,
    )

    df["target_return_t+1"] = df["return_t"].shift(-1)
    df["target_volatility_t+1"] = df["volatility"].shift(-1)

    df = df.dropna(subset=MODEL_FEATURES).reset_index(drop=True)
    if not keep_latest_unlabeled:
        df = df.dropna(subset=["target_return_t+1"]).reset_index(drop=True)

    columns = [
        "date",
        "close",
        "return_t",
        "sentiment_mean_t",
        "sentiment_std_t",
        "article_count_t",
        "target_return_t+1",
        "target_volatility_t+1",
    ] + MODEL_FEATURES
    return df[columns]


def generate_latest_features(sent_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    dataset = engineer_features(sent_df, market_df, keep_latest_unlabeled=True)
    if dataset.empty:
        raise ValueError("Insufficient history to construct the model features.")
    return dataset.iloc[[-1]][["date"] + MODEL_FEATURES].copy()


def main(execution_uuid: str | None = None):
    del execution_uuid
    logging.info("Building trading-session-aligned research features")
    sent_df, market_df = load_datasets()
    final_df = engineer_features(sent_df, market_df, keep_latest_unlabeled=True)

    out_path = os.path.join(os.getcwd(), "data", "final_dataset.csv")
    final_df = final_df.copy()
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime("%Y-%m-%d")
    final_df.to_csv(out_path, index=False)
    logging.info("Saved %s rows to %s", len(final_df), out_path)


if __name__ == "__main__":
    main()
