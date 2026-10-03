import datetime as dt
import logging
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TICKER = "000300.SS"


def download_csi300(start: str, end: str) -> pd.DataFrame:
    last_error = None
    for attempt in range(3):
        try:
            df = yf.download(TICKER, start=start, end=end, progress=False)
            if df.empty:
                raise RuntimeError(f"No data returned for {TICKER}")
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(
                columns={
                    "Date": "date",
                    "Close": "close",
                }
            )
            return df
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CSI 300 download failed after retries: {last_error}")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("date").reset_index(drop=True)
    data = data.dropna(subset=["close"])
    data["return"] = np.log(data["close"] / data["close"].shift(1))
    data["volatility"] = data["return"].rolling(20).std()
    data = data.dropna(subset=["return", "volatility"])
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    return data[["date", "close", "return", "volatility"]]


def main(execution_uuid: str | None = None):
    del execution_uuid
    end_date = dt.date.today() + dt.timedelta(days=1)

    sentiment_path = os.path.join(os.getcwd(), "data", "sentiment_features.csv")
    if os.path.exists(sentiment_path):
        sent = pd.read_csv(sentiment_path)
        first_sentiment = pd.to_datetime(sent["date"], errors="coerce").min()
        if pd.notna(first_sentiment):
            start_date = first_sentiment.date() - dt.timedelta(days=60)
        else:
            start_date = end_date - dt.timedelta(days=365)
    else:
        start_date = end_date - dt.timedelta(days=365)

    raw = download_csi300(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )
    final_df = compute_features(raw)

    out_path = os.path.join(os.getcwd(), "data", "csi300_features.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logging.info("Saved %s CSI 300 observations", len(final_df))


if __name__ == "__main__":
    main()
