import datetime as dt
import logging
import os
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '(China OR PBOC OR Beijing) (economy OR "stock market" OR "financial markets" OR regulation) sourcelang:english'
NON_EMPIRICAL_PATTERNS = (
    "China PBOC announces new liquidity measures to stabilize markets on",
    "China PBOC economy stock market financial markets regulation",
)


def _request_window(start: dt.datetime, end: dt.datetime, query: str = QUERY) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(GDELT_URL, params=params, timeout=20)
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            return pd.DataFrame(payload.get("articles", []))
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"GDELT request failed after retries: {last_error}")


def fetch_historical_news(
    start_date: dt.date,
    end_date: dt.date,
    query: str = QUERY,
) -> pd.DataFrame:
    frames = []
    current = start_date
    while current <= end_date:
        start = dt.datetime.combine(current, dt.time.min, tzinfo=dt.timezone.utc)
        end = dt.datetime.combine(current, dt.time.max, tzinfo=dt.timezone.utc)
        frame = _request_window(start, end, query)
        if not frame.empty:
            frames.append(frame)
        current += dt.timedelta(days=1)
        time.sleep(1.0)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_latest_news(query: str = QUERY, lookback_days: int = 3) -> pd.DataFrame:
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=lookback_days)
    return _request_window(start, end, query)


def process_news(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "article_count", "raw_text"])

    required = {"seendate", "title"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"GDELT payload missing columns: {sorted(missing)}")

    df = raw.copy()
    df["datetime_utc"] = pd.to_datetime(
        df["seendate"],
        format="%Y%m%dT%H%M%SZ",
        errors="coerce",
        utc=True,
    )
    df["text"] = df["title"].astype(str).str.strip()
    df = df.dropna(subset=["datetime_utc"])
    df = df[df["text"].str.len() > 15]
    df = df.drop_duplicates(subset=["datetime_utc", "text"])

    shanghai_time = df["datetime_utc"].dt.tz_convert("Asia/Shanghai")
    df["date"] = shanghai_time.dt.strftime("%Y-%m-%d")

    if df["text"].str.contains("|".join(NON_EMPIRICAL_PATTERNS), regex=False).any():
        raise ValueError("Known development seed text detected in incoming news data.")

    return (
        df.groupby("date")
        .agg(
            article_count=("text", "count"),
            raw_text=("text", lambda x: " || ".join(x)),
        )
        .reset_index()
        .sort_values("date")
    )


def combine_daily_news(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = old_df.copy()
    else:
        combined = pd.concat([old_df, new_df], ignore_index=True)

    if combined.empty:
        return combined

    def merge_articles(series):
        items = []
        seen = set()
        for block in series:
            for article in str(block).split(" || "):
                article = article.strip()
                if article and article not in seen:
                    seen.add(article)
                    items.append(article)
        return " || ".join(items), len(items)

    rows = []
    for date, group in combined.groupby("date", sort=True):
        merged, count = merge_articles(group["raw_text"])
        if any(pattern in merged for pattern in NON_EMPIRICAL_PATTERNS):
            raise ValueError(f"Non-empirical seed text detected for {date}.")
        rows.append({"date": date, "article_count": count, "raw_text": merged})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main(execution_uuid: str | None = None):
    del execution_uuid
    out_path = os.path.join(os.getcwd(), "data", "news_daily.csv")

    if os.path.exists(out_path):
        old_df = pd.read_csv(out_path)
        raw = get_latest_news()
    else:
        old_df = pd.DataFrame(columns=["date", "article_count", "raw_text"])
        end_date = dt.date.today()
        backfill_days = int(os.environ.get("NEWS_BACKFILL_DAYS", "365"))
        raw = fetch_historical_news(
            end_date - dt.timedelta(days=backfill_days),
            end_date,
        )

    processed = process_news(raw)
    combined = combine_daily_news(old_df, processed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)
    logging.info("Saved %s daily news observations", len(combined))


if __name__ == "__main__":
    main()
