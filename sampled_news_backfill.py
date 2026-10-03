"""Relevance-ranked historical GDELT headline sampling.

The historical experiment does not attempt to treat the DOC API ArticleList as
an exhaustive archive. For each Shanghai calendar day it requests a fixed
relevance-ranked candidate set, removes duplicate/syndicated headline text, and
selects up to a fixed number of unique headlines for FinBERT scoring.

Total matching-news volume is collected separately through TimelineVolRaw.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = (
    '(China OR PBOC OR Beijing) '
    '(economy OR "stock market" OR "financial markets" OR regulation) '
    'sourcelang:english'
)
SORT_MODE = "hybridrel"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def shanghai_day_utc_bounds(
    run_date: dt.date,
) -> tuple[dt.datetime, dt.datetime]:
    china_tz = dt.timezone(dt.timedelta(hours=8))
    local_start = dt.datetime.combine(
        run_date,
        dt.time.min,
        tzinfo=china_tz,
    )
    local_end = dt.datetime.combine(
        run_date,
        dt.time.max,
        tzinfo=china_tz,
    )
    return (
        local_start.astimezone(dt.timezone.utc),
        local_end.astimezone(dt.timezone.utc),
    )


def title_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def headline_hash(titles: list[str]) -> str:
    normalized = " || ".join(title_key(value) for value in titles)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def request_daily_candidates(
    session: requests.Session,
    run_date: dt.date,
    query: str,
    request_records: int,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[pd.DataFrame, float]:
    start, end = shanghai_day_utc_bounds(run_date)
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": request_records,
        "sort": SORT_MODE,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        started = time.time()
        try:
            response = session.get(
                GDELT_URL,
                params=params,
                timeout=timeout_seconds,
            )
            elapsed = time.time() - started

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait_seconds = max(int(retry_after), 30)
                else:
                    wait_seconds = min(
                        30 * (2**attempt),
                        600,
                    )
                logging.warning(
                    "GDELT 429 for %s; waiting %ss before retry %s/%s",
                    run_date,
                    wait_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()
            return (
                pd.DataFrame(payload.get("articles", [])),
                elapsed,
            )
        except Exception as exc:
            last_error = exc
            wait_seconds = min(10 * (2**attempt), 180)
            logging.warning(
                "GDELT request error for %s: %s; waiting %ss",
                run_date,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"GDELT daily sample failed for {run_date}: {last_error}"
    )


def normalize_and_select(
    raw: pd.DataFrame,
    run_date: dt.date,
    target_headlines: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        empty = pd.DataFrame(
            columns=[
                "date",
                "datetime_utc",
                "api_rank",
                "title",
                "url",
                "domain",
                "sourcecountry",
                "title_key",
                "selected_for_sentiment",
            ]
        )
        return empty, empty.copy()

    required = {"seendate", "title"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"GDELT payload missing columns: {sorted(missing)}"
        )

    frame = raw.copy().reset_index(drop=True)
    frame["api_rank"] = frame.index + 1
    frame["datetime_utc"] = pd.to_datetime(
        frame["seendate"],
        format="%Y%m%dT%H%M%SZ",
        errors="coerce",
        utc=True,
    )
    frame["title"] = (
        frame["title"].fillna("").astype(str).str.strip()
    )
    for column in ["url", "domain", "sourcecountry"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = (
            frame[column].fillna("").astype(str).str.strip()
        )

    frame = frame.dropna(subset=["datetime_utc"])
    frame = frame[frame["title"].str.len() > 15].copy()

    shanghai_time = frame["datetime_utc"].dt.tz_convert(
        "Asia/Shanghai"
    )
    frame["date"] = shanghai_time.dt.strftime("%Y-%m-%d")
    frame = frame[
        frame["date"] == run_date.isoformat()
    ].copy()

    frame["title_key"] = frame["title"].map(title_key)
    frame["url_key"] = frame["url"].str.casefold()
    frame["url_dedup_key"] = frame["url_key"].where(
        frame["url_key"].str.len() > 0,
        frame["datetime_utc"].astype(str)
        + "|"
        + frame["title_key"],
    )
    frame = frame.drop_duplicates(
        subset=["url_dedup_key"],
        keep="first",
    )

    unique_titles = frame.drop_duplicates(
        subset=["title_key"],
        keep="first",
    ).head(target_headlines)

    selected_keys = set(unique_titles["title_key"])
    frame["selected_for_sentiment"] = frame["title_key"].isin(
        selected_keys
    ) & ~frame.duplicated(subset=["title_key"], keep="first")

    columns = [
        "date",
        "datetime_utc",
        "api_rank",
        "title",
        "url",
        "domain",
        "sourcecountry",
        "title_key",
        "selected_for_sentiment",
    ]
    provenance = (
        frame[columns]
        .sort_values("api_rank")
        .reset_index(drop=True)
    )
    selected = (
        provenance[
            provenance["selected_for_sentiment"]
        ]
        .sort_values("api_rank")
        .reset_index(drop=True)
    )
    return provenance, selected


def collect_day(
    session: requests.Session,
    run_date: dt.date,
    output_dir: Path,
    query: str = QUERY,
    request_records: int = 150,
    target_headlines: int = 100,
    timeout_seconds: int = 120,
    max_retries: int = 8,
    force: bool = False,
) -> dict:
    label = f"{run_date:%Y%m%d}"
    raw_path = output_dir / "raw" / f"articles_{label}.csv.gz"
    daily_path = output_dir / "daily" / f"news_{label}.csv"
    manifest_path = (
        output_dir / "manifests" / f"manifest_{label}.json"
    )

    if (
        not force
        and raw_path.exists()
        and daily_path.exists()
        and manifest_path.exists()
    ):
        return {
            "date": run_date.isoformat(),
            "status": "skipped",
            "manifest": str(manifest_path),
        }

    raw, elapsed = request_daily_candidates(
        session=session,
        run_date=run_date,
        query=query,
        request_records=request_records,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    provenance, selected = normalize_and_select(
        raw,
        run_date,
        target_headlines,
    )

    titles = selected["title"].tolist()
    daily = pd.DataFrame(
        [
            {
                "date": run_date.isoformat(),
                "article_count": int(len(titles)),
                "headline_hash": headline_hash(titles),
                "raw_text": " || ".join(titles),
            }
        ]
    )

    for directory in [
        raw_path.parent,
        daily_path.parent,
        manifest_path.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    provenance.to_csv(
        raw_path,
        index=False,
        compression="gzip",
    )
    daily.to_csv(daily_path, index=False)

    manifest = {
        "date": run_date.isoformat(),
        "query": query,
        "sort": SORT_MODE,
        "request_records": request_records,
        "target_unique_headlines": target_headlines,
        "api_rows_returned": int(len(raw)),
        "in_day_url_level_rows": int(len(provenance)),
        "unique_headlines_available": int(
            provenance["title_key"].nunique()
            if len(provenance)
            else 0
        ),
        "selected_headlines": int(len(selected)),
        "syndicated_or_repeated_rows_excluded": int(
            len(provenance) - provenance["title_key"].nunique()
            if len(provenance)
            else 0
        ),
        "api_result_limit_reached": bool(
            len(raw) >= request_records
        ),
        "elapsed_seconds": round(float(elapsed), 3),
        "acquired_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "raw_file": str(raw_path),
        "daily_file": str(daily_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {
        "date": run_date.isoformat(),
        "status": "complete",
        **manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, type=parse_date)
    parser.add_argument(
        "--output-dir",
        default="data/sampled_backfill_2023_2025",
    )
    parser.add_argument("--request-records", type=int, default=150)
    parser.add_argument("--target-headlines", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "NewsDrivenAlphaAcademicResearch/3.0 "
                "(historical headline sampling)"
            )
        }
    )
    result = collect_day(
        session=session,
        run_date=args.date,
        output_dir=Path(args.output_dir),
        request_records=args.request_records,
        target_headlines=args.target_headlines,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
