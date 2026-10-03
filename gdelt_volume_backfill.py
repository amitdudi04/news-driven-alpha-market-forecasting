"""Collect normalized GDELT news-volume series for the historical sample."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

from sampled_news_backfill import (
    GDELT_URL,
    QUERY,
    parse_date,
    shanghai_day_utc_bounds,
)

MIN_REQUEST_INTERVAL_SECONDS = 6.0


def _wait_for_rate_limit(last_request_at: float | None) -> float:
    if last_request_at is not None:
        elapsed = time.monotonic() - last_request_at
        remaining = MIN_REQUEST_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)
    return time.monotonic()


def request_timeline(
    session: requests.Session,
    start_date: dt.date,
    end_date: dt.date,
    last_request_at: float | None,
    timeout_seconds: int = 180,
    max_retries: int = 8,
) -> tuple[dict, float, float]:
    start, _ = shanghai_day_utc_bounds(start_date)
    _, end = shanghai_day_utc_bounds(end_date)
    params = {
        "query": QUERY,
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        last_request_at = _wait_for_rate_limit(last_request_at)
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
                wait_seconds = (
                    max(int(retry_after), 30)
                    if retry_after.isdigit()
                    else min(30 * (2**attempt), 600)
                )
                print(
                    f"429 for {start_date} -> {end_date}; "
                    f"waiting {wait_seconds}s",
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response.json(), elapsed, last_request_at
        except Exception as exc:
            last_error = exc
            wait_seconds = min(10 * (2**attempt), 180)
            print(
                f"Timeline request error: {exc}; "
                f"waiting {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"TimelineVolRaw failed for {start_date} -> {end_date}: "
        f"{last_error}"
    )


def parse_timeline(
    payload: dict,
    requested_start: dt.date,
    requested_end: dt.date,
) -> pd.DataFrame:
    timeline = payload.get("timeline", [])
    if not timeline:
        return pd.DataFrame(
            columns=[
                "date",
                "news_volume_raw",
                "news_volume_norm",
                "news_volume_share",
            ]
        )

    data = timeline[0].get("data", [])
    rows = []
    for entry in data:
        if "date" not in entry:
            continue
        timestamp = pd.to_datetime(
            entry["date"],
            utc=True,
            errors="coerce",
        )
        if pd.isna(timestamp):
            continue
        research_date = timestamp.tz_convert(
            "Asia/Shanghai"
        ).date()
        if not (requested_start <= research_date <= requested_end):
            continue

        raw_value = pd.to_numeric(
            entry.get("value"),
            errors="coerce",
        )
        norm_value = pd.to_numeric(
            entry.get("norm"),
            errors="coerce",
        )
        if pd.isna(raw_value) or pd.isna(norm_value):
            continue

        rows.append(
            {
                "date": research_date.isoformat(),
                "news_volume_raw": float(raw_value),
                "news_volume_norm": float(norm_value),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "news_volume_raw",
                "news_volume_norm",
                "news_volume_share",
            ]
        )

    frame = pd.DataFrame(rows)
    # If GDELT returns sub-daily resolution, aggregate numerator and
    # denominator before computing the daily normalized share.
    frame = (
        frame.groupby("date", as_index=False)
        .agg(
            news_volume_raw=("news_volume_raw", "sum"),
            news_volume_norm=("news_volume_norm", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    frame["news_volume_share"] = (
        frame["news_volume_raw"]
        / frame["news_volume_norm"].replace(0, pd.NA)
    )
    return frame


def year_chunks(
    start_date: dt.date,
    end_date: dt.date,
):
    year = start_date.year
    while year <= end_date.year:
        chunk_start = max(start_date, dt.date(year, 1, 1))
        chunk_end = min(end_date, dt.date(year, 12, 31))
        yield chunk_start, chunk_end
        year += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=parse_date,
        default=dt.date(2023, 1, 1),
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=dt.date(2025, 12, 31),
    )
    parser.add_argument(
        "--output",
        default="data/gdelt_volume_2023_2025.csv",
    )
    parser.add_argument(
        "--manifest",
        default="data/gdelt_volume_2023_2025_manifest.json",
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be on or after --start")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "NewsDrivenAlphaAcademicResearch/3.0 "
                "(historical volume retrieval)"
            )
        }
    )

    frames = []
    manifest = []
    last_request_at: float | None = None

    for chunk_start, chunk_end in year_chunks(
        args.start,
        args.end,
    ):
        print(
            f"TimelineVolRaw {chunk_start} -> {chunk_end}",
            flush=True,
        )
        payload, elapsed, last_request_at = request_timeline(
            session,
            chunk_start,
            chunk_end,
            last_request_at,
        )
        frame = parse_timeline(
            payload,
            chunk_start,
            chunk_end,
        )
        frames.append(frame)

        source_points = (
            len(payload.get("timeline", [{}])[0].get("data", []))
            if payload.get("timeline")
            else 0
        )
        manifest.append(
            {
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
                "source_points": source_points,
                "daily_rows_retained": int(len(frame)),
                "elapsed_seconds": round(float(elapsed), 3),
                "first_date": (
                    frame["date"].min() if len(frame) else None
                ),
                "last_date": (
                    frame["date"].max() if len(frame) else None
                ),
            }
        )

    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    if len(result):
        result = (
            result.groupby("date", as_index=False)
            .agg(
                news_volume_raw=("news_volume_raw", "sum"),
                news_volume_norm=("news_volume_norm", "sum"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        result["news_volume_share"] = (
            result["news_volume_raw"]
            / result["news_volume_norm"].replace(0, pd.NA)
        )

    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved {len(result)} normalized volume rows to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
