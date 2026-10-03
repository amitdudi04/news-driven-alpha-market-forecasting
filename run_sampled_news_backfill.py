"""Run the historical relevance-ranked headline sample with daily checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import requests

from sampled_news_backfill import (
    QUERY,
    collect_day,
    parse_date,
)


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
        "--output-dir",
        default="data/sampled_backfill_2023_2025",
    )
    parser.add_argument("--request-records", type=int, default=150)
    parser.add_argument("--target-headlines", type=int, default=100)
    parser.add_argument(
        "--inter-day-sleep",
        type=float,
        default=6.5,
        help="Keep above GDELT's published one-request-per-5-seconds limit.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be on or after --start")

    dates = [
        args.start + dt.timedelta(days=offset)
        for offset in range((args.end - args.start).days + 1)
    ]
    if args.max_days is not None:
        dates = dates[: args.max_days]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "NewsDrivenAlphaAcademicResearch/3.0 "
                "(historical headline sampling)"
            )
        }
    )

    output_dir = Path(args.output_dir)
    run_log = output_dir / "run_progress.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Collecting {len(dates)} Shanghai calendar days "
        f"from {dates[0]} through {dates[-1]}.",
        flush=True,
    )

    for index, run_date in enumerate(dates, start=1):
        try:
            result = collect_day(
                session=session,
                run_date=run_date,
                output_dir=output_dir,
                query=QUERY,
                request_records=args.request_records,
                target_headlines=args.target_headlines,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                force=args.force,
            )
        except Exception as exc:
            failure = {
                "date": run_date.isoformat(),
                "status": "failed",
                "error": str(exc),
                "logged_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
            }
            with run_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure) + "\n")
            print(
                f"[{index}/{len(dates)}] {run_date} FAILED: {exc}",
                flush=True,
            )
            raise

        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")

        selected = result.get("selected_headlines", "checkpoint")
        elapsed = result.get("elapsed_seconds", 0)
        print(
            f"[{index}/{len(dates)}] {run_date} "
            f"{result['status']} selected={selected} "
            f"api_seconds={elapsed}",
            flush=True,
        )
        if result["status"] == "complete":
            time.sleep(args.inter_day_sleep)

    print("Requested headline-sampling period is complete.", flush=True)


if __name__ == "__main__":
    main()
