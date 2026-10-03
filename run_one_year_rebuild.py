"""Run the 2023 historical rebuild continuously with weekly checkpoints.

Each week must complete, in order:
1. BigQuery GDELT headline acquisition
2. cumulative news consolidation/validation
3. FinBERT scoring for that week
4. cumulative sentiment consolidation/validation
5. durable weekly checkpoint

Rerunning the script resumes from completed checkpoints and existing daily files.
The canonical public repository data files are never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_START = dt.date(2023, 1, 1)
DEFAULT_END = dt.date(2023, 12, 31)
DEFAULT_PROJECT = "news-alpha-amit-2026"


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def run_command(args: list[str], stage: str) -> None:
    print(f"\n=== {stage} ===", flush=True)
    print(" ".join(args), flush=True)
    completed = subprocess.run(args, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit code "
            f"{completed.returncode}"
        )


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def week_ranges(
    start_date: dt.date,
    end_date: dt.date,
):
    cursor = start_date
    week_number = 1
    while cursor <= end_date:
        week_end = min(
            cursor + dt.timedelta(days=6),
            end_date,
        )
        yield week_number, cursor, week_end
        cursor = week_end + dt.timedelta(days=1)
        week_number += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=parse_date,
        default=DEFAULT_START,
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=DEFAULT_END,
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max-weeks",
        type=int,
        help="Optional test-only cap on processed weeks.",
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be on or after --start")
    if args.start.year != 2023 or args.end.year != 2023:
        raise ValueError(
            "This orchestrator is intentionally restricted "
            "to calendar year 2023."
        )

    root = Path.cwd()
    data_dir = root / "data"
    bigquery_dir = data_dir / "bigquery_backfill_2023_2025"
    finbert_dir = data_dir / "finbert_backfill_2023_2025"
    checkpoint_dir = data_dir / "one_year_2023_checkpoints"
    progress_path = data_dir / "one_year_2023_progress.jsonl"

    news_output = data_dir / "news_daily_bigquery_2023_2025.csv"
    news_summary = (
        data_dir / "bigquery_backfill_2023_2025_summary.json"
    )
    sentiment_output = (
        data_dir / "sentiment_features_bigquery_2023_2025.csv"
    )
    sentiment_summary = (
        data_dir / "finbert_backfill_2023_2025_summary.json"
    )

    python = sys.executable
    weeks = list(week_ranges(args.start, args.end))
    if args.max_weeks is not None:
        weeks = weeks[: args.max_weeks]

    print(
        f"One-year rebuild: {args.start} -> {args.end}; "
        f"weeks={len(weeks)}",
        flush=True,
    )

    run_started = time.time()

    for week_number, week_start, week_end in weeks:
        checkpoint_path = checkpoint_dir / (
            f"week_{week_number:02d}_"
            f"{week_start:%Y%m%d}_{week_end:%Y%m%d}.json"
        )

        if checkpoint_path.exists():
            checkpoint = load_json(checkpoint_path)
            if (
                checkpoint.get("status") == "complete"
                and checkpoint.get("week_start")
                == week_start.isoformat()
                and checkpoint.get("week_end")
                == week_end.isoformat()
            ):
                print(
                    f"\n[{week_number}/{len(weeks)}] "
                    f"SKIP COMPLETE WEEK "
                    f"{week_start} -> {week_end}",
                    flush=True,
                )
                continue

        print(
            f"\n"
            f"############################################\n"
            f"WEEK {week_number}/{len(weeks)}: "
            f"{week_start} -> {week_end}\n"
            f"############################################",
            flush=True,
        )

        week_started = time.time()
        current_stage = "starting"

        try:
            current_stage = "bigquery_acquisition"
            run_command(
                [
                    python,
                    "bigquery_gdelt_backfill.py",
                    "--project",
                    args.project,
                    "--start",
                    week_start.isoformat(),
                    "--end",
                    week_end.isoformat(),
                    "--output-dir",
                    str(bigquery_dir),
                ],
                current_stage,
            )

            current_stage = "news_consolidation"
            run_command(
                [
                    python,
                    "consolidate_bigquery_backfill.py",
                    "--start",
                    args.start.isoformat(),
                    "--end",
                    week_end.isoformat(),
                    "--input-dir",
                    str(bigquery_dir),
                    "--output",
                    str(news_output),
                    "--summary",
                    str(news_summary),
                ],
                current_stage,
            )

            current_stage = "finbert_scoring"
            run_command(
                [
                    python,
                    "finbert_backfill.py",
                    "--input",
                    str(news_output),
                    "--output-dir",
                    str(finbert_dir),
                    "--start",
                    week_start.isoformat(),
                    "--end",
                    week_end.isoformat(),
                    "--batch-size",
                    str(args.batch_size),
                    "--max-length",
                    str(args.max_length),
                ],
                current_stage,
            )

            current_stage = "sentiment_consolidation"
            run_command(
                [
                    python,
                    "consolidate_finbert_backfill.py",
                    "--news",
                    str(news_output),
                    "--input-dir",
                    str(finbert_dir),
                    "--output",
                    str(sentiment_output),
                    "--summary",
                    str(sentiment_summary),
                ],
                current_stage,
            )

            news_stats = load_json(news_summary)
            sentiment_stats = load_json(sentiment_summary)

            expected_days = (
                week_end - args.start
            ).days + 1
            if (
                int(news_stats["calendar_days_requested"])
                != expected_days
            ):
                raise ValueError(
                    "Cumulative news summary requested-day count "
                    "does not match the orchestrator horizon."
                )
            if (
                int(sentiment_stats["overlapping_validated_days"])
                != int(news_stats["calendar_days_with_news"])
            ):
                raise ValueError(
                    "Not all cumulative news days have validated "
                    "FinBERT sentiment outputs."
                )
            if int(sentiment_stats["article_count_mismatches"]) != 0:
                raise ValueError(
                    "Cumulative news/sentiment count mismatch."
                )
            if int(sentiment_stats["headline_hash_mismatches"]) != 0:
                raise ValueError(
                    "Cumulative news/sentiment headline-hash mismatch."
                )

            checkpoint = {
                "status": "complete",
                "week_number": week_number,
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "completed_at_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
                "week_elapsed_seconds": round(
                    time.time() - week_started,
                    3,
                ),
                "cumulative_calendar_days_requested": int(
                    news_stats["calendar_days_requested"]
                ),
                "cumulative_news_days": int(
                    news_stats["calendar_days_with_news"]
                ),
                "cumulative_unique_headlines": int(
                    news_stats["unique_retained_headlines"]
                ),
                "cumulative_sentiment_days": int(
                    sentiment_stats[
                        "overlapping_validated_days"
                    ]
                ),
                "cumulative_headlines_scored": int(
                    sentiment_stats["total_headlines_scored"]
                ),
                "bigquery_bytes_processed": int(
                    news_stats["bigquery_bytes_processed"]
                    or 0
                ),
            }
            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2),
                encoding="utf-8",
            )
            append_jsonl(progress_path, checkpoint)

            print(
                f"WEEK {week_number} COMPLETE: "
                f"cumulative_news_days="
                f"{checkpoint['cumulative_news_days']}, "
                f"cumulative_headlines="
                f"{checkpoint['cumulative_unique_headlines']}",
                flush=True,
            )

        except Exception as exc:
            failure = {
                "status": "failed",
                "week_number": week_number,
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "stage": current_stage,
                "error": str(exc),
                "failed_at_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
            }
            append_jsonl(progress_path, failure)
            print(
                f"WEEK {week_number} FAILED at "
                f"{current_stage}: {exc}",
                flush=True,
            )
            raise

    total_elapsed = time.time() - run_started
    final = {
        "status": "one_year_complete",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "weeks_completed": len(weeks),
        "completed_at_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "elapsed_seconds_this_run": round(
            total_elapsed,
            3,
        ),
    }
    append_jsonl(progress_path, final)
    print(
        f"\n2023 ONE-YEAR REBUILD COMPLETE. "
        f"Stopped at {args.end}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
