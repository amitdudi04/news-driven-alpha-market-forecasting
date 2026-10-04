"""Historical China-finance headline acquisition through GDELT BigQuery.

The research sample is drawn from the GDELT Article List (GAL). Inclusion is
based directly on English headline text: a China identifier and at least one
macro/finance term must both be present. Exact repeated/syndicated headline
text is removed within each Asia/Shanghai calendar day.

Each research day is queried and checkpointed separately so the historical
backfill is resumable and query-cost provenance is retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

try:
    from google.cloud import bigquery
except ImportError as exc:
    raise SystemExit(
        "google-cloud-bigquery is required for this optional historical "
        "backfill. Install requirements-backfill.txt first."
    ) from exc


QUERY = r"""
SELECT DISTINCT
  g.date AS datetime_utc,
  g.url,
  g.domain,
  g.title,
  g.lang AS language
FROM `gdelt-bq.gdeltv2.gal` AS g
WHERE g.date >= TIMESTAMP(@run_date, 'Asia/Shanghai')
  AND g.date < TIMESTAMP(
    DATE_ADD(@run_date, INTERVAL 1 DAY),
    'Asia/Shanghai'
  )
  AND g.lang = 'en'
  AND g.title IS NOT NULL
  AND LENGTH(TRIM(g.title)) > 15
  AND REGEXP_CONTAINS(
    LOWER(g.title),
    r'(^|[^a-z])(china|chinese|pboc|beijing)([^a-z]|$)'
  )
  AND REGEXP_CONTAINS(
    LOWER(g.title),
    r'(^|[^a-z])(economy|economic|growth|gdp|inflation|deflation|trade|exports?|imports?|markets?|stocks?|shares|equity|equities|yuan|renminbi|interest rates?|rates|liquidity|property|real estate|banks?|banking|regulation|regulatory|stimulus|debt|investment|investors?|financial|finance)([^a-z]|$)'
  )
ORDER BY datetime_utc, url
"""


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def query_config(run_date: dt.date, dry_run: bool = False):
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "run_date",
                "DATE",
                run_date,
            )
        ],
        dry_run=dry_run,
        use_query_cache=False if dry_run else True,
    )


def normalize_rows(rows) -> pd.DataFrame:
    records = [dict(row.items()) for row in rows]
    if not records:
        return pd.DataFrame(
            columns=[
                "date",
                "datetime_utc",
                "title",
                "url",
                "domain",
                "language",
            ]
        )

    df = pd.DataFrame(records)
    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        utc=True,
        errors="coerce",
    )
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["domain"] = df["domain"].fillna("").astype(str).str.strip()
    df["language"] = df["language"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["datetime_utc"])
    df = df[df["title"].str.len() > 15].copy()

    url_key = df["url"].str.lower()
    fallback_key = (
        df["datetime_utc"].astype(str)
        + "|"
        + df["title"].str.lower()
    )
    df["dedup_key"] = url_key.where(
        url_key.str.len() > 0,
        fallback_key,
    )
    df = df.drop_duplicates(subset=["dedup_key"]).copy()

    shanghai_time = df["datetime_utc"].dt.tz_convert("Asia/Shanghai")
    df["date"] = shanghai_time.dt.strftime("%Y-%m-%d")
    df["title_key"] = (
        df["title"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.drop_duplicates(
        subset=["date", "title_key"],
        keep="first",
    ).copy()

    return (
        df[
            [
                "date",
                "datetime_utc",
                "title",
                "url",
                "domain",
                "language",
            ]
        ]
        .sort_values(["datetime_utc", "title"])
        .reset_index(drop=True)
    )


def append_manifest(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
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
        default="data/gdelt_headlines",
    )
    parser.add_argument(
        "--dry-run-day",
        type=parse_date,
        help="Estimate bytes processed for one date and exit.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        help="Optional safety limit for a small test run.",
    )
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)

    if args.dry_run_day:
        job = client.query(
            QUERY,
            job_config=query_config(
                args.dry_run_day,
                dry_run=True,
            ),
        )
        gib = job.total_bytes_processed / (1024**3)
        print(
            f"Dry run {args.dry_run_day}: "
            f"{job.total_bytes_processed:,} bytes "
            f"({gib:.3f} GiB)"
        )
        return

    if args.end < args.start:
        raise ValueError("--end must be on or after --start")

    output_dir = Path(args.output_dir)
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "query_manifest.csv"

    dates = pd.date_range(args.start, args.end, freq="D")
    if args.max_days is not None:
        dates = dates[: args.max_days]

    for index, timestamp in enumerate(dates, start=1):
        run_date = timestamp.date()
        output_path = daily_dir / f"articles_{run_date:%Y%m%d}.csv.gz"

        if output_path.exists():
            print(
                f"[{index}/{len(dates)}] SKIP {run_date}: "
                "checkpoint exists"
            )
            continue

        print(f"[{index}/{len(dates)}] QUERY {run_date}")
        job = client.query(
            QUERY,
            job_config=query_config(run_date),
        )
        rows = list(job.result())
        frame = normalize_rows(rows)

        frame.to_csv(
            output_path,
            index=False,
            compression="gzip",
        )
        append_manifest(
            manifest_path,
            {
                "run_date": run_date.isoformat(),
                "job_id": job.job_id,
                "rows_returned": int(len(rows)),
                "rows_retained": int(len(frame)),
                "syndicated_or_repeated_rows_removed": int(
                    len(rows) - len(frame)
                ),
                "total_bytes_processed": int(
                    job.total_bytes_processed or 0
                ),
                "total_bytes_billed": int(
                    job.total_bytes_billed or 0
                ),
                "output_file": str(output_path),
                "completed_at_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
            },
        )

        gib = (job.total_bytes_processed or 0) / (1024**3)
        print(
            f"    retained={len(frame):,}; "
            f"processed={gib:.3f} GiB"
        )

    print("Requested BigQuery backfill dates are complete.")


if __name__ == "__main__":
    main()
