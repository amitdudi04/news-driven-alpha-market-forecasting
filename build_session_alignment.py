"""Build timestamp-safe headline-to-CSI300 trading-session alignment.

GDELT GAL's datetime is treated as a GDELT seen timestamp, not asserted to be
the publisher's original publication time.

Forecast information cutoff:
    15:00 Asia/Shanghai on each genuine CSI 300 trading day.

A headline is assigned to the first market close timestamp greater than or
equal to its GDELT seen timestamp. Therefore each session's information window
is exactly:
    (previous trading close, current trading close]

The pipeline works month-by-month with durable compressed checkpoints so the
full 2023-2026 history is never held in memory at once.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALIGNMENT_VERSION = "v1-gdelt-seen-first-close-1500-asia-shanghai"
TZ = "Asia/Shanghai"
CLOSE_HOUR = 15
DEFAULT_START = dt.date(2023, 1, 1)
DEFAULT_END = dt.date(2026, 10, 3)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def month_ranges(start: dt.date, end: dt.date):
    cursor = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end)
    while cursor <= end_ts:
        next_month = cursor + pd.offsets.MonthBegin(1)
        month_start = max(cursor.date(), start)
        month_end = min(
            (next_month - pd.Timedelta(days=1)).date(),
            end,
        )
        yield month_start, month_end
        cursor = next_month


def load_market_schedule(
    market_path: Path,
) -> pd.DataFrame:
    market = pd.read_csv(market_path)
    required = {
        "date",
        "close",
        "return",
        "volatility",
        "momentum",
        "momentum_acceleration",
        "regime_dummy",
        "is_research_sample",
    }
    missing = required - set(market.columns)
    if missing:
        raise ValueError(
            f"Market feature file missing columns: {sorted(missing)}"
        )

    market["date"] = pd.to_datetime(
        market["date"],
        errors="coerce",
    )
    market = (
        market.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if market["date"].duplicated().any():
        raise ValueError("Duplicate market sessions remain.")

    close_local = (
        market["date"]
        .dt.tz_localize(TZ)
        + pd.Timedelta(hours=CLOSE_HOUR)
    )
    market["close_timestamp_shanghai"] = close_local
    market["previous_close_timestamp_shanghai"] = (
        close_local.shift(1)
    )
    return market


def daily_paths(
    article_dir: Path,
    score_dir: Path,
    start: dt.date,
    end: dt.date,
):
    for stamp in pd.date_range(start, end, freq="D"):
        label = stamp.strftime("%Y%m%d")
        article_path = article_dir / f"articles_{label}.csv.gz"
        score_path = score_dir / f"scores_{label}.csv.gz"

        article_exists = article_path.exists()
        score_exists = score_path.exists()

        if article_exists and not score_exists:
            # The BigQuery collector deliberately writes an empty daily
            # checkpoint for genuine zero-news days. FinBERT correctly has
            # no score file because there are no headlines to score.
            empty_articles = pd.read_csv(article_path)
            if empty_articles.empty:
                continue
            raise FileNotFoundError(
                f"Non-empty article file has no score checkpoint for "
                f"{stamp.date()}"
            )

        if score_exists and not article_exists:
            raise FileNotFoundError(
                f"Score checkpoint has no article file for {stamp.date()}"
            )

        if article_exists and score_exists:
            yield stamp.date(), article_path, score_path


def process_day(
    day: dt.date,
    article_path: Path,
    score_path: Path,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    articles = pd.read_csv(article_path)
    scores = pd.read_csv(score_path)

    if len(articles) != len(scores):
        raise ValueError(
            f"{day}: article/score row counts differ: "
            f"{len(articles)} vs {len(scores)}"
        )
    if articles["title"].duplicated().any():
        raise ValueError(
            f"{day}: duplicate article titles remain before alignment."
        )
    if scores["title"].duplicated().any():
        raise ValueError(
            f"{day}: duplicate score titles remain before alignment."
        )

    joined = articles.merge(
        scores[
            [
                "date",
                "title",
                "positive_probability",
                "negative_probability",
                "neutral_probability",
                "predicted_label",
                "sentiment_score",
            ]
        ],
        on=["date", "title"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = int((joined["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"{day}: {unmatched} article/score rows failed exact join."
        )
    joined = joined.drop(columns=["_merge"])

    seen_utc = pd.to_datetime(
        joined["datetime_utc"],
        utc=True,
        errors="coerce",
    )
    if seen_utc.isna().any():
        raise ValueError(
            f"{day}: invalid GDELT seen timestamps detected."
        )
    seen_local = seen_utc.dt.tz_convert(TZ)
    local_date = seen_local.dt.strftime("%Y-%m-%d")
    stored_date = joined["date"].astype(str)
    local_date_mismatch = int(
        (local_date != stored_date).sum()
    )
    if local_date_mismatch:
        raise ValueError(
            f"{day}: {local_date_mismatch} rows disagree with "
            "their Shanghai calendar date."
        )

    close_series = market["close_timestamp_shanghai"]
    close_ns = close_series.array.asi8
    seen_ns = seen_local.array.asi8
    close_index = np.searchsorted(
        close_ns,
        seen_ns,
        side="left",
    )
    assigned = close_index < len(market)

    session_date = np.full(
        len(joined),
        None,
        dtype=object,
    )
    assigned_close = np.full(
        len(joined),
        None,
        dtype=object,
    )
    previous_close = np.full(
        len(joined),
        None,
        dtype=object,
    )

    assigned_positions = np.flatnonzero(assigned)
    if len(assigned_positions):
        idx = close_index[assigned_positions]
        session_date[assigned_positions] = (
            market.iloc[idx]["date"]
            .dt.strftime("%Y-%m-%d")
            .to_numpy()
        )
        assigned_close[assigned_positions] = (
            market.iloc[idx]["close_timestamp_shanghai"]
            .astype(str)
            .to_numpy()
        )

        prior_idx = idx - 1
        if (prior_idx < 0).any():
            raise ValueError(
                "A news observation predates the available warm-up close."
            )
        previous_close[assigned_positions] = (
            market.iloc[prior_idx][
                "close_timestamp_shanghai"
            ]
            .astype(str)
            .to_numpy()
        )

    trading_dates = set(
        market["date"].dt.strftime("%Y-%m-%d")
    )
    relation = np.empty(len(joined), dtype=object)
    for pos in range(len(joined)):
        if not assigned[pos]:
            relation[pos] = "pending_after_last_known_close"
            continue
        if session_date[pos] == local_date.iloc[pos]:
            relation[pos] = "same_day_at_or_before_close"
        elif local_date.iloc[pos] in trading_dates:
            relation[pos] = "after_close_to_next_session"
        else:
            relation[pos] = "nontrading_day_to_next_session"

    result = pd.DataFrame(
        {
            "news_calendar_date": stored_date,
            "gdelt_seen_time_utc": seen_utc.astype(str),
            "gdelt_seen_time_shanghai": seen_local.astype(str),
            "title": joined["title"].astype(str),
            "url": joined["url"].fillna("").astype(str),
            "domain": joined["domain"].fillna("").astype(str),
            "language": joined["language"].fillna("").astype(str),
            "positive_probability": joined[
                "positive_probability"
            ].astype(float),
            "negative_probability": joined[
                "negative_probability"
            ].astype(float),
            "neutral_probability": joined[
                "neutral_probability"
            ].astype(float),
            "predicted_label": joined[
                "predicted_label"
            ].astype(str),
            "sentiment_score": joined[
                "sentiment_score"
            ].astype(float),
            "assigned_session_date": session_date,
            "previous_close_shanghai": previous_close,
            "assigned_close_shanghai": assigned_close,
            "assignment_relation": relation,
        }
    )

    assigned_frame = result[
        result["assigned_session_date"].notna()
    ].copy()
    timing_violations = 0
    if len(assigned_frame):
        assigned_seen = pd.to_datetime(
            assigned_frame["gdelt_seen_time_shanghai"],
            utc=True,
        )
        assigned_close_ts = pd.to_datetime(
            assigned_frame["assigned_close_shanghai"],
            utc=True,
        )
        previous_close_ts = pd.to_datetime(
            assigned_frame["previous_close_shanghai"],
            utc=True,
        )
        timing_violations = int(
            (
                (assigned_seen > assigned_close_ts)
                | (assigned_seen <= previous_close_ts)
            ).sum()
        )
        if timing_violations:
            raise ValueError(
                f"{day}: {timing_violations} causal-window violations."
            )

    relation_counts = (
        result["assignment_relation"]
        .value_counts()
        .to_dict()
    )
    summary = {
        "day": day.isoformat(),
        "input_articles": int(len(articles)),
        "input_scores": int(len(scores)),
        "joined_rows": int(len(result)),
        "unmatched_rows": unmatched,
        "local_date_mismatches": local_date_mismatch,
        "timing_violations": timing_violations,
        "assigned_rows": int(
            result["assigned_session_date"].notna().sum()
        ),
        "pending_rows": int(
            result["assigned_session_date"].isna().sum()
        ),
        "relation_counts": {
            str(key): int(value)
            for key, value in relation_counts.items()
        },
    }
    return result, summary


def process_month(
    month_start: dt.date,
    month_end: dt.date,
    article_dir: Path,
    score_dir: Path,
    market: pd.DataFrame,
    chunk_dir: Path,
    force: bool,
) -> dict:
    label = month_start.strftime("%Y-%m")
    chunk_path = chunk_dir / f"{label}.csv.gz"
    manifest_path = chunk_dir / f"{label}.json"

    if (
        not force
        and chunk_path.exists()
        and manifest_path.exists()
    ):
        existing = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            existing.get("status") == "complete"
            and existing.get("alignment_version")
            == ALIGNMENT_VERSION
            and existing.get("month_start")
            == month_start.isoformat()
            and existing.get("month_end")
            == month_end.isoformat()
        ):
            print(
                f"{label}: skip validated checkpoint "
                f"rows={existing['joined_rows']}",
                flush=True,
            )
            return existing

    frames = []
    day_summaries = []
    for day, article_path, score_path in daily_paths(
        article_dir,
        score_dir,
        month_start,
        month_end,
    ):
        frame, summary = process_day(
            day,
            article_path,
            score_path,
            market,
        )
        frames.append(frame)
        day_summaries.append(summary)

    if frames:
        month = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        month = pd.DataFrame(
            columns=[
                "news_calendar_date",
                "gdelt_seen_time_utc",
                "gdelt_seen_time_shanghai",
                "title",
                "url",
                "domain",
                "language",
                "positive_probability",
                "negative_probability",
                "neutral_probability",
                "predicted_label",
                "sentiment_score",
                "assigned_session_date",
                "previous_close_shanghai",
                "assigned_close_shanghai",
                "assignment_relation",
            ]
        )

    chunk_dir.mkdir(parents=True, exist_ok=True)
    month.to_csv(
        chunk_path,
        index=False,
        compression="gzip",
    )

    relation_counts = (
        month["assignment_relation"]
        .value_counts()
        .to_dict()
        if len(month)
        else {}
    )
    manifest = {
        "status": "complete",
        "alignment_version": ALIGNMENT_VERSION,
        "month_start": month_start.isoformat(),
        "month_end": month_end.isoformat(),
        "news_days_processed": int(len(day_summaries)),
        "joined_rows": int(len(month)),
        "assigned_rows": int(
            month["assigned_session_date"].notna().sum()
        ),
        "pending_rows": int(
            month["assigned_session_date"].isna().sum()
        ),
        "unmatched_rows": int(
            sum(x["unmatched_rows"] for x in day_summaries)
        ),
        "local_date_mismatches": int(
            sum(
                x["local_date_mismatches"]
                for x in day_summaries
            )
        ),
        "timing_violations": int(
            sum(x["timing_violations"] for x in day_summaries)
        ),
        "relation_counts": {
            str(key): int(value)
            for key, value in relation_counts.items()
        },
        "chunk_file": str(chunk_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"{label}: rows={manifest['joined_rows']} "
        f"assigned={manifest['assigned_rows']} "
        f"pending={manifest['pending_rows']}",
        flush=True,
    )
    return manifest


def combine_chunks_and_sessions(
    chunk_dir: Path,
    market: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    aligned_output: Path,
    session_output: Path,
    summary_output: Path,
) -> dict:
    chunk_paths = sorted(chunk_dir.glob("????-??.csv.gz"))
    manifests = []
    for path in sorted(chunk_dir.glob("????-??.json")):
        manifests.append(
            json.loads(path.read_text(encoding="utf-8"))
        )
    if not chunk_paths:
        raise FileNotFoundError(
            "No timestamp-alignment chunks were produced."
        )

    aggregate_parts = []
    total_rows = 0
    pending_rows = 0
    relation_totals: dict[str, int] = {}
    first_header = True

    aligned_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with gzip.open(
        aligned_output,
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        for chunk_path in chunk_paths:
            frame = pd.read_csv(chunk_path)
            frame.to_csv(
                handle,
                index=False,
                header=first_header,
            )
            first_header = False
            total_rows += len(frame)
            pending_rows += int(
                frame["assigned_session_date"].isna().sum()
            )
            for key, value in (
                frame["assignment_relation"]
                .value_counts()
                .to_dict()
                .items()
            ):
                relation_totals[key] = (
                    relation_totals.get(key, 0)
                    + int(value)
                )

            assigned = frame[
                frame["assigned_session_date"].notna()
            ].copy()
            if assigned.empty:
                continue

            assigned["sentiment_sq"] = (
                assigned["sentiment_score"].astype(float) ** 2
            )
            assigned["positive_count"] = (
                assigned["predicted_label"] == "positive"
            ).astype(int)
            assigned["negative_count"] = (
                assigned["predicted_label"] == "negative"
            ).astype(int)
            assigned["neutral_count"] = (
                assigned["predicted_label"] == "neutral"
            ).astype(int)

            grouped = assigned.groupby(
                "assigned_session_date",
                as_index=False,
            ).agg(
                article_count_t=("title", "size"),
                sentiment_sum=("sentiment_score", "sum"),
                sentiment_sq_sum=("sentiment_sq", "sum"),
                positive_count=("positive_count", "sum"),
                negative_count=("negative_count", "sum"),
                neutral_count=("neutral_count", "sum"),
                first_seen_shanghai=(
                    "gdelt_seen_time_shanghai",
                    "min",
                ),
                last_seen_shanghai=(
                    "gdelt_seen_time_shanghai",
                    "max",
                ),
            )
            aggregate_parts.append(grouped)

    partial = pd.concat(
        aggregate_parts,
        ignore_index=True,
    )
    combined = partial.groupby(
        "assigned_session_date",
        as_index=False,
    ).agg(
        article_count_t=("article_count_t", "sum"),
        sentiment_sum=("sentiment_sum", "sum"),
        sentiment_sq_sum=("sentiment_sq_sum", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        neutral_count=("neutral_count", "sum"),
        first_seen_shanghai=("first_seen_shanghai", "min"),
        last_seen_shanghai=("last_seen_shanghai", "max"),
    )

    n = combined["article_count_t"].astype(float)
    combined["sentiment_mean_t"] = (
        combined["sentiment_sum"] / n
    )
    numerator = (
        combined["sentiment_sq_sum"]
        - (combined["sentiment_sum"] ** 2) / n
    ).clip(lower=0.0)
    combined["sentiment_std_t"] = np.where(
        n > 1,
        np.sqrt(numerator / (n - 1)),
        0.0,
    )
    combined["positive_share_t"] = (
        combined["positive_count"] / n
    )
    combined["negative_share_t"] = (
        combined["negative_count"] / n
    )
    combined["neutral_share_t"] = (
        combined["neutral_count"] / n
    )

    research_market = market[
        (market["date"].dt.date >= start)
        & (market["date"].dt.date <= end)
        & (market["is_research_sample"] == 1)
    ].copy()
    research_market["date_str"] = (
        research_market["date"].dt.strftime("%Y-%m-%d")
    )

    sessions = research_market.merge(
        combined,
        left_on="date_str",
        right_on="assigned_session_date",
        how="left",
    )
    sessions["article_count_t"] = (
        sessions["article_count_t"]
        .fillna(0)
        .astype(int)
    )
    sessions["has_news_t"] = (
        sessions["article_count_t"] > 0
    ).astype(int)

    sessions["window_start_shanghai"] = (
        sessions[
            "previous_close_timestamp_shanghai"
        ].astype(str)
    )
    sessions["window_end_shanghai"] = (
        sessions["close_timestamp_shanghai"].astype(str)
    )

    keep = [
        "date_str",
        "window_start_shanghai",
        "window_end_shanghai",
        "close",
        "return",
        "volatility",
        "momentum",
        "momentum_acceleration",
        "regime_dummy",
        "has_news_t",
        "article_count_t",
        "sentiment_mean_t",
        "sentiment_std_t",
        "positive_share_t",
        "negative_share_t",
        "neutral_share_t",
        "first_seen_shanghai",
        "last_seen_shanghai",
    ]
    sessions = sessions[keep].rename(
        columns={"date_str": "date"}
    )
    sessions.to_csv(
        session_output,
        index=False,
    )

    assigned_rows = total_rows - pending_rows

    # Partial validation windows can contain late headlines whose first
    # eligible trading close falls just beyond the requested end date.
    assigned_in_requested_sessions = 0
    assigned_beyond_requested_sessions = 0
    for chunk_path in chunk_paths:
        scope = pd.read_csv(
            chunk_path,
            usecols=["assigned_session_date"],
        )
        assigned_dates = pd.to_datetime(
            scope["assigned_session_date"],
            errors="coerce",
        )
        is_assigned = assigned_dates.notna()
        in_scope = (
            is_assigned
            & (assigned_dates >= pd.Timestamp(start))
            & (assigned_dates <= pd.Timestamp(end))
        )
        assigned_in_requested_sessions += int(in_scope.sum())
        assigned_beyond_requested_sessions += int(
            (is_assigned & ~in_scope).sum()
        )

    session_count_sum = int(
        sessions["article_count_t"].sum()
    )
    if session_count_sum != assigned_in_requested_sessions:
        raise ValueError(
            "Session aggregate count does not equal in-scope assigned "
            "headline count: "
            f"{session_count_sum} vs "
            f"{assigned_in_requested_sessions}"
        )

    manifest_joined = int(
        sum(x.get("joined_rows", 0) for x in manifests)
    )
    manifest_unmatched = int(
        sum(x.get("unmatched_rows", 0) for x in manifests)
    )
    manifest_date_mismatch = int(
        sum(
            x.get("local_date_mismatches", 0)
            for x in manifests
        )
    )
    manifest_timing_violations = int(
        sum(
            x.get("timing_violations", 0)
            for x in manifests
        )
    )
    if manifest_joined != total_rows:
        raise ValueError(
            "Manifest/final aligned-row count mismatch."
        )
    if (
        manifest_unmatched
        or manifest_date_mismatch
        or manifest_timing_violations
    ):
        raise ValueError(
            "Alignment validation errors remain in monthly manifests."
        )

    pending_frame_parts = []
    for chunk_path in chunk_paths:
        frame = pd.read_csv(
            chunk_path,
            usecols=[
                "gdelt_seen_time_shanghai",
                "assigned_session_date",
            ],
        )
        pending = frame[
            frame["assigned_session_date"].isna()
        ]
        if len(pending):
            pending_frame_parts.append(pending)
    if pending_frame_parts:
        pending_all = pd.concat(
            pending_frame_parts,
            ignore_index=True,
        )
        pending_first = str(
            pending_all["gdelt_seen_time_shanghai"].min()
        )
        pending_last = str(
            pending_all["gdelt_seen_time_shanghai"].max()
        )
    else:
        pending_first = None
        pending_last = None

    summary = {
        "alignment_version": ALIGNMENT_VERSION,
        "news_start": start.isoformat(),
        "news_end": end.isoformat(),
        "prediction_cutoff": "15:00 Asia/Shanghai",
        "information_window": (
            "(previous CSI 300 trading close, "
            "current CSI 300 trading close]"
        ),
        "timestamp_semantics": (
            "GDELT GAL seen timestamp; not guaranteed to equal "
            "publisher-original publication time"
        ),
        "input_headlines_joined": int(total_rows),
        "assigned_headlines_total": int(assigned_rows),
        "assigned_to_sessions_in_requested_period": int(
            assigned_in_requested_sessions
        ),
        "assigned_beyond_requested_period": int(
            assigned_beyond_requested_sessions
        ),
        "pending_after_last_known_close": int(pending_rows),
        "assignment_relation_counts": {
            str(key): int(value)
            for key, value in relation_totals.items()
        },
        "monthly_chunks": int(len(chunk_paths)),
        "monthly_unmatched_rows": manifest_unmatched,
        "monthly_local_date_mismatches": manifest_date_mismatch,
        "monthly_timing_violations": manifest_timing_violations,
        "market_sessions_in_research_period": int(
            len(research_market)
        ),
        "market_sessions_with_news": int(
            sessions["has_news_t"].sum()
        ),
        "market_sessions_without_news": int(
            (sessions["has_news_t"] == 0).sum()
        ),
        "session_article_count_sum": session_count_sum,
        "first_research_market_session": (
            sessions["date"].min()
        ),
        "last_known_market_session": (
            sessions["date"].max()
        ),
        "last_known_market_close_shanghai": str(
            research_market[
                "close_timestamp_shanghai"
            ].max()
        ),
        "pending_first_seen_shanghai": pending_first,
        "pending_last_seen_shanghai": pending_last,
        "article_level_output": str(aligned_output),
        "session_level_output": str(session_output),
    }
    summary_output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


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
        "--market",
        default="data/csi300_market_features_2022_2026.csv",
    )
    parser.add_argument(
        "--article-dir",
        default="data/gdelt_headlines/daily",
    )
    parser.add_argument(
        "--score-dir",
        default="data/finbert_scores/headline_scores",
    )
    parser.add_argument(
        "--chunk-dir",
        default="data/timestamp_alignment_chunks",
    )
    parser.add_argument(
        "--aligned-output",
        default="data/timestamp_aligned_headlines_2023_2026.csv.gz",
    )
    parser.add_argument(
        "--session-output",
        default="data/session_sentiment_timestamp_safe_2023_2026.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="data/timestamp_alignment_2023_2026_summary.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be on or after --start")

    market = load_market_schedule(Path(args.market))
    article_dir = Path(args.article_dir)
    score_dir = Path(args.score_dir)
    chunk_dir = Path(args.chunk_dir)

    for month_start, month_end in month_ranges(
        args.start,
        args.end,
    ):
        process_month(
            month_start,
            month_end,
            article_dir,
            score_dir,
            market,
            chunk_dir,
            args.force,
        )

    combine_chunks_and_sessions(
        chunk_dir=chunk_dir,
        market=market,
        start=args.start,
        end=args.end,
        aligned_output=Path(args.aligned_output),
        session_output=Path(args.session_output),
        summary_output=Path(args.summary_output),
    )


if __name__ == "__main__":
    main()
