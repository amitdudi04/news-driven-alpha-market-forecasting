"""Build the causal CSI 300 master session dataset for modeling.

Inputs
------
1. Timestamp-safe article-level FinBERT alignment.
2. Timestamp-safe CSI 300 session market file.

Methodology
-----------
- Re-deduplicate exact normalized titles within each assigned trading-session
  information window so repeated weekend/holiday syndication cannot receive
  extra weight.
- Preserve genuinely missing sentiment as missing.
- Use strict 5/10/20 trading-session sentiment windows: a rolling feature is
  missing whenever any session inside its required window has missing
  sentiment.
- Define news intensity as current unique-headline count divided by the mean
  unique-headline count over the PRIOR 20 trading sessions.
- Construct next-session return/direction targets strictly by chronological
  trading-session shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROLLING_WINDOWS = (5, 10, 20)

PREDICTOR_COLUMNS = [
    "close",
    "return_t",
    "volatility_20_t",
    "momentum_5_20_t",
    "momentum_acceleration_t",
    "regime_dummy_t",
    "unique_headline_count_t",
    "headline_observation_count_t",
    "sentiment_mean_t",
    "sentiment_std_t",
    "positive_share_t",
    "negative_share_t",
    "neutral_share_t",
    "sentiment_roll_5",
    "sentiment_roll_10",
    "sentiment_roll_20",
    "prior_20_session_mean_unique_headlines",
    "news_intensity_20",
    "sentiment_x_volatility",
    "sentiment_roll_5_x_volatility",
    "sentiment_roll_20_x_volatility",
]

STRICT_MODEL_FEATURES = [
    "volatility_20_t",
    "momentum_5_20_t",
    "momentum_acceleration_t",
    "regime_dummy_t",
    "unique_headline_count_t",
    "sentiment_mean_t",
    "sentiment_std_t",
    "sentiment_roll_5",
    "sentiment_roll_10",
    "sentiment_roll_20",
    "news_intensity_20",
    "sentiment_x_volatility",
    "sentiment_roll_5_x_volatility",
    "sentiment_roll_20_x_volatility",
]


def normalize_title(value: object) -> str:
    return " ".join(str(value).casefold().split())


def load_session_market(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "date",
        "window_start_shanghai",
        "window_end_shanghai",
        "close",
        "return",
        "volatility",
        "momentum",
        "momentum_acceleration",
        "regime_dummy",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Session market input missing columns: {sorted(missing)}"
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate trading-session dates detected.")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Trading sessions are not chronological.")
    return frame


def aggregate_unique_headlines(
    aligned_path: Path,
) -> tuple[pd.DataFrame, dict]:
    usecols = [
        "assigned_session_date",
        "gdelt_seen_time_shanghai",
        "title",
        "sentiment_score",
        "predicted_label",
    ]
    aligned = pd.read_csv(
        aligned_path,
        usecols=usecols,
    )
    assigned = aligned[
        aligned["assigned_session_date"].notna()
    ].copy()
    observation_rows = int(len(assigned))

    assigned["assigned_session_date"] = pd.to_datetime(
        assigned["assigned_session_date"],
        errors="raise",
    )
    assigned["gdelt_seen_time_shanghai"] = pd.to_datetime(
        assigned["gdelt_seen_time_shanghai"],
        utc=True,
        errors="raise",
    )
    assigned["title_key"] = assigned["title"].map(
        normalize_title
    )

    if (assigned["title_key"].str.len() == 0).any():
        raise ValueError("Blank normalized title encountered.")

    observation_counts = (
        assigned.groupby(
            "assigned_session_date",
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "headline_observation_count_t"}
        )
    )

    # Deterministic session-level de-duplication: keep the earliest GDELT seen
    # observation for each normalized title within the same trading window.
    unique = (
        assigned.sort_values(
            [
                "assigned_session_date",
                "gdelt_seen_time_shanghai",
                "title_key",
            ]
        )
        .drop_duplicates(
            ["assigned_session_date", "title_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    unique_rows = int(len(unique))
    removed = observation_rows - unique_rows

    unique["positive_count"] = (
        unique["predicted_label"] == "positive"
    ).astype(int)
    unique["negative_count"] = (
        unique["predicted_label"] == "negative"
    ).astype(int)
    unique["neutral_count"] = (
        unique["predicted_label"] == "neutral"
    ).astype(int)

    grouped = (
        unique.groupby(
            "assigned_session_date",
            as_index=False,
        )
        .agg(
            unique_headline_count_t=("title_key", "size"),
            sentiment_mean_t=("sentiment_score", "mean"),
            sentiment_std_raw=(
                "sentiment_score",
                lambda x: x.std(ddof=1),
            ),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
            neutral_count=("neutral_count", "sum"),
        )
    )
    grouped["sentiment_std_t"] = (
        grouped["sentiment_std_raw"].fillna(0.0)
    )
    grouped = grouped.drop(columns=["sentiment_std_raw"])

    n = grouped["unique_headline_count_t"].astype(float)
    grouped["positive_share_t"] = (
        grouped["positive_count"] / n
    )
    grouped["negative_share_t"] = (
        grouped["negative_count"] / n
    )
    grouped["neutral_share_t"] = (
        grouped["neutral_count"] / n
    )
    grouped = grouped.drop(
        columns=[
            "positive_count",
            "negative_count",
            "neutral_count",
        ]
    )

    grouped = grouped.merge(
        observation_counts,
        on="assigned_session_date",
        how="left",
        validate="one_to_one",
    )

    audit = {
        "assigned_headline_observations": observation_rows,
        "session_unique_normalized_headlines": unique_rows,
        "within_session_repeated_headlines_removed": removed,
        "within_session_repeat_fraction": (
            float(removed / observation_rows)
            if observation_rows
            else 0.0
        ),
        "sessions_with_assigned_news": int(
            grouped["assigned_session_date"].nunique()
        ),
    }
    return grouped, audit


def add_predictors(base: pd.DataFrame) -> pd.DataFrame:
    data = base.copy().sort_values("date").reset_index(drop=True)

    for window in ROLLING_WINDOWS:
        data[f"sentiment_roll_{window}"] = (
            data["sentiment_mean_t"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

    # A genuinely causal baseline: only PRIOR trading sessions determine the
    # reference news volume. The current session count is the numerator.
    data["prior_20_session_mean_unique_headlines"] = (
        data["unique_headline_count_t"]
        .shift(1)
        .rolling(window=20, min_periods=20)
        .mean()
    )
    data["news_intensity_20"] = (
        data["unique_headline_count_t"]
        / data["prior_20_session_mean_unique_headlines"]
    )

    data["sentiment_x_volatility"] = (
        data["sentiment_mean_t"]
        * data["volatility_20_t"]
    )
    data["sentiment_roll_5_x_volatility"] = (
        data["sentiment_roll_5"]
        * data["volatility_20_t"]
    )
    data["sentiment_roll_20_x_volatility"] = (
        data["sentiment_roll_20"]
        * data["volatility_20_t"]
    )
    return data


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["target_session_date"] = out["date"].shift(-1)
    out["target_return_t_plus_1"] = out["return_t"].shift(-1)

    target = out["target_return_t_plus_1"]
    out["target_direction_t_plus_1"] = np.where(
        target.notna(),
        (target > 0).astype(float),
        np.nan,
    )
    out["target_available"] = target.notna().astype(int)
    return out


def build_master(
    session_market: pd.DataFrame,
    session_news: pd.DataFrame,
) -> pd.DataFrame:
    market = session_market[
        [
            "date",
            "window_start_shanghai",
            "window_end_shanghai",
            "close",
            "return",
            "volatility",
            "momentum",
            "momentum_acceleration",
            "regime_dummy",
        ]
    ].rename(
        columns={
            "return": "return_t",
            "volatility": "volatility_20_t",
            "momentum": "momentum_5_20_t",
            "momentum_acceleration": (
                "momentum_acceleration_t"
            ),
            "regime_dummy": "regime_dummy_t",
        }
    )

    data = market.merge(
        session_news,
        left_on="date",
        right_on="assigned_session_date",
        how="left",
        validate="one_to_one",
    ).drop(columns=["assigned_session_date"])

    count_cols = [
        "unique_headline_count_t",
        "headline_observation_count_t",
    ]
    for col in count_cols:
        data[col] = data[col].fillna(0).astype(int)

    # Counts of zero are factual. Sentiment itself is intentionally NOT filled.
    data["has_news_t"] = (
        data["unique_headline_count_t"] > 0
    ).astype(int)

    data = add_predictors(data)
    data = add_targets(data)

    strict_complete = data[
        STRICT_MODEL_FEATURES
        + ["target_return_t_plus_1"]
    ].notna().all(axis=1)
    data["strict_model_ready"] = strict_complete.astype(int)
    return data


def validate_master(
    master: pd.DataFrame,
    session_market: pd.DataFrame,
    headline_audit: dict,
) -> dict:
    if len(master) != len(session_market):
        raise ValueError(
            "Master/session-market row count mismatch."
        )
    if master["date"].duplicated().any():
        raise ValueError("Duplicate master session dates detected.")
    if not master["date"].is_monotonic_increasing:
        raise ValueError("Master sessions are not chronological.")

    no_news = master["unique_headline_count_t"].eq(0)
    no_news_dates = (
        master.loc[no_news, "date"]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    sentiment_cols = [
        "sentiment_mean_t",
        "sentiment_std_t",
        "positive_share_t",
        "negative_share_t",
        "neutral_share_t",
    ]
    if not master.loc[no_news, sentiment_cols].isna().all().all():
        raise ValueError(
            "Missing-news sessions were imputed with sentiment values."
        )
    if master.loc[~no_news, sentiment_cols].isna().any().any():
        raise ValueError(
            "A session with headlines has missing pooled sentiment."
        )

    # FinBERT label shares should sum to one whenever news exists.
    share_sum = master[
        ["positive_share_t", "negative_share_t", "neutral_share_t"]
    ].sum(axis=1, min_count=3)
    share_error = float(
        (share_sum.loc[~no_news] - 1.0).abs().max()
    )
    if share_error > 1e-12:
        raise ValueError(
            f"FinBERT class shares fail to sum to one: {share_error}"
        )

    # Strict rolling windows: compare against an independent recomputation.
    rolling_errors = {}
    for window in ROLLING_WINDOWS:
        expected = (
            master["sentiment_mean_t"]
            .rolling(window=window, min_periods=window)
            .mean()
        )
        actual = master[f"sentiment_roll_{window}"]
        valid = expected.notna() & actual.notna()
        error = (
            float((expected[valid] - actual[valid]).abs().max())
            if valid.any()
            else 0.0
        )
        missing_disagreement = int(
            (expected.isna() != actual.isna()).sum()
        )
        if error > 1e-12 or missing_disagreement:
            raise ValueError(
                f"Rolling sentiment validation failed for {window}."
            )
        rolling_errors[str(window)] = {
            "max_abs_error": error,
            "missing_disagreement": missing_disagreement,
        }

    expected_prior_count = (
        master["unique_headline_count_t"]
        .shift(1)
        .rolling(window=20, min_periods=20)
        .mean()
    )
    valid_prior = (
        expected_prior_count.notna()
        & master[
            "prior_20_session_mean_unique_headlines"
        ].notna()
    )
    prior_error = float(
        (
            expected_prior_count[valid_prior]
            - master.loc[
                valid_prior,
                "prior_20_session_mean_unique_headlines",
            ]
        ).abs().max()
    )
    if prior_error > 1e-12:
        raise ValueError("Prior news-volume baseline mismatch.")

    # Target audit.
    expected_target_return = master["return_t"].shift(-1)
    valid_target = expected_target_return.notna()
    target_error = float(
        (
            expected_target_return[valid_target]
            - master.loc[
                valid_target,
                "target_return_t_plus_1",
            ]
        ).abs().max()
    )
    if target_error > 1e-12:
        raise ValueError("Next-session return target mismatch.")

    expected_target_date = master["date"].shift(-1)
    target_date_mismatch = int(
        (
            expected_target_date.fillna(pd.Timestamp("1900-01-01"))
            != master["target_session_date"].fillna(
                pd.Timestamp("1900-01-01")
            )
        ).sum()
    )
    if target_date_mismatch:
        raise ValueError("Next-session date target mismatch.")

    expected_direction = np.where(
        expected_target_return.notna(),
        (expected_target_return > 0).astype(float),
        np.nan,
    )
    direction_actual = master[
        "target_direction_t_plus_1"
    ].to_numpy()
    direction_mismatch = int(
        np.sum(
            ~np.isclose(
                np.nan_to_num(
                    expected_direction,
                    nan=-9.0,
                ),
                np.nan_to_num(
                    direction_actual,
                    nan=-9.0,
                ),
            )
        )
    )
    if direction_mismatch:
        raise ValueError(
            "Next-session direction target mismatch."
        )

    if master.iloc[-1]["target_available"] != 0:
        raise ValueError(
            "Last known session should not have a future target."
        )
    if int(master["target_available"].sum()) != len(master) - 1:
        raise ValueError(
            "Exactly one final target should be unavailable."
        )

    # Truncation invariance: predictor values at a cutoff must not change if
    # all future rows are removed. This is a direct look-ahead/leakage test.
    cutoff_indices = [
        min(50, len(master) - 1),
        min(250, len(master) - 1),
        min(500, len(master) - 1),
        min(750, len(master) - 1),
        len(master) - 1,
    ]
    cutoff_indices = sorted(set(cutoff_indices))
    leakage_max_error = 0.0
    leakage_missing_mismatches = 0

    base_cols = [
        "date",
        "window_start_shanghai",
        "window_end_shanghai",
        "close",
        "return_t",
        "volatility_20_t",
        "momentum_5_20_t",
        "momentum_acceleration_t",
        "regime_dummy_t",
        "unique_headline_count_t",
        "headline_observation_count_t",
        "sentiment_mean_t",
        "sentiment_std_t",
        "positive_share_t",
        "negative_share_t",
        "neutral_share_t",
        "has_news_t",
    ]
    causal_feature_cols = [
        "sentiment_roll_5",
        "sentiment_roll_10",
        "sentiment_roll_20",
        "prior_20_session_mean_unique_headlines",
        "news_intensity_20",
        "sentiment_x_volatility",
        "sentiment_roll_5_x_volatility",
        "sentiment_roll_20_x_volatility",
    ]

    for idx in cutoff_indices:
        prefix = add_predictors(
            master.loc[:idx, base_cols].copy()
        )
        full_row = master.loc[idx, causal_feature_cols]
        prefix_row = prefix.loc[idx, causal_feature_cols]

        for col in causal_feature_cols:
            a = full_row[col]
            b = prefix_row[col]
            if pd.isna(a) or pd.isna(b):
                if not (pd.isna(a) and pd.isna(b)):
                    leakage_missing_mismatches += 1
                continue
            leakage_max_error = max(
                leakage_max_error,
                abs(float(a) - float(b)),
            )

    if (
        leakage_max_error > 1e-12
        or leakage_missing_mismatches
    ):
        raise ValueError(
            "Predictor truncation-invariance leakage check failed."
        )

    yearly = (
        master.assign(year=master["date"].dt.year)
        .groupby("year")
        .agg(
            sessions=("date", "size"),
            labeled_targets=("target_available", "sum"),
            strict_model_ready=("strict_model_ready", "sum"),
        )
    )

    labeled = master[
        master["target_direction_t_plus_1"].notna()
    ]
    up_count = int(
        (labeled["target_direction_t_plus_1"] == 1).sum()
    )
    down_or_flat_count = int(
        (labeled["target_direction_t_plus_1"] == 0).sum()
    )

    missing_counts = {
        col: int(master[col].isna().sum())
        for col in [
            "sentiment_mean_t",
            "sentiment_std_t",
            "sentiment_roll_5",
            "sentiment_roll_10",
            "sentiment_roll_20",
            "news_intensity_20",
            "sentiment_x_volatility",
            "sentiment_roll_5_x_volatility",
            "sentiment_roll_20_x_volatility",
            "target_return_t_plus_1",
            "target_direction_t_plus_1",
        ]
    }

    summary = {
        **headline_audit,
        "master_session_rows": int(len(master)),
        "first_session": master["date"].min().strftime("%Y-%m-%d"),
        "last_session": master["date"].max().strftime("%Y-%m-%d"),
        "no_news_session_count": int(no_news.sum()),
        "no_news_session_dates": no_news_dates,
        "labeled_next_session_targets": int(
            master["target_available"].sum()
        ),
        "unlabeled_final_sessions": int(
            (master["target_available"] == 0).sum()
        ),
        "next_session_up_targets": up_count,
        "next_session_down_or_flat_targets": down_or_flat_count,
        "next_session_up_rate": float(
            up_count / len(labeled)
        ),
        "strict_model_ready_rows": int(
            master["strict_model_ready"].sum()
        ),
        "missing_counts": missing_counts,
        "yearly_rows": {
            str(int(year)): {
                "sessions": int(row["sessions"]),
                "labeled_targets": int(row["labeled_targets"]),
                "strict_model_ready": int(
                    row["strict_model_ready"]
                ),
            }
            for year, row in yearly.iterrows()
        },
        "unique_headline_count_stats": {
            "min": int(master["unique_headline_count_t"].min()),
            "median": float(
                master["unique_headline_count_t"].median()
            ),
            "mean": float(
                master["unique_headline_count_t"].mean()
            ),
            "max": int(master["unique_headline_count_t"].max()),
        },
        "sentiment_mean_stats_nonmissing": {
            "min": float(master["sentiment_mean_t"].min()),
            "median": float(master["sentiment_mean_t"].median()),
            "mean": float(master["sentiment_mean_t"].mean()),
            "max": float(master["sentiment_mean_t"].max()),
        },
        "news_intensity_stats_nonmissing": {
            "min": float(master["news_intensity_20"].min()),
            "median": float(master["news_intensity_20"].median()),
            "mean": float(master["news_intensity_20"].mean()),
            "max": float(master["news_intensity_20"].max()),
        },
        "validation": {
            "finbert_share_max_abs_sum_error": share_error,
            "rolling_sentiment_checks": rolling_errors,
            "prior_news_baseline_max_abs_error": prior_error,
            "target_return_max_abs_error": target_error,
            "target_date_mismatches": target_date_mismatch,
            "target_direction_mismatches": direction_mismatch,
            "predictor_truncation_invariance_max_abs_error": (
                leakage_max_error
            ),
            "predictor_truncation_missing_mismatches": (
                leakage_missing_mismatches
            ),
        },
        "feature_definitions": {
            "unique_headline_count_t": (
                "normalized unique titles within the timestamp-safe "
                "trading-session information window"
            ),
            "sentiment_mean_t": (
                "mean FinBERT score over session-unique headlines"
            ),
            "sentiment_std_t": (
                "sample standard deviation of FinBERT score over "
                "session-unique headlines; 0 when only one headline"
            ),
            "sentiment_roll_5_10_20": (
                "strict complete trading-session rolling means; "
                "missing if any session in the required window has "
                "missing sentiment"
            ),
            "news_intensity_20": (
                "current unique-headline count divided by mean unique-"
                "headline count over the prior 20 trading sessions"
            ),
            "target_return_t_plus_1": (
                "next genuine CSI 300 trading-session log return"
            ),
            "target_direction_t_plus_1": (
                "1 if next-session log return > 0, else 0; missing "
                "for the final session without a known next session"
            ),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-input",
        default=(
            "data/session_sentiment_timestamp_safe_2023_2026.csv"
        ),
    )
    parser.add_argument(
        "--aligned-headlines",
        default=(
            "data/timestamp_aligned_headlines_2023_2026.csv.gz"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/master_session_dataset_2023_2026.csv",
    )
    parser.add_argument(
        "--summary",
        default="data/master_session_dataset_2023_2026_summary.json",
    )
    args = parser.parse_args()

    session_market = load_session_market(
        Path(args.session_input)
    )
    session_news, headline_audit = aggregate_unique_headlines(
        Path(args.aligned_headlines)
    )
    master = build_master(
        session_market,
        session_news,
    )
    summary = validate_master(
        master,
        session_market,
        headline_audit,
    )

    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)

    save = master.copy()
    save["date"] = save["date"].dt.strftime("%Y-%m-%d")
    save["target_session_date"] = (
        save["target_session_date"]
        .dt.strftime("%Y-%m-%d")
    )
    save.to_csv(output, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Master session dataset: {output}")
    print(f"Validation summary: {summary_path}")


if __name__ == "__main__":
    main()
