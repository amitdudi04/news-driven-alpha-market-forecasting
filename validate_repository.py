"""Static and optional local-artifact validation for the research repository."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.exists():
        fail(f"Missing required JSON: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_static_structure() -> None:
    tracked = tracked_files()

    required = {
        "README.md",
        "DATA_CARD.md",
        "MODEL_CARD.md",
        "PIPELINE.md",
        "ARCHITECTURE.md",
        "PROJECT_LIMITATIONS.md",
        "ACADEMIC_DISCLOSURE.md",
        "app.py",
        "headline_utils.py",
        "bigquery_gdelt_backfill.py",
        "finbert_backfill.py",
        "build_historical_market.py",
        "build_timestamp_safe_alignment.py",
        "build_master_session_dataset.py",
        "run_directional_experiment.py",
        "evaluate_oos_uncertainty.py",
        "run_garch_oos_simulation.py",
        "config/directional_experiment_2023_2026.json",
        "config/oos_uncertainty_spec.json",
        "config/garch_simulation_spec.json",
        "tests/test_research_methodology.py",
        "docs/RESULTS.md",
        "docs/INTERVIEW_DEFENSE_2023_2026.md",
    }
    missing = sorted(required - tracked)
    if missing:
        fail(f"Required tracked files missing: {missing}")

    obsolete = {
        "descriptive_analysis.py",
        "module1_news.py",
        "module2_sentiment.py",
        "module3_market.py",
        "module4_features.py",
        "module5_xgboost.py",
        "module6_garch.py",
        "module7_backtesting.py",
        "module12_inference.py",
        "module13_signal_engine.py",
        "run_research_pipeline.py",
        "run_daily_pipeline.py",
        "run_one_year_rebuild.py",
        "sampled_news_backfill.py",
        "run_sampled_news_backfill.py",
        "config/research_config.py",
        "data/news_daily.csv",
        "data/sentiment_features.csv",
        "data/csi300_features.csv",
    }
    still_tracked = sorted(obsolete & tracked)
    if still_tracked:
        fail(
            "Obsolete short-sample prototype still tracked: "
            f"{still_tracked}"
        )

    public_docs = [
        "README.md",
        "DATA_CARD.md",
        "MODEL_CARD.md",
        "PIPELINE.md",
        "ARCHITECTURE.md",
        "PROJECT_LIMITATIONS.md",
        "ACADEMIC_DISCLOSURE.md",
        "docs/RESULTS.md",
        "docs/INTERVIEW_DEFENSE_2023_2026.md",
    ]
    forbidden_phrases = [
        "49 news days",
        "6,919",
        "only 8 labelled",
        "only 8 labeled",
        "2026-04-22",
        "run_research_pipeline.py",
        "data/news_daily.csv",
        "data/csi300_features.csv",
    ]
    for relative in public_docs:
        text = (
            ROOT / relative
        ).read_text(encoding="utf-8").casefold()
        for phrase in forbidden_phrases:
            if phrase.casefold() in text:
                fail(
                    f"Stale phrase {phrase!r} remains in {relative}"
                )

    # Check local Markdown links without requiring network access.
    import re

    for relative in public_docs:
        raw = (ROOT / relative).read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", raw):
            target = target.strip()
            if (
                target.startswith(("http://", "https://", "#", "mailto:"))
                or not target
            ):
                continue
            clean = target.split("#", 1)[0]
            resolved = (ROOT / relative).parent / clean
            if not resolved.exists():
                fail(
                    f"Broken local Markdown link in {relative}: {target}"
                )

    requirement_names = {
        line.split("==", 1)[0].strip().lower()
        for line in (ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required_dependencies = {
        "akshare",
        "arch",
        "google-cloud-bigquery",
        "joblib",
        "numpy",
        "pandas",
        "plotly",
        "requests",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "streamlit",
        "torch",
        "transformers",
        "xgboost",
    }
    missing_dependencies = sorted(
        required_dependencies - requirement_names
    )
    if missing_dependencies:
        fail(
            "Required dependency declarations missing: "
            f"{missing_dependencies}"
        )


def validate_specs() -> None:
    direction = load_json(
        "config/directional_experiment_2023_2026.json"
    )
    if not direction.get("frozen_before_model_fit"):
        fail("Directional experiment is not marked frozen.")
    if direction.get("partition_key") != "target_session_date":
        fail("Directional split must use target_session_date.")

    expected_split = {
        "initial_development_train": (2023, 221),
        "chronological_validation": (2024, 242),
        "untouched_final_holdout": (2025, 223),
        "post_sample_robustness": (2026, 181),
    }
    for key, (year, rows) in expected_split.items():
        item = direction["split"][key]
        if int(item["target_year"]) != year:
            fail(f"Unexpected target year in split {key}.")
        if int(item["rows"]) != rows:
            fail(f"Unexpected row count in split {key}.")

    if len(direction.get("master_dataset_sha256", "")) != 64:
        fail("Frozen master dataset SHA-256 is malformed.")

    uncertainty = load_json(
        "config/oos_uncertainty_spec.json"
    )
    boot = uncertainty["bootstrap"]
    if int(boot["replications"]) != 5000:
        fail("Bootstrap replication count changed.")
    if int(boot["block_length_sessions"]) != 10:
        fail("Bootstrap block length changed.")
    if not bool(boot["paired_indices"]):
        fail("Bootstrap must remain paired.")

    garch = load_json(
        "config/garch_simulation_spec.json"
    )
    signal = garch["directional_signal"]
    if float(signal["long_probability_threshold"]) != 0.55:
        fail("Long threshold changed.")
    if float(signal["short_probability_threshold"]) != 0.45:
        fail("Short threshold changed.")
    if bool(signal["thresholds_optimized"]):
        fail("Signal thresholds must remain non-optimized.")
    cost = garch["transaction_cost"]
    if float(cost["rate_per_unit_turnover"]) != 0.001:
        fail("Transaction-cost convention changed.")
    if bool(cost["optimized"]):
        fail("Transaction-cost convention must remain non-optimized.")


def validate_local_artifacts_if_present() -> None:
    direction = load_json(
        "config/directional_experiment_2023_2026.json"
    )

    market_summary = (
        ROOT / "data/csi300_market_2022_2026_summary.json"
    )
    if market_summary.exists():
        summary = json.loads(
            market_summary.read_text(encoding="utf-8")
        )
        if int(summary["total_trading_sessions"]) != 1150:
            fail("Local market session count is not 1,150.")
        if int(summary["research_trading_sessions"]) != 908:
            fail("Local research market session count is not 908.")
        if int(summary["duplicate_market_dates"]) != 0:
            fail("Duplicate market dates remain.")

    alignment_summary = (
        ROOT / "data/timestamp_alignment_2023_2026_summary.json"
    )
    if alignment_summary.exists():
        summary = json.loads(
            alignment_summary.read_text(encoding="utf-8")
        )
        if int(summary["input_headlines_joined"]) != 292373:
            fail("Local aligned headline input count changed.")
        if int(summary["assigned_headlines_total"]) != 291973:
            fail("Local assigned headline count changed.")
        if int(summary["pending_after_last_known_close"]) != 400:
            fail("Local pending headline count changed.")
        if int(summary["monthly_unmatched_rows"]) != 0:
            fail("Local article/score mismatches remain.")
        if int(summary["monthly_local_date_mismatches"]) != 0:
            fail("Local Shanghai-date mismatches remain.")
        if int(summary["monthly_timing_violations"]) != 0:
            fail("Local timing violations remain.")

    master_path = (
        ROOT / "data/master_session_dataset_2023_2026.csv"
    )
    if master_path.exists():
        actual_hash = sha256_file(master_path)
        if actual_hash != direction["master_dataset_sha256"]:
            fail(
                "Local master dataset hash differs from frozen spec."
            )
        master = pd.read_csv(master_path)
        if len(master) != 908:
            fail("Local master dataset row count is not 908.")
        if int(master["strict_model_ready"].sum()) != 867:
            fail("Local strict model-ready count is not 867.")
        if int(master["target_available"].sum()) != 907:
            fail("Local target count is not 907.")
        if master["date"].duplicated().any():
            fail("Duplicate master session dates detected.")

    direction_dir = (
        ROOT / "outputs/directional_experiment_v1"
    )
    metrics_path = direction_dir / "metrics.csv"
    pred_path = direction_dir / "predictions.csv"
    if metrics_path.exists() and pred_path.exists():
        metrics = pd.read_csv(metrics_path)
        predictions = pd.read_csv(pred_path)
        if len(metrics) != 12:
            fail("Directional metrics should contain 12 rows.")
        if len(predictions) != 2584:
            fail("Directional predictions should contain 2,584 rows.")
        expected_period_counts = {
            "2024_validation": 242,
            "2025_holdout": 223,
            "2026_robustness": 181,
        }
        grouped = (
            predictions.groupby(
                ["model_name", "period"]
            ).size()
        )
        for (_, period), count in grouped.items():
            if int(count) != expected_period_counts[period]:
                fail(
                    f"Unexpected prediction count for {period}: "
                    f"{count}"
                )

    bootstrap_path = (
        direction_dir
        / "uncertainty/paired_block_bootstrap_summary.csv"
    )
    if bootstrap_path.exists():
        bootstrap = pd.read_csv(bootstrap_path)
        if len(bootstrap) != 30:
            fail("Bootstrap summary should contain 30 rows.")
        if not (
            bootstrap["valid_bootstrap_draws"].astype(int)
            == 5000
        ).all():
            fail("A bootstrap metric has fewer than 5,000 valid draws.")

    garch_dir = ROOT / "outputs/garch_simulation_v1"
    forecast_path = garch_dir / "garch_forecasts.csv"
    simulation_path = garch_dir / "simulation_metrics.csv"
    if forecast_path.exists():
        forecasts = pd.read_csv(forecast_path)
        if len(forecasts) != 907:
            fail("GARCH forecast count is not 907.")
        if forecasts["date"].duplicated().any():
            fail("Duplicate GARCH dates detected.")
        if int((forecasts["convergence_flag"] != 0).sum()) != 0:
            fail("GARCH convergence failures detected.")
    if simulation_path.exists():
        simulation = pd.read_csv(simulation_path)
        if len(simulation) != 24:
            fail("Simulation metrics should contain 24 rows.")


def main() -> None:
    validate_static_structure()
    validate_specs()
    validate_local_artifacts_if_present()
    print(
        "Repository validation passed: current pipeline, frozen "
        "specifications, documentation, and available local artifacts "
        "are internally consistent."
    )


if __name__ == "__main__":
    main()
