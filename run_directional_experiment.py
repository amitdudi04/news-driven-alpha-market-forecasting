"""Run the frozen four-model directional experiment.

This script refuses to run if the master dataset hash differs from the
pre-model-fit frozen experiment specification.

Principal models:
1. Logistic market-only
2. Logistic market + sentiment
3. XGBoost market-only
4. XGBoost market + sentiment

Hyperparameters are selected using only 2024 monthly expanding-window
validation. The selected model is refit on 2023-2024 and then evaluated,
without retuning, on 2025 holdout and 2026 post-sample robustness.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SPEC_PATH = Path(
    "config/directional_experiment_2023_2026.json"
)
OUTPUT_DIR = Path("outputs/directional_experiment_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec() -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not spec.get("frozen_before_model_fit"):
        raise ValueError("Experiment specification is not frozen.")
    return spec


def load_dataset(spec: dict[str, Any]) -> pd.DataFrame:
    path = Path(spec["master_dataset"])
    actual_hash = sha256_file(path)
    expected_hash = spec["master_dataset_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            "Master dataset SHA-256 differs from frozen experiment "
            f"specification. expected={expected_hash}, actual={actual_hash}"
        )

    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["target_session_date"] = pd.to_datetime(
        data["target_session_date"],
        errors="coerce",
    )
    eligible = data[
        data["strict_model_ready"].eq(1)
    ].copy()
    if eligible["target_session_date"].isna().any():
        raise ValueError(
            "An eligible modeling row lacks target_session_date."
        )

    eligible["target_year"] = (
        eligible["target_session_date"].dt.year
    )
    eligible["target_month"] = (
        eligible["target_session_date"].dt.to_period("M")
    )

    expected_counts = {
        int(year): int(info["rows"])
        for year, info in {
            "2023": spec["split"]["initial_development_train"],
            "2024": spec["split"]["chronological_validation"],
            "2025": spec["split"]["untouched_final_holdout"],
            "2026": spec["split"]["post_sample_robustness"],
        }.items()
    }
    actual_counts = (
        eligible.groupby("target_year").size().to_dict()
    )
    if actual_counts != expected_counts:
        raise ValueError(
            "Frozen split row counts do not match the current dataset. "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    return eligible.sort_values(
        "target_session_date"
    ).reset_index(drop=True)


def features_for_variant(
    spec: dict[str, Any],
    variant: str,
) -> list[str]:
    market = list(spec["market_only_features"])
    if variant == "market_only":
        return market
    if variant == "market_plus_sentiment":
        return market + list(spec["sentiment_features_added"])
    raise ValueError(f"Unknown feature variant: {variant}")


def make_logistic(
    candidate: dict[str, Any],
    spec: dict[str, Any],
) -> Pipeline:
    settings = spec["model_families"]["logistic_regression"]
    model = LogisticRegression(
        C=float(candidate["C"]),
        penalty=settings["penalty"],
        solver=settings["solver"],
        max_iter=int(settings["max_iter"]),
        class_weight=settings["class_weight"],
        random_state=2027,
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def make_xgboost(
    candidate: dict[str, Any],
    spec: dict[str, Any],
) -> XGBClassifier:
    settings = spec["model_families"]["xgboost"]
    params = {
        key: value
        for key, value in candidate.items()
        if key != "id"
    }
    return XGBClassifier(
        **params,
        objective=settings["objective"],
        eval_metric=settings["eval_metric"],
        tree_method=settings["tree_method"],
        n_jobs=int(settings["n_jobs"]),
        random_state=int(settings["random_state"]),
        verbosity=0,
    )


def make_model(
    family: str,
    candidate: dict[str, Any],
    spec: dict[str, Any],
):
    if family == "logistic":
        return make_logistic(candidate, spec)
    if family == "xgboost":
        return make_xgboost(candidate, spec)
    raise ValueError(f"Unknown family: {family}")


def metric_dict(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y_true)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted)
        ),
        "accuracy": float(
            accuracy_score(y_true, predicted)
        ),
        "brier_score_loss": float(
            brier_score_loss(y_true, probability)
        ),
        "log_loss": float(
            log_loss(y_true, probability, labels=[0, 1])
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probability)
        ),
        "mean_predicted_up_probability": float(
            probability.mean()
        ),
        "observed_up_rate": float(
            y_true.mean()
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def chronological_validation_predictions(
    data: pd.DataFrame,
    features: list[str],
    family: str,
    candidate: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    validation = data[data["target_year"].eq(2024)].copy()
    months = sorted(validation["target_month"].unique())

    prediction_parts = []
    folds = []

    for fold_number, month in enumerate(months, start=1):
        test = validation[
            validation["target_month"].eq(month)
        ].copy()
        first_target = test["target_session_date"].min()

        train = data[
            (data["target_session_date"] < first_target)
            & (data["target_year"].isin([2023, 2024]))
        ].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"Empty expanding fold at {month}."
            )
        if train["target_direction_t_plus_1"].nunique() < 2:
            raise ValueError(
                f"Training fold at {month} lacks both classes."
            )

        model = make_model(
            family,
            candidate,
            spec,
        )
        model.fit(
            train[features],
            train["target_direction_t_plus_1"].astype(int),
        )
        probability = model.predict_proba(
            test[features]
        )[:, 1]

        fold_predictions = test[
            [
                "date",
                "target_session_date",
                "target_direction_t_plus_1",
            ]
        ].copy()
        fold_predictions["probability"] = probability
        fold_predictions["fold_month"] = str(month)
        prediction_parts.append(fold_predictions)

        folds.append(
            {
                "fold_number": fold_number,
                "validation_month": str(month),
                "train_rows": int(len(train)),
                "validation_rows": int(len(test)),
                "train_last_target_date": train[
                    "target_session_date"
                ].max().strftime("%Y-%m-%d"),
                "validation_first_target_date": first_target.strftime(
                    "%Y-%m-%d"
                ),
                "validation_last_target_date": test[
                    "target_session_date"
                ].max().strftime("%Y-%m-%d"),
            }
        )

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    ).sort_values("target_session_date")

    if len(predictions) != len(validation):
        raise ValueError(
            "2024 OOS validation predictions do not cover all rows."
        )
    if predictions["target_session_date"].duplicated().any():
        raise ValueError(
            "Duplicate 2024 OOS validation predictions detected."
        )
    return predictions.reset_index(drop=True), folds


def select_hyperparameters(
    data: pd.DataFrame,
    family: str,
    variant: str,
    spec: dict[str, Any],
    threshold: float,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    features = features_for_variant(spec, variant)
    if family == "logistic":
        grid = spec["model_families"][
            "logistic_regression"
        ]["candidate_grid"]
    else:
        grid = spec["model_families"]["xgboost"][
            "candidate_grid"
        ]

    candidate_rows = []
    prediction_map: dict[str, pd.DataFrame] = {}
    fold_map: dict[str, list[dict[str, Any]]] = {}

    for candidate in grid:
        predictions, folds = (
            chronological_validation_predictions(
                data=data,
                features=features,
                family=family,
                candidate=candidate,
                spec=spec,
            )
        )
        metrics = metric_dict(
            predictions["target_direction_t_plus_1"],
            predictions["probability"],
            threshold,
        )
        candidate_id = candidate["id"]
        candidate_rows.append(
            {
                "family": family,
                "variant": variant,
                "candidate_id": candidate_id,
                **metrics,
            }
        )
        prediction_map[candidate_id] = predictions
        fold_map[candidate_id] = folds

    table = pd.DataFrame(candidate_rows).sort_values(
        [
            "brier_score_loss",
            "log_loss",
            "candidate_id",
        ],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    selected_id = str(table.iloc[0]["candidate_id"])
    selected = next(
        candidate
        for candidate in grid
        if candidate["id"] == selected_id
    )

    return (
        dict(selected),
        table,
        prediction_map[selected_id],
        fold_map[selected_id],
    )


def fit_final_and_predict(
    data: pd.DataFrame,
    features: list[str],
    family: str,
    candidate: dict[str, Any],
    spec: dict[str, Any],
    evaluation_year: int,
) -> tuple[Any, pd.DataFrame]:
    train = data[
        data["target_year"].isin([2023, 2024])
    ].copy()
    test = data[
        data["target_year"].eq(evaluation_year)
    ].copy()

    if test.empty:
        raise ValueError(
            f"No evaluation rows for {evaluation_year}."
        )

    model = make_model(family, candidate, spec)
    model.fit(
        train[features],
        train["target_direction_t_plus_1"].astype(int),
    )
    probability = model.predict_proba(
        test[features]
    )[:, 1]

    predictions = test[
        [
            "date",
            "target_session_date",
            "target_direction_t_plus_1",
            "target_return_t_plus_1",
        ]
    ].copy()
    predictions["probability"] = probability
    return model, predictions


def incremental_rows(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for family in ["logistic", "xgboost"]:
        for period in [
            "2024_validation",
            "2025_holdout",
            "2026_robustness",
        ]:
            pair = metrics[
                (metrics["family"] == family)
                & (metrics["period"] == period)
            ].set_index("variant")
            market = pair.loc["market_only"]
            sentiment = pair.loc["market_plus_sentiment"]
            rows.append(
                {
                    "family": family,
                    "period": period,
                    "n": int(market["n"]),
                    "balanced_accuracy_improvement": float(
                        sentiment["balanced_accuracy"]
                        - market["balanced_accuracy"]
                    ),
                    "accuracy_improvement": float(
                        sentiment["accuracy"]
                        - market["accuracy"]
                    ),
                    "roc_auc_improvement": float(
                        sentiment["roc_auc"]
                        - market["roc_auc"]
                    ),
                    "brier_improvement": float(
                        market["brier_score_loss"]
                        - sentiment["brier_score_loss"]
                    ),
                    "log_loss_improvement": float(
                        market["log_loss"]
                        - sentiment["log_loss"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    spec = load_spec()
    data = load_dataset(spec)
    threshold = float(
        spec["target"]["probability_threshold"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_params: dict[str, Any] = {}
    candidate_tables = []
    metric_rows = []
    all_predictions = []
    fold_records = []

    print(
        "Frozen dataset hash verified. "
        "Beginning 2024-only hyperparameter selection.",
        flush=True,
    )

    selected_prediction_cache = {}

    for family in ["logistic", "xgboost"]:
        for variant in [
            "market_only",
            "market_plus_sentiment",
        ]:
            model_name = f"{family}_{variant}"
            print(
                f"Selecting {model_name} using 2024 monthly "
                "expanding validation only...",
                flush=True,
            )
            (
                selected,
                candidate_table,
                validation_predictions,
                folds,
            ) = select_hyperparameters(
                data=data,
                family=family,
                variant=variant,
                spec=spec,
                threshold=threshold,
            )

            selected_params[model_name] = selected
            candidate_tables.append(candidate_table)
            selected_prediction_cache[model_name] = (
                validation_predictions
            )

            for fold in folds:
                fold_records.append(
                    {
                        "family": family,
                        "variant": variant,
                        "selected_candidate_id": selected["id"],
                        **fold,
                    }
                )

            validation_metrics = metric_dict(
                validation_predictions[
                    "target_direction_t_plus_1"
                ],
                validation_predictions["probability"],
                threshold,
            )
            metric_rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "model_name": model_name,
                    "period": "2024_validation",
                    "selected_candidate_id": selected["id"],
                    **validation_metrics,
                }
            )
            vp = validation_predictions.copy()
            vp["family"] = family
            vp["variant"] = variant
            vp["model_name"] = model_name
            vp["period"] = "2024_validation"
            vp["predicted_direction"] = (
                vp["probability"] >= threshold
            ).astype(int)
            all_predictions.append(vp)

    # Persist the selections before any holdout evaluation is executed.
    selection_path = (
        OUTPUT_DIR / "selected_hyperparameters_from_2024.json"
    )
    selection_path.write_text(
        json.dumps(selected_params, indent=2),
        encoding="utf-8",
    )
    print(
        "2024 model selection complete and persisted. "
        "Now evaluating untouched 2025 holdout and locked-model 2026.",
        flush=True,
    )

    final_models: dict[str, Any] = {}

    for family in ["logistic", "xgboost"]:
        for variant in [
            "market_only",
            "market_plus_sentiment",
        ]:
            model_name = f"{family}_{variant}"
            features = features_for_variant(spec, variant)
            selected = selected_params[model_name]

            # Fit exactly once on all 2023-2024 eligible targets.
            model, holdout_predictions = fit_final_and_predict(
                data=data,
                features=features,
                family=family,
                candidate=selected,
                spec=spec,
                evaluation_year=2025,
            )
            final_models[model_name] = model

            holdout_metrics = metric_dict(
                holdout_predictions[
                    "target_direction_t_plus_1"
                ],
                holdout_predictions["probability"],
                threshold,
            )
            metric_rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "model_name": model_name,
                    "period": "2025_holdout",
                    "selected_candidate_id": selected["id"],
                    **holdout_metrics,
                }
            )
            hp = holdout_predictions.copy()
            hp["family"] = family
            hp["variant"] = variant
            hp["model_name"] = model_name
            hp["period"] = "2025_holdout"
            hp["predicted_direction"] = (
                hp["probability"] >= threshold
            ).astype(int)
            all_predictions.append(hp)

            # Locked model: no 2025 refit.
            robustness = data[
                data["target_year"].eq(2026)
            ].copy()
            robustness_probability = model.predict_proba(
                robustness[features]
            )[:, 1]
            robustness_predictions = robustness[
                [
                    "date",
                    "target_session_date",
                    "target_direction_t_plus_1",
                    "target_return_t_plus_1",
                ]
            ].copy()
            robustness_predictions["probability"] = (
                robustness_probability
            )

            robustness_metrics = metric_dict(
                robustness_predictions[
                    "target_direction_t_plus_1"
                ],
                robustness_predictions["probability"],
                threshold,
            )
            metric_rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "model_name": model_name,
                    "period": "2026_robustness",
                    "selected_candidate_id": selected["id"],
                    **robustness_metrics,
                }
            )
            rp = robustness_predictions.copy()
            rp["family"] = family
            rp["variant"] = variant
            rp["model_name"] = model_name
            rp["period"] = "2026_robustness"
            rp["predicted_direction"] = (
                rp["probability"] >= threshold
            ).astype(int)
            all_predictions.append(rp)

            joblib.dump(
                model,
                OUTPUT_DIR
                / "models"
                / f"{model_name}_fit_2023_2024.joblib",
            )

    metrics = pd.DataFrame(metric_rows)
    candidates = pd.concat(
        candidate_tables,
        ignore_index=True,
    )
    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )
    folds = pd.DataFrame(fold_records)
    incremental = incremental_rows(metrics)

    # Structural audits.
    final_train = data[
        data["target_year"].isin([2023, 2024])
    ]
    holdout = data[data["target_year"].eq(2025)]
    robustness = data[data["target_year"].eq(2026)]

    if final_train["target_session_date"].max() >= (
        holdout["target_session_date"].min()
    ):
        raise ValueError(
            "Final-fit target dates overlap the 2025 holdout."
        )
    if holdout["target_session_date"].max() >= (
        robustness["target_session_date"].min()
    ):
        raise ValueError(
            "2025 and 2026 target dates overlap."
        )

    expected_prediction_rows = (
        4 * (242 + 223 + 181)
    )
    if len(predictions) != expected_prediction_rows:
        raise ValueError(
            "Unexpected total prediction row count: "
            f"{len(predictions)} vs {expected_prediction_rows}"
        )

    metrics.to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False,
    )
    candidates.to_csv(
        OUTPUT_DIR / "validation_candidate_metrics.csv",
        index=False,
    )
    folds.to_csv(
        OUTPUT_DIR / "validation_folds.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )
    incremental.to_csv(
        OUTPUT_DIR / "incremental_sentiment_comparison.csv",
        index=False,
    )

    result = {
        "experiment_id": spec["experiment_id"],
        "dataset_sha256_verified": spec[
            "master_dataset_sha256"
        ],
        "strict_model_rows": int(len(data)),
        "split_counts_by_target_year": {
            str(int(year)): int(count)
            for year, count in (
                data.groupby("target_year").size().items()
            )
        },
        "final_fit_rows_2023_2024": int(len(final_train)),
        "selected_hyperparameters": selected_params,
        "metrics": metrics.to_dict(orient="records"),
        "incremental_sentiment_comparison": (
            incremental.to_dict(orient="records")
        ),
        "guardrails": spec["interpretation_guardrails"],
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (
        OUTPUT_DIR / "results_summary.json"
    ).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nSELECTED HYPERPARAMETERS")
    print(json.dumps(selected_params, indent=2))
    print("\nMETRICS")
    print(
        metrics[
            [
                "model_name",
                "period",
                "n",
                "balanced_accuracy",
                "accuracy",
                "brier_score_loss",
                "log_loss",
                "roc_auc",
                "mean_predicted_up_probability",
                "observed_up_rate",
            ]
        ].to_string(index=False)
    )
    print("\nINCREMENTAL SENTIMENT COMPARISON")
    print(incremental.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
