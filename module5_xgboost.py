import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss

from config.research_config import (
    MIN_DIRECTION_TRAIN_OBSERVATIONS,
    RANDOM_STATE,
)
from module4_features import MARKET_ONLY_FEATURES, MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = os.path.join("models", "news_alpha_xgboost.pkl")


def load_data() -> pd.DataFrame:
    path = os.path.join(os.getcwd(), "data", "final_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing data/final_dataset.csv. Run module4_features.py first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["target_return_t+1"]).reset_index(drop=True)
    return df


def _new_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=2,
        learning_rate=0.03,
        n_estimators=140,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _fit_or_constant(X: pd.DataFrame, y: pd.Series):
    classes = np.unique(y)
    if len(classes) < 2:
        return {"constant_probability": float(np.mean(y))}
    model = _new_model()
    model.fit(X, y)
    return model


def _predict_probability(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(X), model["constant_probability"], dtype=float)
    return model.predict_proba(X)[:, 1]


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    predicted = (probability >= 0.5).astype(int)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def walk_forward_evaluation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate true one-step expanding-window out-of-sample probabilities.

    At each forecast date, both models are refit using only labelled rows that
    occurred earlier in time. Feature definitions are fixed ex ante. No
    full-sample scaler, target-based feature selection, or random
    cross-validation is used.
    """
    min_train = MIN_DIRECTION_TRAIN_OBSERVATIONS
    if len(df) <= min_train:
        raise ValueError(
            f"More than {MIN_DIRECTION_TRAIN_OBSERVATIONS} labelled observations "
            "are required after feature construction for walk-forward evaluation."
        )

    X_full = df[MODEL_FEATURES]
    X_market = df[MARKET_ONLY_FEATURES]
    y = (df["target_return_t+1"] > 0).astype(int)

    prediction_rows = []
    for idx in range(min_train, len(df)):
        train_idx = np.arange(0, idx)
        test_idx = [idx]

        full_model = _fit_or_constant(
            X_full.iloc[train_idx],
            y.iloc[train_idx],
        )
        market_model = _fit_or_constant(
            X_market.iloc[train_idx],
            y.iloc[train_idx],
        )

        p_full = float(
            _predict_probability(full_model, X_full.iloc[test_idx])[0]
        )
        p_market = float(
            _predict_probability(market_model, X_market.iloc[test_idx])[0]
        )

        prediction_rows.append(
            {
                "date": df.loc[idx, "date"],
                "train_observations": int(idx),
                "actual_return_t+1": float(
                    df.loc[idx, "target_return_t+1"]
                ),
                "actual_direction_t+1": int(y.iloc[idx]),
                "direction_probability_full": p_full,
                "direction_probability_market_only": p_market,
                "predicted_direction_full": int(p_full >= 0.5),
                "predicted_direction_market_only": int(p_market >= 0.5),
            }
        )

    pred = pd.DataFrame(prediction_rows).sort_values("date").reset_index(drop=True)
    y_oos = pred["actual_direction_t+1"].to_numpy()
    full_metrics = _metrics(
        y_oos,
        pred["direction_probability_full"].to_numpy(),
    )
    market_metrics = _metrics(
        y_oos,
        pred["direction_probability_market_only"].to_numpy(),
    )

    metrics = pd.DataFrame(
        [
            {"model": "market_plus_sentiment", **full_metrics},
            {"model": "market_only", **market_metrics},
        ]
    )
    return pred, metrics


def fit_final_model(df: pd.DataFrame) -> dict:
    X = df[MODEL_FEATURES]
    y = (df["target_return_t+1"] > 0).astype(int)
    model = _fit_or_constant(X, y)
    if isinstance(model, dict):
        raise ValueError("Final training sample contains only one direction class.")

    means = X.mean().to_dict()
    stds = X.std(ddof=1).replace(0, 1.0).fillna(1.0).to_dict()
    return {
        "model": model,
        "feature_cols": MODEL_FEATURES,
        "market_only_feature_cols": MARKET_ONLY_FEATURES,
        "feature_mean": means,
        "feature_std": stds,
        "training_start": df["date"].min().strftime("%Y-%m-%d"),
        "training_end": df["date"].max().strftime("%Y-%m-%d"),
        "model_version": "research-v2",
        "target": "next_trading_session_direction",
        "methodology": "expanding-window OOS evaluation; final model refit on all labeled history",
    }


def main():
    logging.info("Starting time-safe XGBoost evaluation")
    df = load_data()
    predictions, metrics = walk_forward_evaluation(df)

    os.makedirs(os.path.join(os.getcwd(), "outputs"), exist_ok=True)
    pred_path = os.path.join(os.getcwd(), "outputs", "oos_predictions.csv")
    metrics_path = os.path.join(os.getcwd(), "outputs", "model_evaluation.csv")
    predictions.assign(date=predictions["date"].dt.strftime("%Y-%m-%d")).to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    artifact = fit_final_model(df)
    os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)
    model_path = os.path.join(os.getcwd(), MODEL_PATH)
    joblib.dump(artifact, model_path)

    model = artifact["model"]
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": model.feature_importances_})
        importance.sort_values("importance", ascending=False).to_csv(
            os.path.join(os.getcwd(), "outputs", "feature_importance.csv"), index=False
        )

    metadata = {k: v for k, v in artifact.items() if k != "model"}
    with open(os.path.join(os.getcwd(), "models", "news_alpha_xgboost_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Saved OOS predictions, evaluation metrics, and canonical model artifact")


if __name__ == "__main__":
    main()
