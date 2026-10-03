import logging
import os

import joblib
import numpy as np
import pandas as pd

from module4_features import generate_latest_features, load_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = os.path.join("models", "news_alpha_xgboost.pkl")


def load_inference_model():
    path = os.path.join(os.getcwd(), MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing models/news_alpha_xgboost.pkl. Run module5_xgboost.py first."
        )
    return joblib.load(path)


def predict_next_day(artifact: dict, latest_features: pd.DataFrame) -> dict:
    if latest_features.empty:
        raise ValueError("Empty feature frame supplied for inference.")

    feature_cols = artifact["feature_cols"]
    missing = [c for c in feature_cols if c not in latest_features.columns]
    if missing:
        raise ValueError(f"Missing inference features: {missing}")

    X = latest_features[feature_cols].astype(float)
    probability = float(artifact["model"].predict_proba(X)[0, 1])
    strength = float(abs(probability - 0.5) * 2.0)

    means = pd.Series(artifact.get("feature_mean", {}), dtype=float)
    stds = pd.Series(artifact.get("feature_std", {}), dtype=float).replace(0, 1.0)
    aligned_means = means.reindex(feature_cols).fillna(0.0)
    aligned_stds = stds.reindex(feature_cols).replace(0, 1.0).fillna(1.0)
    z = (X.iloc[0] - aligned_means) / aligned_stds
    max_abs_train_z = float(np.nanmax(np.abs(z.to_numpy())))

    return {
        "direction_probability": probability,
        "direction_strength": strength,
        "predicted_direction": int(probability >= 0.5),
        "regime_dummy": int(round(float(X["regime_dummy"].iloc[0]))),
        "max_abs_train_z": max_abs_train_z,
        "model_version": artifact.get("model_version", "research-v2"),
    }


def main(execution_uuid: str | None = None):
    logging.info("Running next-trading-session research inference")
    sent_df, market_df = load_datasets()
    latest = generate_latest_features(sent_df, market_df)
    artifact = load_inference_model()
    result = predict_next_day(artifact, latest)

    if result["max_abs_train_z"] > 8:
        logging.warning(
            "Latest feature vector is far from the training distribution (max |z| %.2f).",
            result["max_abs_train_z"],
        )

    out = pd.DataFrame(
        [
            {
                "Feature_Date": pd.to_datetime(latest["date"].iloc[0]).strftime("%Y-%m-%d"),
                "Direction_Prob_t+1": result["direction_probability"],
                "Direction_Strength": result["direction_strength"],
                "Predicted_Direction_t+1": result["predicted_direction"],
                "Regime_Dummy": result["regime_dummy"],
                "Max_Abs_Train_Z": result["max_abs_train_z"],
                "Model_Version": result["model_version"],
                "execution_uuid": execution_uuid or "manual-research-run",
            }
        ]
    )

    os.makedirs(os.path.join(os.getcwd(), "outputs"), exist_ok=True)
    out.to_csv(
        os.path.join(os.getcwd(), "outputs", "live_inference.csv"),
        index=False,
    )
    logging.info("Saved latest inference to outputs/live_inference.csv")


if __name__ == "__main__":
    main()
