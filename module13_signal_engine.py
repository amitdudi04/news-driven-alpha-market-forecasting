import logging
import os

import numpy as np
import pandas as pd

from config.research_config import (
    LONG_PROBABILITY_THRESHOLD,
    MAX_ABS_POSITION,
    SHORT_PROBABILITY_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_signal(
    direction_probability: float,
    pred_volatility: float,
    target_volatility: float,
) -> dict:
    if not np.isfinite(direction_probability):
        raise ValueError("Direction probability must be finite.")

    if direction_probability >= LONG_PROBABILITY_THRESHOLD:
        signal = "LONG"
        direction = 1.0
    elif direction_probability <= SHORT_PROBABILITY_THRESHOLD:
        signal = "SHORT"
        direction = -1.0
    else:
        signal = "NO TRADE"
        direction = 0.0

    if direction == 0.0:
        position = 0.0
    else:
        safe_forecast = max(float(pred_volatility), 1e-8)
        vol_scale = np.clip(
            float(target_volatility) / safe_forecast,
            0.0,
            MAX_ABS_POSITION,
        )
        position = float(direction * vol_scale)

    return {
        "signal": signal,
        "position": position,
        "long_threshold": LONG_PROBABILITY_THRESHOLD,
        "short_threshold": SHORT_PROBABILITY_THRESHOLD,
    }


def main(execution_uuid: str | None = None):
    inference_path = os.path.join(
        os.getcwd(), "outputs", "live_inference.csv"
    )
    garch_path = os.path.join(
        os.getcwd(), "outputs", "garch_oos_forecasts.csv"
    )
    if not os.path.exists(inference_path) or not os.path.exists(garch_path):
        raise FileNotFoundError(
            "Run modules 6 and 12 before generating the paper-trading signal."
        )

    inference = pd.read_csv(inference_path)
    garch = pd.read_csv(garch_path)
    if inference.empty or garch.empty:
        raise ValueError("Inference or GARCH forecast output is empty.")

    latest_inf = inference.iloc[-1]
    feature_date = pd.to_datetime(latest_inf["Feature_Date"])
    garch["date"] = pd.to_datetime(garch["date"])
    matched_garch = garch[garch["date"] == feature_date]
    if matched_garch.empty:
        raise ValueError(
            f"No GARCH forecast is available for inference date {feature_date.date()}."
        )
    latest_garch = matched_garch.iloc[-1]
    historical_vol = pd.to_numeric(
        garch.loc[garch["date"] < feature_date, "pred_vol_t+1"],
        errors="coerce",
    ).dropna()
    if historical_vol.empty:
        target_vol = float(latest_garch["pred_vol_t+1"])
    else:
        target_vol = float(historical_vol.median())

    result = generate_signal(
        direction_probability=float(latest_inf["Direction_Prob_t+1"]),
        pred_volatility=float(latest_garch["pred_vol_t+1"]),
        target_volatility=target_vol,
    )

    explanation = (
        f"{result['signal']}: p(up)={float(latest_inf['Direction_Prob_t+1']):.3f}; "
        f"research thresholds={SHORT_PROBABILITY_THRESHOLD:.2f}/"
        f"{LONG_PROBABILITY_THRESHOLD:.2f}; "
        f"GARCH vol forecast={float(latest_garch['pred_vol_t+1']):.4f}"
    )

    row = pd.DataFrame(
        [
            {
                "date": str(latest_inf["Feature_Date"]),
                "direction_probability": float(
                    latest_inf["Direction_Prob_t+1"]
                ),
                "direction_strength": float(
                    latest_inf["Direction_Strength"]
                ),
                "signal": result["signal"],
                "position": result["position"],
                "pred_vol_t+1": float(latest_garch["pred_vol_t+1"]),
                "target_volatility": target_vol,
                "explanation": explanation,
                "model_version": str(latest_inf["Model_Version"]),
                "execution_uuid": (
                    execution_uuid
                    or latest_inf.get(
                        "execution_uuid",
                        "manual-research-run",
                    )
                ),
            }
        ]
    )

    out_path = os.path.join(
        os.getcwd(), "outputs", "daily_prediction.csv"
    )
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if "date" in existing.columns:
            existing = existing[
                existing["date"].astype(str)
                != str(row["date"].iloc[0])
            ]
        row = pd.concat([existing, row], ignore_index=True)

    row.to_csv(out_path, index=False)
    logging.info("Saved paper-trading signal to outputs/daily_prediction.csv")


if __name__ == "__main__":
    main()
