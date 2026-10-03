import logging
import os
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from config.research_config import MIN_GARCH_TRAIN_OBSERVATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data() -> pd.DataFrame:
    path = os.path.join(os.getcwd(), "data", "csi300_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing data/csi300_features.csv. Run module3_market.py first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["target_volatility_t+1"] = df["volatility"].shift(-1)
    return df


def one_step_garch_forecasts(
    df: pd.DataFrame,
    min_train: int = MIN_GARCH_TRAIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Refit GARCH(1,1) through day t and forecast volatility for t+1."""
    if len(df) <= min_train:
        raise ValueError(f"Need more than {min_train} observations for GARCH forecasting.")

    returns = pd.to_numeric(df["return"], errors="coerce")
    rows = []
    for i in range(min_train - 1, len(df)):
        train = returns.iloc[: i + 1].dropna() * 100.0
        if len(train) < min_train:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(
                    train,
                    mean="Zero",
                    vol="GARCH",
                    p=1,
                    q=1,
                    dist="normal",
                    rescale=False,
                )
                result = model.fit(disp="off")
                variance = float(
                    result.forecast(horizon=1, reindex=False).variance.values[-1, 0]
                )
            forecast_vol = np.sqrt(max(variance, 0.0)) / 100.0
        except Exception as exc:
            logging.warning("GARCH fit failed on %s: %s", df.loc[i, "date"], exc)
            continue

        rows.append(
            {
                "date": df.loc[i, "date"],
                "pred_vol_t+1": forecast_vol,
                "actual_volatility_t+1": df.loc[i, "target_volatility_t+1"],
            }
        )

    return pd.DataFrame(rows)


def main():
    logging.info("Starting one-step-ahead GARCH risk forecasts")
    df = load_data()
    forecasts = one_step_garch_forecasts(df)
    os.makedirs(os.path.join(os.getcwd(), "outputs"), exist_ok=True)
    out = forecasts.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(
        os.path.join(os.getcwd(), "outputs", "garch_oos_forecasts.csv"),
        index=False,
    )
    logging.info("Saved %s GARCH forecasts", len(out))


if __name__ == "__main__":
    main()
