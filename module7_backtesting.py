import logging
import os

import numpy as np
import pandas as pd

from config.research_config import (
    LONG_PROBABILITY_THRESHOLD,
    MAX_ABS_POSITION,
    SHORT_PROBABILITY_THRESHOLD,
    TRANSACTION_COST,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    return float(((wealth - peak) / peak).min())


def build_backtest(predictions: pd.DataFrame, garch: pd.DataFrame) -> pd.DataFrame:
    garch = garch.sort_values("date").reset_index(drop=True).copy()
    # The volatility target for day t uses only GARCH forecasts available
    # strictly before t.
    garch["target_volatility"] = (
        garch["pred_vol_t+1"].shift(1).expanding(min_periods=5).median()
    )

    df = pd.merge(
        predictions,
        garch[["date", "pred_vol_t+1", "target_volatility"]],
        on="date",
        how="inner",
    )
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No overlapping OOS XGBoost and GARCH forecast dates.")

    p = df["direction_probability_full"]
    df["signal"] = np.select(
        [
            p >= LONG_PROBABILITY_THRESHOLD,
            p <= SHORT_PROBABILITY_THRESHOLD,
        ],
        [1.0, -1.0],
        default=0.0,
    )

    scale = (
        df["target_volatility"]
        / df["pred_vol_t+1"].replace(0, np.nan)
    ).clip(lower=0.0, upper=MAX_ABS_POSITION)
    scale = scale.fillna(1.0)
    df["position"] = df["signal"] * scale

    df["turnover"] = (
        df["position"].diff().abs().fillna(df["position"].abs())
    )
    df["transaction_cost"] = df["turnover"] * TRANSACTION_COST

    # Market data are stored as log returns. Convert to simple returns before
    # applying position weights and proportional transaction costs.
    df["benchmark_return"] = np.expm1(df["actual_return_t+1"])
    df["strategy_return"] = (
        df["position"] * df["benchmark_return"]
        - df["transaction_cost"]
    )

    if (df["strategy_return"] <= -1.0).any():
        raise ValueError("Strategy return at or below -100% is not economically valid.")

    df["strategy_wealth"] = (1.0 + df["strategy_return"]).cumprod()
    df["benchmark_wealth"] = (1.0 + df["benchmark_return"]).cumprod()
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    def sharpe_zero_rf(series: pd.Series) -> float:
        std = series.std(ddof=1)
        return (
            float(series.mean() / std * np.sqrt(252))
            if std and np.isfinite(std)
            else np.nan
        )

    active = df[df["signal"] != 0]
    directional_hit = (
        (
            np.sign(active["signal"])
            == np.sign(active["actual_return_t+1"])
        ).mean()
        if len(active)
        else np.nan
    )

    strategy_total = float(df["strategy_wealth"].iloc[-1] - 1.0)
    benchmark_total = float(df["benchmark_wealth"].iloc[-1] - 1.0)

    metrics = {
        "observations": int(len(df)),
        "active_observations": int(len(active)),
        "strategy_total_return": strategy_total,
        "benchmark_total_return": benchmark_total,
        "active_total_return": strategy_total - benchmark_total,
        "strategy_sharpe": sharpe_zero_rf(df["strategy_return"]),
        "benchmark_sharpe": sharpe_zero_rf(df["benchmark_return"]),
        "strategy_max_drawdown": _max_drawdown(df["strategy_wealth"]),
        "benchmark_max_drawdown": _max_drawdown(df["benchmark_wealth"]),
        "active_directional_hit_rate": (
            float(directional_hit)
            if np.isfinite(directional_hit)
            else np.nan
        ),
        "average_turnover": float(df["turnover"].mean()),
        "total_transaction_cost": float(df["transaction_cost"].sum()),
    }
    return pd.DataFrame([metrics])


def main():
    pred_path = os.path.join(os.getcwd(), "outputs", "oos_predictions.csv")
    garch_path = os.path.join(
        os.getcwd(), "outputs", "garch_oos_forecasts.csv"
    )
    if not os.path.exists(pred_path) or not os.path.exists(garch_path):
        raise FileNotFoundError(
            "Run modules 5 and 6 before the OOS backtest."
        )

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    garch = pd.read_csv(garch_path, parse_dates=["date"])
    backtest = build_backtest(pred, garch)
    metrics = compute_metrics(backtest)

    out_bt = backtest.copy()
    out_bt["date"] = pd.to_datetime(out_bt["date"]).dt.strftime("%Y-%m-%d")
    out_bt.to_csv(
        os.path.join(os.getcwd(), "outputs", "oos_backtest.csv"),
        index=False,
    )
    metrics.to_csv(
        os.path.join(os.getcwd(), "outputs", "oos_backtest_metrics.csv"),
        index=False,
    )
    logging.info("Saved OOS backtest and metrics")


if __name__ == "__main__":
    main()
