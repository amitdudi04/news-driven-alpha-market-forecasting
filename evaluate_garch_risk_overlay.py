"""Run frozen GARCH risk overlay and fixed-rule OOS simulation."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


SPEC_PATH = Path("config/garch_simulation_spec.json")
OUTPUT_DIR = Path("outputs/garch_simulation_v1")


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def load_market(spec: dict) -> pd.DataFrame:
    market = pd.read_csv(spec["market_history"])
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    market = market.sort_values("date").reset_index(drop=True)
    if market["date"].duplicated().any():
        raise ValueError("Duplicate market dates.")
    required = {"date", "return", "close"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"Missing market columns: {sorted(missing)}")
    return market


def fit_one_garch(train_returns: pd.Series) -> tuple[float, int]:
    returns_pct = train_returns.dropna().astype(float) * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            returns_pct,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        result = model.fit(
            disp="off",
            show_warning=False,
            options={"maxiter": 500},
        )
        variance = float(
            result.forecast(
                horizon=1,
                reindex=False,
            ).variance.values[-1, 0]
        )
    forecast_vol = math.sqrt(max(variance, 0.0)) / 100.0
    return forecast_vol, int(result.convergence_flag)


def generate_garch_forecasts(
    market: pd.DataFrame,
    spec: dict,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / "garch_forecasts.csv"

    gspec = spec["garch"]
    min_train = int(gspec["min_train_observations"])
    start_date = pd.Timestamp(gspec["forecast_start"])
    min_reference = int(gspec["risk_reference_min_periods"])

    if checkpoint.exists():
        existing = pd.read_csv(
            checkpoint,
            parse_dates=["date", "target_session_date"],
        )
        existing_dates = set(existing["date"])
        rows = existing.to_dict(orient="records")
        print(
            f"Resume GARCH checkpoint: {len(existing)} forecasts",
            flush=True,
        )
    else:
        existing_dates = set()
        rows = []

    returns = pd.to_numeric(
        market["return"],
        errors="coerce",
    )

    eligible_positions = [
        i
        for i in range(len(market) - 1)
        if market.loc[i, "date"] >= start_date
        and int(returns.iloc[: i + 1].notna().sum()) >= min_train
    ]

    completed_since_save = 0
    for counter, i in enumerate(eligible_positions, start=1):
        date = market.loc[i, "date"]
        if date in existing_dates:
            continue

        pred_vol, convergence_flag = fit_one_garch(
            returns.iloc[: i + 1]
        )
        next_date = market.loc[i + 1, "date"]
        actual_return = float(market.loc[i + 1, "return"])
        pred_var = pred_vol ** 2
        realized_sq = actual_return ** 2

        prior_forecasts = [
            float(row["pred_vol_t_plus_1"])
            for row in rows
            if pd.Timestamp(row["date"]) < date
            and np.isfinite(float(row["pred_vol_t_plus_1"]))
        ]
        if len(prior_forecasts) >= min_reference:
            risk_reference = float(np.median(prior_forecasts))
            risk_scale = float(
                np.clip(
                    risk_reference / pred_vol
                    if pred_vol > 0
                    else 1.0,
                    0.0,
                    1.0,
                )
            )
        else:
            risk_reference = np.nan
            risk_scale = np.nan

        qlike = (
            float(np.log(pred_var) + realized_sq / pred_var)
            if pred_var > 0
            else np.nan
        )
        variance_sq_error = float(
            (pred_var - realized_sq) ** 2
        )
        volatility_sq_error = float(
            (pred_vol - abs(actual_return)) ** 2
        )

        rows.append(
            {
                "date": date,
                "target_session_date": next_date,
                "pred_vol_t_plus_1": pred_vol,
                "pred_variance_t_plus_1": pred_var,
                "risk_reference_prior_median": risk_reference,
                "risk_scale": risk_scale,
                "actual_return_t_plus_1": actual_return,
                "realized_squared_return_t_plus_1": realized_sq,
                "qlike": qlike,
                "variance_squared_error": variance_sq_error,
                "volatility_squared_error": volatility_sq_error,
                "convergence_flag": convergence_flag,
                "train_observations": int(
                    returns.iloc[: i + 1].notna().sum()
                ),
            }
        )
        completed_since_save += 1

        if completed_since_save >= 25:
            save_garch_checkpoint(rows, checkpoint)
            completed_since_save = 0
            print(
                f"GARCH checkpoint: {counter}/{len(eligible_positions)} "
                f"through {date.date()}",
                flush=True,
            )

    result = save_garch_checkpoint(rows, checkpoint)

    if result["date"].duplicated().any():
        raise ValueError("Duplicate GARCH forecast dates.")
    if (result["pred_vol_t_plus_1"] <= 0).any():
        raise ValueError("Non-positive GARCH volatility forecast.")
    return result


def save_garch_checkpoint(rows, path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_session_date"] = pd.to_datetime(
        frame["target_session_date"]
    )
    frame = (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    save = frame.copy()
    save["date"] = save["date"].dt.strftime("%Y-%m-%d")
    save["target_session_date"] = (
        save["target_session_date"].dt.strftime("%Y-%m-%d")
    )
    save.to_csv(path, index=False)
    return frame


def garch_metrics(
    forecasts: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    unique_period_dates = (
        predictions[
            ["period", "date"]
        ]
        .drop_duplicates()
        .copy()
    )
    unique_period_dates["date"] = pd.to_datetime(
        unique_period_dates["date"]
    )

    rows = []
    for period, period_dates in unique_period_dates.groupby("period"):
        subset = forecasts.merge(
            period_dates[["date"]],
            on="date",
            how="inner",
            validate="one_to_one",
        )
        if subset.empty:
            continue
        rows.append(
            {
                "period": period,
                "n": int(len(subset)),
                "mean_predicted_volatility": float(
                    subset["pred_vol_t_plus_1"].mean()
                ),
                "mean_absolute_next_return": float(
                    subset["actual_return_t_plus_1"].abs().mean()
                ),
                "qlike_mean": float(subset["qlike"].mean()),
                "variance_mse": float(
                    subset["variance_squared_error"].mean()
                ),
                "volatility_rmse_vs_absolute_return": float(
                    np.sqrt(
                        subset["volatility_squared_error"].mean()
                    )
                ),
                "mean_risk_scale": float(
                    subset["risk_scale"].mean()
                ),
                "median_risk_scale": float(
                    subset["risk_scale"].median()
                ),
                "convergence_failures": int(
                    (subset["convergence_flag"] != 0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    return float(((wealth - peak) / peak).min())


def sharpe_zero_rf(series: pd.Series) -> float:
    std = float(series.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return np.nan
    return float(
        series.mean() / std * np.sqrt(252.0)
    )


def simulate_position_path(
    frame: pd.DataFrame,
    position_col: str,
    cost_rate: float,
) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    position = out[position_col].astype(float)

    previous = position.shift(1)
    previous.iloc[0] = 0.0
    out["turnover"] = (position - previous).abs()
    out["transaction_cost"] = out["turnover"] * cost_rate

    out["gross_strategy_return"] = (
        position * out["benchmark_return"]
    )
    out["net_strategy_return"] = (
        out["gross_strategy_return"]
        - out["transaction_cost"]
    )
    if (out["net_strategy_return"] <= -1.0).any():
        raise ValueError("Net strategy return <= -100%.")

    out["gross_wealth"] = (
        1.0 + out["gross_strategy_return"]
    ).cumprod()
    out["net_wealth"] = (
        1.0 + out["net_strategy_return"]
    ).cumprod()
    out["benchmark_wealth"] = (
        1.0 + out["benchmark_return"]
    ).cumprod()

    active = out[out["signal"] != 0]
    hit = (
        float(
            (
                np.sign(active["signal"])
                == np.sign(active["benchmark_return"])
            ).mean()
        )
        if len(active)
        else np.nan
    )

    gross_total = float(out["gross_wealth"].iloc[-1] - 1.0)
    net_total = float(out["net_wealth"].iloc[-1] - 1.0)
    benchmark_total = float(
        out["benchmark_wealth"].iloc[-1] - 1.0
    )

    metrics = {
        "observations": int(len(out)),
        "active_observations": int(len(active)),
        "long_observations": int((out["signal"] == 1).sum()),
        "short_observations": int((out["signal"] == -1).sum()),
        "flat_observations": int((out["signal"] == 0).sum()),
        "gross_total_return": gross_total,
        "net_total_return": net_total,
        "benchmark_total_return": benchmark_total,
        "active_total_return": net_total - benchmark_total,
        "transaction_cost_compounding_drag": (
            gross_total - net_total
        ),
        "net_sharpe_zero_rf": sharpe_zero_rf(
            out["net_strategy_return"]
        ),
        "benchmark_sharpe_zero_rf": sharpe_zero_rf(
            out["benchmark_return"]
        ),
        "net_max_drawdown": max_drawdown(out["net_wealth"]),
        "benchmark_max_drawdown": max_drawdown(
            out["benchmark_wealth"]
        ),
        "directional_hit_rate_active": hit,
        "average_turnover": float(out["turnover"].mean()),
        "total_transaction_cost_rate": float(
            out["transaction_cost"].sum()
        ),
        "average_abs_position": float(position.abs().mean()),
    }
    return out, metrics


def build_simulations(
    predictions: pd.DataFrame,
    forecasts: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_spec = spec["directional_signal"]
    cost_rate = float(
        spec["transaction_cost"]["rate_per_unit_turnover"]
    )
    long_threshold = float(
        signal_spec["long_probability_threshold"]
    )
    short_threshold = float(
        signal_spec["short_probability_threshold"]
    )

    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="raise",
    )
    predictions["target_session_date"] = pd.to_datetime(
        predictions["target_session_date"],
        errors="raise",
    )

    simulations = []
    metric_rows = []

    for model_name in spec["simulation"]["models"]:
        for period in spec["simulation"]["periods"]:
            pred = predictions[
                (predictions["model_name"] == model_name)
                & (predictions["period"] == period)
            ].copy()
            if pred.empty:
                raise ValueError(
                    f"Missing predictions for {model_name} {period}."
                )

            merged = pred.merge(
                forecasts[
                    [
                        "date",
                        "target_session_date",
                        "pred_vol_t_plus_1",
                        "risk_reference_prior_median",
                        "risk_scale",
                        "actual_return_t_plus_1",
                    ]
                ],
                on="date",
                how="left",
                suffixes=("_pred", "_garch"),
                validate="one_to_one",
            )
            if merged["pred_vol_t_plus_1"].isna().any():
                raise ValueError(
                    f"Missing GARCH forecasts for {model_name} {period}."
                )

            target_date_match = (
                merged["target_session_date_pred"]
                == merged["target_session_date_garch"]
            )
            if not target_date_match.all():
                raise ValueError(
                    f"Target-date mismatch for {model_name} {period}."
                )

            return_error = (
                merged["target_return_t_plus_1"]
                - merged["actual_return_t_plus_1"]
            ).abs().max()
            if float(return_error) > 1e-12:
                raise ValueError(
                    f"Realized-return mismatch for {model_name} {period}: "
                    f"{return_error}"
                )

            p = merged["probability"].astype(float)
            merged["signal"] = np.select(
                [
                    p >= long_threshold,
                    p <= short_threshold,
                ],
                [1.0, -1.0],
                default=0.0,
            )
            merged["position_unscaled"] = merged["signal"]
            merged["position_garch_scaled"] = (
                merged["signal"]
                * merged["risk_scale"].fillna(1.0)
            )
            merged["benchmark_return"] = np.expm1(
                merged["actual_return_t_plus_1"]
            )
            merged = merged.sort_values("date").reset_index(
                drop=True
            )

            for overlay, position_col in [
                ("unscaled", "position_unscaled"),
                ("garch_scaled", "position_garch_scaled"),
            ]:
                path, metrics = simulate_position_path(
                    merged,
                    position_col,
                    cost_rate,
                )
                path["model_name"] = model_name
                path["period"] = period
                path["overlay"] = overlay
                simulations.append(path)

                metric_rows.append(
                    {
                        "model_name": model_name,
                        "family": (
                            "logistic"
                            if model_name.startswith("logistic")
                            else "xgboost"
                        ),
                        "variant": (
                            "market_plus_sentiment"
                            if "plus_sentiment" in model_name
                            else "market_only"
                        ),
                        "period": period,
                        "overlay": overlay,
                        **metrics,
                    }
                )

    simulation = pd.concat(
        simulations,
        ignore_index=True,
    )
    metrics = pd.DataFrame(metric_rows)

    pair_rows = []
    for family in ["logistic", "xgboost"]:
        for period in spec["simulation"]["periods"]:
            for overlay in ["unscaled", "garch_scaled"]:
                pair = metrics[
                    (metrics["family"] == family)
                    & (metrics["period"] == period)
                    & (metrics["overlay"] == overlay)
                ].set_index("variant")
                market = pair.loc["market_only"]
                sentiment = pair.loc[
                    "market_plus_sentiment"
                ]
                pair_rows.append(
                    {
                        "family": family,
                        "period": period,
                        "overlay": overlay,
                        "sentiment_minus_market_net_total_return": float(
                            sentiment["net_total_return"]
                            - market["net_total_return"]
                        ),
                        "sentiment_minus_market_sharpe": float(
                            sentiment["net_sharpe_zero_rf"]
                            - market["net_sharpe_zero_rf"]
                        ),
                        "sentiment_minus_market_max_drawdown": float(
                            sentiment["net_max_drawdown"]
                            - market["net_max_drawdown"]
                        ),
                        "sentiment_minus_market_turnover": float(
                            sentiment["average_turnover"]
                            - market["average_turnover"]
                        ),
                    }
                )
    paired = pd.DataFrame(pair_rows)
    return simulation, metrics, paired


def save_outputs(
    forecasts: pd.DataFrame,
    garch_eval: pd.DataFrame,
    simulation: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    spec: dict,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_forecasts = forecasts.copy()
    save_forecasts["date"] = (
        save_forecasts["date"].dt.strftime("%Y-%m-%d")
    )
    save_forecasts["target_session_date"] = (
        save_forecasts["target_session_date"].dt.strftime(
            "%Y-%m-%d"
        )
    )
    save_forecasts.to_csv(
        OUTPUT_DIR / "garch_forecasts.csv",
        index=False,
    )
    garch_eval.to_csv(
        OUTPUT_DIR / "garch_forecast_metrics.csv",
        index=False,
    )
    simulation.to_csv(
        OUTPUT_DIR / "simulation_paths.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(
        OUTPUT_DIR / "simulation_metrics.csv",
        index=False,
    )
    paired.to_csv(
        OUTPUT_DIR / "paired_sentiment_simulation_comparison.csv",
        index=False,
    )

    summary = {
        "analysis_id": spec["analysis_id"],
        "garch_spec": spec["garch"],
        "signal_spec": spec["directional_signal"],
        "transaction_cost": spec["transaction_cost"],
        "garch_forecast_metrics": garch_eval.to_dict(
            orient="records"
        ),
        "simulation_metrics": metrics.to_dict(
            orient="records"
        ),
        "paired_sentiment_simulation_comparison": (
            paired.to_dict(orient="records")
        ),
        "guardrails": spec["interpretation_guardrails"],
    }
    (
        OUTPUT_DIR / "results_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main():
    spec = load_spec()
    market = load_market(spec)
    predictions = pd.read_csv(
        spec["directional_predictions"]
    )

    forecasts = generate_garch_forecasts(market, spec)

    # Require every saved OOS directional prediction date to have a
    # successful GARCH forecast and matching next-session return.
    prediction_dates = set(
        pd.to_datetime(predictions["date"]).unique()
    )
    forecast_dates = set(forecasts["date"].unique())
    missing = prediction_dates - forecast_dates
    if missing:
        raise ValueError(
            f"Missing GARCH coverage for {len(missing)} OOS dates."
        )

    garch_eval = garch_metrics(
        forecasts,
        predictions,
    )
    simulation, metrics, paired = build_simulations(
        predictions,
        forecasts,
        spec,
    )
    save_outputs(
        forecasts,
        garch_eval,
        simulation,
        metrics,
        paired,
        spec,
    )

    print("\nGARCH FORECAST METRICS")
    print(garch_eval.to_string(index=False))
    print("\nSIMULATION METRICS")
    print(
        metrics[
            [
                "model_name",
                "period",
                "overlay",
                "observations",
                "active_observations",
                "net_total_return",
                "benchmark_total_return",
                "active_total_return",
                "net_sharpe_zero_rf",
                "net_max_drawdown",
                "average_turnover",
                "transaction_cost_compounding_drag",
            ]
        ].to_string(index=False)
    )
    print("\nPAIRED SENTIMENT SIMULATION COMPARISON")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
