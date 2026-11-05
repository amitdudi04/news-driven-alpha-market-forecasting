"""Independent arithmetic checks of the compact public empirical result release.

Does not refit ML/GARCH models or reproduce bootstrap draws from private inputs.
Run from the repository root: python scripts/verify_published_results.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    log_loss, roc_auc_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ATOL = 3e-10


def close(label: str, actual: float, published: float) -> None:
    if not np.isclose(actual, published, rtol=0, atol=ATOL):
        raise AssertionError(f"{label}: recomputed={actual}, released={published}")


def check_directional() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(RESULTS / "oos_predictions.csv")
    metrics = pd.read_csv(RESULTS / "directional_oos_metrics.csv")
    increments = pd.read_csv(RESULTS / "incremental_sentiment_comparison.csv")
    assert len(predictions) == 2584 and len(metrics) == 12 and len(increments) == 6
    for row in metrics.itertuples(index=False):
        selected = predictions.loc[
            predictions.model_name.eq(row.model_name) & predictions.period.eq(row.period)
        ].sort_values("target_session_date")
        assert len(selected) == row.n
        assert not selected.target_session_date.duplicated().any()
        y = selected.target_direction_t_plus_1.astype(int).to_numpy()
        p = selected.probability.to_numpy(dtype=float)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        direction = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, direction, labels=[0, 1]).ravel()
        for key, value in {
            "accuracy": accuracy_score(y, direction),
            "balanced_accuracy": balanced_accuracy_score(y, direction),
            "brier_score_loss": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
            "roc_auc": roc_auc_score(y, p),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }.items():
            close(f"{row.model_name}/{row.period}/{key}", float(value), float(getattr(row, key)))
    for row in increments.itertuples(index=False):
        two = metrics.loc[metrics.family.eq(row.family) & metrics.period.eq(row.period)]
        a = two.loc[two.variant.eq("market_only")].iloc[0]
        b = two.loc[two.variant.eq("market_plus_sentiment")].iloc[0]
        for name, value in {
            "balanced_accuracy_improvement": b.balanced_accuracy - a.balanced_accuracy,
            "accuracy_improvement": b.accuracy - a.accuracy,
            "roc_auc_improvement": b.roc_auc - a.roc_auc,
            "brier_improvement": a.brier_score_loss - b.brier_score_loss,
            "log_loss_improvement": a.log_loss - b.log_loss,
        }.items():
            close(f"{row.family}/{row.period}/{name}", value, float(getattr(row, name)))
    return predictions, metrics


def check_bootstrap(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    summary = pd.read_csv(RESULTS / "block_bootstrap_summary.csv")
    assert len(summary) == 30 and summary.valid_bootstrap_draws.eq(5000).all()
    for row in summary.itertuples(index=False):
        subset = metrics.loc[metrics.family.eq(row.family) & metrics.period.eq(row.period)]
        a = subset.loc[subset.variant.eq("market_only")].iloc[0]
        b = subset.loc[subset.variant.eq("market_plus_sentiment")].iloc[0]
        field = row.metric
        if field in ("balanced_accuracy_improvement", "roc_auc_improvement"):
            col = field.removesuffix("_improvement")
            point = float(b[col] - a[col])
        elif field in ("brier_improvement", "log_loss_improvement"):
            col = {"brier_improvement": "brier_score_loss",
                   "log_loss_improvement": "log_loss"}[field]
            point = float(a[col] - b[col])
        else:
            # Calibration ECE point is obtained from the separate calibration file.
            assert field == "calibration_ece_improvement"
            cal = pd.read_csv(RESULTS / "calibration_metrics.csv")
            pair = cal.loc[cal.family.eq(row.family) & cal.period.eq(row.period)]
            point = float(pair.loc[pair.variant.eq("market_only"), "expected_calibration_error"].iloc[0]
                          - pair.loc[pair.variant.eq("market_plus_sentiment"), "expected_calibration_error"].iloc[0])
        close(f"bootstrap point {row.family}/{row.period}/{field}", point, row.point_increment)
        result = ("sentiment_better" if row.ci_lower > 0 else
                  "market_only_better" if row.ci_upper < 0 else "unresolved")
        assert row.resolution_95pct == result
        assert row.ci_lower <= row.ci_upper
    # The untracked 5,000-replicate bootstrap draws cannot be independently regenerated here.


def check_garch(predictions: pd.DataFrame) -> pd.DataFrame:
    garch = pd.read_csv(RESULTS / "garch_forecasts.csv")
    by_period = pd.read_csv(RESULTS / "garch_forecast_metrics.csv")
    assert len(garch) == 907 and not garch.date.duplicated().any()
    assert garch.convergence_flag.eq(0).all()
    for row in garch.itertuples(index=False):
        close("GARCH variance", row.pred_vol_t_plus_1**2, row.pred_variance_t_plus_1)
        close("GARCH realized variance", row.actual_return_t_plus_1**2,
              row.realized_squared_return_t_plus_1)
        close("GARCH QLIKE", np.log(row.pred_variance_t_plus_1)
              + row.realized_squared_return_t_plus_1 / row.pred_variance_t_plus_1, row.qlike)
        close("GARCH squared error", (row.pred_variance_t_plus_1
              - row.realized_squared_return_t_plus_1)**2, row.variance_squared_error)
        close("GARCH volatility squared error",
              (row.pred_vol_t_plus_1 - abs(row.actual_return_t_plus_1))**2,
              row.volatility_squared_error)
    for row in by_period.itertuples(index=False):
        dates = set(predictions.loc[
            predictions.model_name.eq("logistic_market_only") & predictions.period.eq(row.period),
            "date"])
        subset = garch.loc[garch.date.isin(dates)]
        assert len(subset) == row.n
        for key, value in {
            "mean_predicted_volatility": subset.pred_vol_t_plus_1.mean(),
            "mean_absolute_next_return": subset.actual_return_t_plus_1.abs().mean(),
            "qlike_mean": subset.qlike.mean(),
            "variance_mse": subset.variance_squared_error.mean(),
            "volatility_rmse_vs_absolute_return": np.sqrt(subset.volatility_squared_error.mean()),
            "mean_risk_scale": subset.risk_scale.mean(),
            "median_risk_scale": subset.risk_scale.median(),
            "convergence_failures": int(subset.convergence_flag.ne(0).sum()),
        }.items():
            close(f"GARCH/{row.period}/{key}", float(value), float(getattr(row, key)))
    return garch


def calc_drawdown(returns: np.ndarray) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def check_simulations(predictions: pd.DataFrame, garch: pd.DataFrame) -> None:
    results = pd.read_csv(RESULTS / "simulation_metrics.csv")
    paired = pd.read_csv(RESULTS / "paired_simulation_comparison.csv")
    assert len(results) == 24 and len(paired) == 12
    for row in results.itertuples(index=False):
        x = predictions.loc[
            predictions.model_name.eq(row.model_name) & predictions.period.eq(row.period)
        ].sort_values("date")
        joined = x.merge(garch[["date", "target_session_date", "risk_scale",
                                "actual_return_t_plus_1"]],
                         on="date", validate="one_to_one", suffixes=("_pred", "_garch"))
        assert len(joined) == len(x) == row.observations
        assert joined.target_session_date_pred.eq(joined.target_session_date_garch).all()
        if joined.target_return_t_plus_1.notna().any():
            known = joined.target_return_t_plus_1.notna()
            assert np.allclose(joined.loc[known, "target_return_t_plus_1"],
                               joined.loc[known, "actual_return_t_plus_1"], rtol=0, atol=1e-12)
        p = joined.probability.to_numpy()
        signal = np.where(p >= 0.55, 1., np.where(p <= 0.45, -1., 0.))
        scale = (joined.risk_scale.fillna(1).to_numpy()
                 if row.overlay == "garch_scaled" else np.ones(len(joined)))
        position = signal * scale
        turnover = np.abs(position - np.r_[0., position[:-1]])
        benchmark = np.expm1(joined.actual_return_t_plus_1.to_numpy())
        gross = position * benchmark
        net = gross - 0.001 * turnover
        gross_total = float(np.prod(1. + gross) - 1.)
        net_total = float(np.prod(1. + net) - 1.)
        benchmark_total = float(np.prod(1. + benchmark) - 1.)
        net_sd = float(np.std(net, ddof=1))
        bench_sd = float(np.std(benchmark, ddof=1))
        active = signal != 0.
        values = {
            "observations": len(x), "active_observations": int(active.sum()),
            "long_observations": int((signal == 1).sum()),
            "short_observations": int((signal == -1).sum()),
            "flat_observations": int((signal == 0).sum()),
            "gross_total_return": gross_total,
            "net_total_return": net_total,
            "benchmark_total_return": benchmark_total,
            "active_total_return": net_total - benchmark_total,
            "transaction_cost_compounding_drag": gross_total - net_total,
            "net_sharpe_zero_rf": float(np.mean(net) / net_sd * np.sqrt(252)),
            "benchmark_sharpe_zero_rf": float(np.mean(benchmark) / bench_sd * np.sqrt(252)),
            "net_max_drawdown": calc_drawdown(net),
            "benchmark_max_drawdown": calc_drawdown(benchmark),
            "directional_hit_rate_active": float(np.mean(
                np.sign(signal[active]) == np.sign(benchmark[active]))),
            "average_turnover": float(turnover.mean()),
            "total_transaction_cost_rate": float(0.001 * turnover.sum()),
            "average_abs_position": float(np.abs(position).mean()),
        }
        for key, value in values.items():
            close(f"simulation/{row.model_name}/{row.period}/{row.overlay}/{key}",
                  value, float(getattr(row, key)))
    for row in paired.itertuples(index=False):
        subset = results.loc[
            results.family.eq(row.family) & results.period.eq(row.period)
            & results.overlay.eq(row.overlay)]
        market = subset.loc[subset.variant.eq("market_only")].iloc[0]
        sentiment = subset.loc[subset.variant.eq("market_plus_sentiment")].iloc[0]
        for field, metric in {
            "sentiment_minus_market_net_total_return": "net_total_return",
            "sentiment_minus_market_sharpe": "net_sharpe_zero_rf",
            "sentiment_minus_market_max_drawdown": "net_max_drawdown",
            "sentiment_minus_market_turnover": "average_turnover",
        }.items():
            close(f"paired/{row.family}/{row.period}/{row.overlay}/{field}",
                  float(sentiment[metric] - market[metric]), float(getattr(row, field)))


def main() -> None:
    predictions, metrics = check_directional()
    check_bootstrap(predictions, metrics)
    garch = check_garch(predictions)
    check_simulations(predictions, garch)
    print("PASS: 2,584 OOS predictions; 12 directional rows; 6 increments; "
          "30 bootstrap summaries (stored points/interval labels); "
          "907 GARCH rows; 3 GARCH periods; 24 simulated strategies; "
          "12 paired strategy comparisons.")


if __name__ == "__main__":
    main()
