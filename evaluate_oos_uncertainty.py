"""Evaluate frozen OOS predictions with calibration and paired block bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


SPEC_PATH = Path("config/oos_uncertainty_spec.json")
OUTPUT_DIR = Path("outputs/directional_experiment_v1/uncertainty")


def load_inputs():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    pred = pd.read_csv(spec["source_predictions"])
    pred["target_session_date"] = pd.to_datetime(
        pred["target_session_date"],
        errors="raise",
    )
    return spec, pred


def calibration_intercept_slope(
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[float, float]:
    eps = 1e-6
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    logit_p = np.log(p / (1.0 - p))
    x = sm.add_constant(logit_p, has_constant="add")
    try:
        fit = sm.GLM(
            y,
            x,
            family=sm.families.Binomial(),
        ).fit()
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def reliability_bins(
    y: np.ndarray,
    p: np.ndarray,
    bins: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=int),
            "p": np.asarray(p, dtype=float),
        }
    ).sort_values("p").reset_index(drop=True)

    pieces = np.array_split(np.arange(len(frame)), bins)
    rows = []
    for index, positions in enumerate(pieces, start=1):
        if len(positions) == 0:
            continue
        part = frame.iloc[positions]
        rows.append(
            {
                "bin": index,
                "n": int(len(part)),
                "probability_min": float(part["p"].min()),
                "probability_max": float(part["p"].max()),
                "mean_predicted_probability": float(
                    part["p"].mean()
                ),
                "observed_up_rate": float(part["y"].mean()),
                "absolute_calibration_gap": float(
                    abs(part["p"].mean() - part["y"].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    bins: int,
) -> float:
    table = reliability_bins(y, p, bins)
    weights = table["n"] / table["n"].sum()
    return float(
        (weights * table["absolute_calibration_gap"]).sum()
    )


def point_metrics(
    y: np.ndarray,
    p: np.ndarray,
    bins: int,
) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)

    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "brier_score_loss": float(
            brier_score_loss(y, p)
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "roc_auc": float(roc_auc_score(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": (
            expected_calibration_error(y, p, bins)
        ),
        "mean_predicted_up_probability": float(p.mean()),
        "observed_up_rate": float(y.mean()),
    }


def pair_frame(
    predictions: pd.DataFrame,
    family: str,
    period: str,
) -> pd.DataFrame:
    columns = [
        "target_session_date",
        "target_direction_t_plus_1",
        "probability",
    ]
    market = predictions[
        (predictions["family"] == family)
        & (predictions["variant"] == "market_only")
        & (predictions["period"] == period)
    ][columns].copy()
    sentiment = predictions[
        (predictions["family"] == family)
        & (predictions["variant"] == "market_plus_sentiment")
        & (predictions["period"] == period)
    ][columns].copy()

    market = market.rename(
        columns={
            "target_direction_t_plus_1": "y_market",
            "probability": "p_market",
        }
    )
    sentiment = sentiment.rename(
        columns={
            "target_direction_t_plus_1": "y_sentiment",
            "probability": "p_sentiment",
        }
    )
    paired = market.merge(
        sentiment,
        on="target_session_date",
        how="inner",
        validate="one_to_one",
    ).sort_values("target_session_date").reset_index(drop=True)

    if len(paired) != len(market) or len(paired) != len(sentiment):
        raise ValueError(
            f"Paired row count mismatch for {family} {period}."
        )
    if not (
        paired["y_market"].astype(int)
        == paired["y_sentiment"].astype(int)
    ).all():
        raise ValueError(
            f"Paired targets differ for {family} {period}."
        )
    paired["y"] = paired["y_market"].astype(int)
    return paired[
        [
            "target_session_date",
            "y",
            "p_market",
            "p_sentiment",
        ]
    ]


def _fast_balanced_accuracy(
    y: np.ndarray,
    p: np.ndarray,
) -> float:
    pred = p >= 0.5
    pos = y == 1
    neg = ~pos
    if not pos.any() or not neg.any():
        return np.nan
    sensitivity = np.mean(pred[pos])
    specificity = np.mean(~pred[neg])
    return float((sensitivity + specificity) / 2.0)


def _fast_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _fast_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    )


def _fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(p, method="average")
    rank_sum_pos = float(ranks[pos].sum())
    return float(
        (
            rank_sum_pos
            - n_pos * (n_pos + 1) / 2.0
        )
        / (n_pos * n_neg)
    )


def _fast_ece(
    y: np.ndarray,
    p: np.ndarray,
    bins: int,
) -> float:
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, bins)
    total = float(len(y))
    ece = 0.0
    for idx in groups:
        if len(idx) == 0:
            continue
        ece += (
            len(idx)
            / total
            * abs(float(p[idx].mean() - y[idx].mean()))
        )
    return float(ece)


def delta_metrics(
    y: np.ndarray,
    p_market: np.ndarray,
    p_sentiment: np.ndarray,
    bins: int,
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pm = np.asarray(p_market, dtype=float)
    ps = np.asarray(p_sentiment, dtype=float)

    return {
        "balanced_accuracy_improvement": (
            _fast_balanced_accuracy(y, ps)
            - _fast_balanced_accuracy(y, pm)
        ),
        "brier_improvement": (
            _fast_brier(y, pm)
            - _fast_brier(y, ps)
        ),
        "log_loss_improvement": (
            _fast_log_loss(y, pm)
            - _fast_log_loss(y, ps)
        ),
        "calibration_ece_improvement": (
            _fast_ece(y, pm, bins)
            - _fast_ece(y, ps, bins)
        ),
        "roc_auc_improvement": (
            _fast_auc(y, ps)
            - _fast_auc(y, pm)
        ),
    }


def circular_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    length = min(block_length, n)
    blocks_needed = int(np.ceil(n / length))
    starts = rng.integers(0, n, size=blocks_needed)
    offsets = np.arange(length)
    indices = np.concatenate(
        [((start + offsets) % n) for start in starts]
    )
    return indices[:n]


def bootstrap_pair(
    paired: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap = spec["bootstrap"]
    reps = int(bootstrap["replications"])
    block_length = int(bootstrap["block_length_sessions"])
    seed = int(bootstrap["random_seed"])
    bins = int(spec["calibration"]["equal_frequency_bins"])

    y = paired["y"].to_numpy(dtype=int)
    pm = paired["p_market"].to_numpy(dtype=float)
    ps = paired["p_sentiment"].to_numpy(dtype=float)

    point = delta_metrics(y, pm, ps, bins)
    rng = np.random.default_rng(seed)

    metric_names = list(point.keys())
    values = {
        name: np.full(reps, np.nan, dtype=float)
        for name in metric_names
    }

    for rep in range(reps):
        idx = circular_block_indices(
            len(paired),
            block_length,
            rng,
        )
        sample = delta_metrics(
            y[idx],
            pm[idx],
            ps[idx],
            bins,
        )
        for name in metric_names:
            values[name][rep] = sample[name]

    alpha = 1.0 - float(bootstrap["confidence_interval"])
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    summary_rows = []
    draw_rows = []
    for name in metric_names:
        vals = values[name]
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            lower = upper = prob = np.nan
        else:
            lower, upper = np.percentile(
                finite,
                [lower_q, upper_q],
            )
            prob = float(np.mean(finite > 0.0))

        if np.isfinite(lower) and lower > 0:
            resolution = "sentiment_better"
        elif np.isfinite(upper) and upper < 0:
            resolution = "market_only_better"
        else:
            resolution = "unresolved"

        summary_rows.append(
            {
                "metric": name,
                "point_increment": float(point[name]),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "bootstrap_probability_increment_gt_zero": prob,
                "resolution_95pct": resolution,
                "valid_bootstrap_draws": int(len(finite)),
            }
        )
        draw_rows.extend(
            {
                "replication": int(i),
                "metric": name,
                "increment": float(v),
            }
            for i, v in enumerate(vals)
            if np.isfinite(v)
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def main():
    spec, predictions = load_inputs()
    bins = int(spec["calibration"]["equal_frequency_bins"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_rows = []
    reliability_parts = []
    bootstrap_summaries = []
    bootstrap_draw_parts = []

    for family in spec["families"]:
        for period in spec["periods"]:
            paired = pair_frame(predictions, family, period)

            for variant, probability_col in [
                ("market_only", "p_market"),
                ("market_plus_sentiment", "p_sentiment"),
            ]:
                metrics = point_metrics(
                    paired["y"].to_numpy(),
                    paired[probability_col].to_numpy(),
                    bins,
                )
                calibration_rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "period": period,
                        **metrics,
                    }
                )

                reliability = reliability_bins(
                    paired["y"].to_numpy(),
                    paired[probability_col].to_numpy(),
                    bins,
                )
                reliability.insert(0, "period", period)
                reliability.insert(0, "variant", variant)
                reliability.insert(0, "family", family)
                reliability_parts.append(reliability)

            summary, draws = bootstrap_pair(paired, spec)
            summary.insert(0, "period", period)
            summary.insert(0, "family", family)
            bootstrap_summaries.append(summary)

            draws.insert(0, "period", period)
            draws.insert(0, "family", family)
            bootstrap_draw_parts.append(draws)

            print(
                f"Completed {family} {period}: n={len(paired)}",
                flush=True,
            )

    calibration = pd.DataFrame(calibration_rows)
    reliability = pd.concat(
        reliability_parts,
        ignore_index=True,
    )
    bootstrap_summary = pd.concat(
        bootstrap_summaries,
        ignore_index=True,
    )
    bootstrap_draws = pd.concat(
        bootstrap_draw_parts,
        ignore_index=True,
    )

    calibration.to_csv(
        OUTPUT_DIR / "oos_calibration_metrics.csv",
        index=False,
    )
    reliability.to_csv(
        OUTPUT_DIR / "oos_reliability_bins.csv",
        index=False,
    )
    bootstrap_summary.to_csv(
        OUTPUT_DIR / "paired_block_bootstrap_summary.csv",
        index=False,
    )
    bootstrap_draws.to_csv(
        OUTPUT_DIR / "paired_block_bootstrap_draws.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = {
        "analysis_id": spec["analysis_id"],
        "bootstrap": spec["bootstrap"],
        "calibration": spec["calibration"],
        "calibration_metrics": calibration.to_dict(
            orient="records"
        ),
        "paired_increment_uncertainty": (
            bootstrap_summary.to_dict(orient="records")
        ),
    }
    (
        OUTPUT_DIR / "uncertainty_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nCALIBRATION")
    print(
        calibration[
            [
                "family",
                "variant",
                "period",
                "n",
                "calibration_intercept",
                "calibration_slope",
                "expected_calibration_error",
            ]
        ].to_string(index=False)
    )
    print("\nPAIRED BLOCK-BOOTSTRAP UNCERTAINTY")
    print(bootstrap_summary.to_string(index=False))


if __name__ == "__main__":
    main()
