import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="News-Driven Alpha — CSI 300",
    page_icon="📈",
    layout="wide",
)


def load_csv(path, parse_dates=None):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value, digits=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def num(value, digits=3):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


alignment = load_json("data/timestamp_alignment_2023_2026_summary.json")
master_summary = load_json(
    "data/master_session_dataset_2023_2026_summary.json"
)
master = load_csv(
    "data/master_session_dataset_2023_2026.csv",
    parse_dates=["date", "target_session_date"],
)

metrics = load_csv(
    "outputs/directional_experiment_v1/metrics.csv"
)
incremental = load_csv(
    "outputs/directional_experiment_v1/"
    "incremental_sentiment_comparison.csv"
)
predictions = load_csv(
    "outputs/directional_experiment_v1/predictions.csv",
    parse_dates=["date", "target_session_date"],
)
bootstrap = load_csv(
    "outputs/directional_experiment_v1/uncertainty/"
    "paired_block_bootstrap_summary.csv"
)
calibration = load_csv(
    "outputs/directional_experiment_v1/uncertainty/"
    "oos_calibration_metrics.csv"
)
reliability = load_csv(
    "outputs/directional_experiment_v1/uncertainty/"
    "oos_reliability_bins.csv"
)

garch_metrics = load_csv(
    "outputs/garch_simulation_v1/garch_forecast_metrics.csv"
)
simulation_metrics = load_csv(
    "outputs/garch_simulation_v1/simulation_metrics.csv"
)
simulation_paths = load_csv(
    "outputs/garch_simulation_v1/simulation_paths.csv.gz",
    parse_dates=["date", "target_session_date_pred"],
)


st.title("News-Driven Alpha")
st.caption(
    "Timestamp-safe CSI 300 research dashboard: 2023–2026 "
    "financial-news sentiment, genuine OOS direction forecasts, "
    "paired block-bootstrap uncertainty, and a separate GARCH risk overlay."
)

st.sidebar.header("Research status")
st.sidebar.markdown(
    """**Directional experiment:** frozen and completed  
**2025:** untouched holdout  
**2026:** locked-model robustness  
**GARCH:** risk overlay only"""
)
st.sidebar.caption(
    "Generated data/results are local artifacts and are excluded "
    "from version control."
)

tabs = st.tabs(
    [
        "Overview",
        "Data & timing",
        "Directional OOS",
        "Uncertainty & calibration",
        "GARCH & simulation",
        "Reproducibility",
    ]
)


with tabs[0]:
    st.subheader("Research question")
    st.markdown(
        "Does **timestamp-safe China-focused financial-news sentiment** "
        "add incremental next-session forecasting information beyond "
        "market-only variables for the CSI 300?"
    )

    if alignment and master_summary:
        cols = st.columns(4)
        cols[0].metric(
            "Headline observations",
            f"{alignment['input_headlines_joined']:,}",
        )
        cols[1].metric(
            "Assigned to market windows",
            f"{alignment['assigned_headlines_total']:,}",
        )
        cols[2].metric(
            "Session-unique headlines",
            f"{master_summary['session_unique_normalized_headlines']:,}",
        )
        cols[3].metric(
            "CSI 300 master sessions",
            f"{master_summary['master_session_rows']:,}",
        )

        cols = st.columns(4)
        cols[0].metric(
            "Strict model-ready rows",
            f"{master_summary['strict_model_ready_rows']:,}",
        )
        cols[1].metric("2025 holdout", "223")
        cols[2].metric("2026 robustness", "181")
        cols[3].metric(
            "Timing violations",
            str(alignment["monthly_timing_violations"]),
        )

    st.markdown(
        """
        ### Current evidence

        The result is **mixed, not uniformly positive**. The strongest
        directional point result occurs for **XGBoost + sentiment in the
        locked 2026 robustness period**, but its paired block-bootstrap
        uncertainty interval still crosses zero. Therefore the project does
        **not** claim a statistically resolved sentiment alpha effect.

        The 2025 untouched holdout also shows mixed evidence: sentiment can
        improve probability losses/ranking in one model family while failing
        to improve 0.50-threshold classification.
        """
    )

    if bootstrap is not None:
        key = bootstrap[
            (bootstrap["family"] == "xgboost")
            & (bootstrap["period"] == "2026_robustness")
            & (
                bootstrap["metric"]
                == "balanced_accuracy_improvement"
            )
        ]
        if not key.empty:
            row = key.iloc[0]
            cols = st.columns(3)
            cols[0].metric(
                "2026 XGB BA increment",
                f"{row['point_increment'] * 100:+.2f} pp",
            )
            cols[1].metric(
                "95% CI lower",
                f"{row['ci_lower'] * 100:+.2f} pp",
            )
            cols[2].metric(
                "95% CI upper",
                f"{row['ci_upper'] * 100:+.2f} pp",
            )


with tabs[1]:
    st.subheader("Timestamp-safe session dataset")

    if master is None:
        st.info(
            "Run the historical alignment and master-dataset builders "
            "to populate this tab."
        )
    else:
        st.caption(
            "Forecast cutoff: 15:00 Asia/Shanghai. Headlines are assigned "
            "to (previous trading close, current trading close]."
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=master["date"],
                y=master["unique_headline_count_t"],
                name="Unique headlines",
                mode="lines",
            )
        )
        fig.update_layout(
            title="Timestamp-safe unique headline count by CSI 300 session",
            xaxis_title="Trading session",
            yaxis_title="Unique headlines",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=master["date"],
                y=master["sentiment_mean_t"],
                name="Pooled FinBERT sentiment",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=master["date"],
                y=master["sentiment_roll_20"],
                name="20-session sentiment",
                mode="lines",
            )
        )
        fig.update_layout(
            title="Session sentiment",
            xaxis_title="Trading session",
            yaxis_title="FinBERT score",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=master["date"],
                y=master["volatility_20_t"],
                name="20-session volatility",
                mode="lines",
            )
        )
        fig.update_layout(
            title="CSI 300 rolling volatility",
            xaxis_title="Trading session",
            yaxis_title="Daily volatility",
        )
        st.plotly_chart(fig, use_container_width=True)

        no_news = master[
            master["unique_headline_count_t"] == 0
        ]
        st.markdown(
            f"**Genuine no-news sessions:** {len(no_news)}"
        )
        if not no_news.empty:
            st.dataframe(
                no_news[
                    [
                        "date",
                        "unique_headline_count_t",
                        "sentiment_mean_t",
                        "sentiment_roll_20",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


with tabs[2]:
    st.subheader("Frozen directional OOS experiment")

    if metrics is None or predictions is None:
        st.info(
            "Run python run_directional_experiment.py to populate "
            "directional results."
        )
    else:
        display = metrics[
            [
                "model_name",
                "period",
                "n",
                "balanced_accuracy",
                "accuracy",
                "brier_score_loss",
                "log_loss",
                "roc_auc",
            ]
        ].copy()
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        period = st.selectbox(
            "Prediction period",
            sorted(predictions["period"].unique()),
            key="direction_period",
        )
        family = st.selectbox(
            "Model family",
            ["logistic", "xgboost"],
            key="direction_family",
        )

        pair = predictions[
            (predictions["period"] == period)
            & (predictions["family"] == family)
        ].copy()

        market_p = pair[
            pair["variant"] == "market_only"
        ][["target_session_date", "probability"]].rename(
            columns={"probability": "market"}
        )
        sentiment_p = pair[
            pair["variant"] == "market_plus_sentiment"
        ][["target_session_date", "probability"]].rename(
            columns={"probability": "sentiment"}
        )
        joined = market_p.merge(
            sentiment_p,
            on="target_session_date",
            validate="one_to_one",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=joined["target_session_date"],
                y=joined["market"],
                name="Market-only p(up)",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=joined["target_session_date"],
                y=joined["sentiment"],
                name="Market + sentiment p(up)",
                mode="lines",
            )
        )
        fig.add_hline(y=0.5, line_dash="dash")
        fig.update_layout(
            title=f"{family.title()} OOS probabilities — {period}",
            xaxis_title="Target trading session",
            yaxis_title="Probability of positive next-session return",
        )
        st.plotly_chart(fig, use_container_width=True)

        if incremental is not None:
            st.subheader("Incremental sentiment differences")
            st.dataframe(
                incremental,
                use_container_width=True,
                hide_index=True,
            )


with tabs[3]:
    st.subheader("Paired block-bootstrap uncertainty")

    if bootstrap is None:
        st.info(
            "Run python evaluate_oos_uncertainty.py to populate "
            "uncertainty results."
        )
    else:
        family = st.selectbox(
            "Family",
            ["logistic", "xgboost"],
            key="uncert_family",
        )
        period = st.selectbox(
            "Period",
            [
                "2024_validation",
                "2025_holdout",
                "2026_robustness",
            ],
            key="uncert_period",
        )
        subset = bootstrap[
            (bootstrap["family"] == family)
            & (bootstrap["period"] == period)
        ].copy()

        st.dataframe(
            subset[
                [
                    "metric",
                    "point_increment",
                    "ci_lower",
                    "ci_upper",
                    "bootstrap_probability_increment_gt_zero",
                    "resolution_95pct",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=subset["point_increment"],
                y=subset["metric"],
                mode="markers",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(
                        subset["ci_upper"]
                        - subset["point_increment"]
                    ),
                    arrayminus=(
                        subset["point_increment"]
                        - subset["ci_lower"]
                    ),
                ),
                name="Incremental sentiment effect",
            )
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(
            title="95% paired moving-block bootstrap intervals",
            xaxis_title=(
                "Increment (positive values favor sentiment)"
            ),
            yaxis_title="Metric",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Calibration")
    if calibration is not None:
        st.dataframe(
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
            ],
            use_container_width=True,
            hide_index=True,
        )

    if reliability is not None:
        rel_family = st.selectbox(
            "Reliability family",
            ["logistic", "xgboost"],
            key="rel_family",
        )
        rel_period = st.selectbox(
            "Reliability period",
            [
                "2024_validation",
                "2025_holdout",
                "2026_robustness",
            ],
            key="rel_period",
        )
        rel_variant = st.selectbox(
            "Reliability variant",
            ["market_only", "market_plus_sentiment"],
            key="rel_variant",
        )
        rel = reliability[
            (reliability["family"] == rel_family)
            & (reliability["period"] == rel_period)
            & (reliability["variant"] == rel_variant)
        ].copy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rel["mean_predicted_probability"],
                y=rel["observed_up_rate"],
                mode="lines+markers",
                name="Observed",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ideal",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(
            title="5-bin reliability curve",
            xaxis_title="Mean predicted p(up)",
            yaxis_title="Observed up rate",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)


with tabs[4]:
    st.subheader("GARCH(1,1) risk overlay")

    if garch_metrics is None or simulation_metrics is None:
        st.info(
            "Run python run_garch_oos_simulation.py to populate "
            "GARCH and simulation results."
        )
    else:
        st.dataframe(
            garch_metrics,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "GARCH determines risk scaling only; it does not generate "
            "the directional probability."
        )

        model_name = st.selectbox(
            "Simulation model",
            sorted(simulation_metrics["model_name"].unique()),
            key="sim_model",
        )
        period = st.selectbox(
            "Simulation period",
            [
                "2024_validation",
                "2025_holdout",
                "2026_robustness",
            ],
            key="sim_period",
        )
        overlay = st.selectbox(
            "Position rule",
            ["garch_scaled", "unscaled"],
            key="sim_overlay",
        )

        row = simulation_metrics[
            (simulation_metrics["model_name"] == model_name)
            & (simulation_metrics["period"] == period)
            & (simulation_metrics["overlay"] == overlay)
        ]
        if not row.empty:
            r = row.iloc[0]
            cols = st.columns(4)
            cols[0].metric(
                "Net return",
                pct(r["net_total_return"]),
            )
            cols[1].metric(
                "Benchmark",
                pct(r["benchmark_total_return"]),
            )
            cols[2].metric(
                "Net Sharpe",
                num(r["net_sharpe_zero_rf"], 2),
            )
            cols[3].metric(
                "Max drawdown",
                pct(r["net_max_drawdown"]),
            )

            cols = st.columns(4)
            cols[0].metric(
                "Active sessions",
                int(r["active_observations"]),
            )
            cols[1].metric(
                "Avg turnover",
                num(r["average_turnover"], 3),
            )
            cols[2].metric(
                "Cost drag",
                pct(r["transaction_cost_compounding_drag"]),
            )
            cols[3].metric(
                "Active-return difference",
                pct(r["active_total_return"]),
            )

        if simulation_paths is not None:
            path = simulation_paths[
                (simulation_paths["model_name"] == model_name)
                & (simulation_paths["period"] == period)
                & (simulation_paths["overlay"] == overlay)
            ].sort_values("date")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=path["date"],
                    y=path["net_wealth"],
                    name="Net simulated strategy",
                    mode="lines",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=path["date"],
                    y=path["benchmark_wealth"],
                    name="CSI 300 benchmark",
                    mode="lines",
                )
            )
            fig.update_layout(
                title="OOS simulated wealth",
                xaxis_title="Feature date",
                yaxis_title="Wealth, base 1.0",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Simulation conventions are fixed: p>=0.55 long, p<=0.45 "
            "short, 10 bps per unit turnover, max |position|=1. "
            "Results are historical research, not live-trading evidence."
        )


with tabs[5]:
    st.subheader("Reproducible workflow")
    st.code(
        """python build_historical_market.py
python build_timestamp_safe_alignment.py
python build_master_session_dataset.py
python run_directional_experiment.py
python evaluate_oos_uncertainty.py
python run_garch_oos_simulation.py
python -m streamlit run app.py""",
        language="bash",
    )

    st.markdown(
        """
        **Frozen specifications**

        - `config/directional_experiment_2023_2026.json`
        - `config/oos_uncertainty_spec.json`
        - `config/garch_simulation_spec.json`

        The project retains null and negative results. It does not claim
        causality, Jensen alpha, statistically resolved sentiment alpha,
        or live-trading profitability.
        """
    )


st.divider()
st.caption(
    "Research and educational use only. No broker connection or "
    "live-capital execution is included."
)
