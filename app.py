import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="News-Driven Alpha",
    page_icon="📈",
    layout="wide",
)


def load_csv(path: str, parse_dates=None):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def metric_text(value, percent=False, digits=3):
    if value is None or pd.isna(value):
        return "—"
    if percent:
        return f"{float(value) * 100:.{digits}f}%"
    return f"{float(value):.{digits}f}"


st.title("News-Driven Alpha")
st.caption(
    "CSI 300 research dashboard for China-focused news sentiment, "
    "next-session direction forecasting and a separate GARCH volatility-risk overlay."
)

st.sidebar.header("Research workflow")
st.sidebar.code("python run_research_pipeline.py")
st.sidebar.caption(
    "The dashboard is read-only. Descriptive analysis, model fitting and "
    "backtesting are produced by the research pipeline."
)

news = load_csv("data/news_daily.csv", parse_dates=["date"])
sentiment = load_csv("data/sentiment_features.csv", parse_dates=["date"])
market = load_csv("data/csi300_features.csv", parse_dates=["date"])

descriptive_summary = load_csv("outputs/descriptive_summary.csv")
sentiment_terciles = load_csv("outputs/sentiment_terciles.csv")

evaluation = load_csv("outputs/model_evaluation.csv")
predictions = load_csv("outputs/oos_predictions.csv", parse_dates=["date"])
backtest = load_csv("outputs/oos_backtest.csv", parse_dates=["date"])
backtest_metrics = load_csv("outputs/oos_backtest_metrics.csv")
importance = load_csv("outputs/feature_importance.csv")
latest_signal = load_csv("outputs/daily_prediction.csv")

overview_tab, data_tab, model_tab, backtest_tab, signal_tab = st.tabs(
    ["Overview", "Data", "Model evaluation", "OOS backtest", "Latest research signal"]
)

with overview_tab:
    st.markdown(
        """
        The research question is whether **China-focused economic and financial
        news sentiment adds incremental next-session information beyond
        market-only variables for the CSI 300**.

        The directional comparison is a **market + sentiment XGBoost model**
        versus a **market-only XGBoost baseline** under chronological
        expanding-window evaluation. GARCH(1,1) is estimated separately and is
        used only for one-step-ahead volatility risk scaling.
        """
    )

    if descriptive_summary is not None and not descriptive_summary.empty:
        row = descriptive_summary.iloc[-1]
        cols = st.columns(4)
        cols[0].metric(
            "Headline observations",
            f"{int(row.get('headline_title_observations', 0)):,}",
        )
        cols[1].metric(
            "Aligned sentiment-return pairs",
            int(row.get("sentiment_next_return_pairs", 0)),
        )
        cols[2].metric(
            "Spearman rank correlation",
            metric_text(
                row.get("spearman_time_series_rank_correlation"),
                digits=3,
            ),
        )
        cols[3].metric(
            "CSI 300 compounded return",
            metric_text(
                row.get("compounded_return_from_stored_session_log_returns"),
                percent=True,
                digits=2,
            ),
        )

        cols = st.columns(4)
        cols[0].metric("News days", int(row.get("news_days", 0)))
        cols[1].metric(
            "Naive sentiment-sign hit rate",
            metric_text(
                row.get("naive_sentiment_sign_hit_rate"),
                percent=True,
                digits=1,
            ),
        )
        cols[2].metric(
            "Annualized realized volatility",
            metric_text(
                row.get("annualized_realized_volatility"),
                percent=True,
                digits=2,
            ),
        )
        cols[3].metric(
            "Labelled model rows",
            int(row.get("labelled_model_rows", 0)),
        )

        st.caption(
            "The CSI 300 return compounds the 49 stored session log returns; "
            "the 22 April return is measured from the preceding trading close. "
            "The current statistics are descriptive and do not establish "
            "statistical significance, causality or a tradable sentiment effect."
        )

        if sentiment_terciles is not None and not sentiment_terciles.empty:
            st.subheader("Sentiment-sorted next-session returns")
            display_terciles = sentiment_terciles.copy()
            display_terciles["mean_next_session_return"] = (
                display_terciles["mean_next_session_return"].astype(float) * 100
            )
            display_terciles["next_session_positive_rate"] = (
                display_terciles["next_session_positive_rate"].astype(float) * 100
            )
            display_terciles = display_terciles.rename(
                columns={
                    "sentiment_group": "Sentiment group",
                    "observations": "Observations",
                    "mean_sentiment": "Mean sentiment",
                    "mean_next_session_return": "Mean next-session return (%)",
                    "next_session_positive_rate": "Next session positive (%)",
                }
            )
            st.dataframe(
                display_terciles,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "Run python run_research_pipeline.py to generate the reproducible "
            "descriptive summary shown on this page."
        )

    if news is not None and not news.empty:
        st.caption(
            f"Committed news sample: {news['date'].min().date()} "
            f"to {news['date'].max().date()}."
        )

with data_tab:
    if sentiment is None or sentiment.empty:
        st.warning("No sentiment data found.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=sentiment["date"],
                y=sentiment["sentiment_mean"],
                name="FinBERT daily mean",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=sentiment["date"],
                y=sentiment["article_count"],
                name="Title/headline observations",
                opacity=0.35,
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Daily China-focused news sentiment and headline volume",
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Sentiment", secondary_y=False)
        fig.update_yaxes(
            title_text="Title/headline observations",
            secondary_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    if market is not None and not market.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=market["date"],
                y=market["volatility"],
                name="20-day realized volatility",
            )
        )
        fig.update_layout(
            title="CSI 300 rolling volatility",
            xaxis_title="Date",
            yaxis_title="Daily volatility",
        )
        st.plotly_chart(fig, use_container_width=True)

with model_tab:
    if evaluation is None or evaluation.empty:
        st.info(
            "Model-performance statistics are not available for the current "
            "sample because only 8 labelled model rows remain after rolling-feature construction and the configured OOS reporting threshold has not "
            "been reached."
        )
    else:
        st.subheader("Market-only versus market + sentiment")
        display = evaluation.copy()
        for col in ["accuracy", "balanced_accuracy", "brier", "log_loss"]:
            if col in display:
                display[col] = display[col].astype(float)
        st.dataframe(display, use_container_width=True, hide_index=True)

    if predictions is not None and not predictions.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predictions["date"],
                y=predictions["direction_probability_full"],
                name="Market + sentiment p(up)",
                mode="lines+markers",
            )
        )
        if "direction_probability_market_only" in predictions:
            fig.add_trace(
                go.Scatter(
                    x=predictions["date"],
                    y=predictions["direction_probability_market_only"],
                    name="Market-only p(up)",
                    mode="lines",
                )
            )
        fig.add_hline(y=0.5, line_dash="dash")
        fig.update_layout(
            title="Out-of-sample direction probabilities",
            yaxis_title="Probability of positive next-session return",
            xaxis_title="Date",
        )
        st.plotly_chart(fig, use_container_width=True)

    if importance is not None and not importance.empty:
        ranked = importance.sort_values("importance", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=ranked["importance"],
                y=ranked["feature"],
                orientation="h",
            )
        )
        fig.update_layout(
            title="Final-model feature importance",
            xaxis_title="XGBoost importance",
            yaxis_title="Feature",
        )
        st.plotly_chart(fig, use_container_width=True)

with backtest_tab:
    if backtest is None or backtest.empty:
        st.info(
            "No OOS simulated-strategy backtest is reportable for the current "
            "sample. Generated backtest outputs remain outside version control."
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=backtest["date"],
                y=backtest["strategy_wealth"],
                name="Research strategy",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=backtest["date"],
                y=backtest["benchmark_wealth"],
                name="CSI 300 benchmark",
            )
        )
        fig.update_layout(
            title="Out-of-sample cumulative wealth",
            xaxis_title="Date",
            yaxis_title="Wealth (base 1.0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    if backtest_metrics is not None and not backtest_metrics.empty:
        row = backtest_metrics.iloc[-1]
        cols = st.columns(4)
        cols[0].metric(
            "Strategy return",
            metric_text(row.get("strategy_total_return"), percent=True, digits=2),
        )
        cols[1].metric(
            "Benchmark return",
            metric_text(row.get("benchmark_total_return"), percent=True, digits=2),
        )
        cols[2].metric(
            "Strategy Sharpe",
            metric_text(row.get("strategy_sharpe"), digits=2),
        )
        cols[3].metric(
            "Max drawdown",
            metric_text(row.get("strategy_max_drawdown"), percent=True, digits=2),
        )
        st.dataframe(backtest_metrics, use_container_width=True, hide_index=True)

with signal_tab:
    if latest_signal is None or latest_signal.empty:
        st.info(
            "No research signal is available in this checkout. "
            "The daily research refresh requires a trained canonical model artifact."
        )
    else:
        row = latest_signal.iloc[-1]
        cols = st.columns(4)
        cols[0].metric("Feature date", str(row.get("date", "—")))
        cols[1].metric("Signal", str(row.get("signal", "—")))
        cols[2].metric(
            "p(up)",
            metric_text(row.get("direction_probability"), digits=3),
        )
        cols[3].metric(
            "Position",
            metric_text(row.get("position"), digits=3),
        )
        st.write(str(row.get("explanation", "")))

st.divider()
st.caption(
    "Research/educational use only. No broker connection or live-capital execution is included."
)
