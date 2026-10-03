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
    "Research dashboard for CSI 300 next-session direction forecasting "
    "using GDELT news, FinBERT sentiment, XGBoost and a GARCH risk overlay."
)

st.sidebar.header("Research workflow")
st.sidebar.code("python run_research_pipeline.py")
st.sidebar.caption(
    "The dashboard is read-only. Model fitting and backtesting happen in the research pipeline."
)

news = load_csv("data/news_daily.csv", parse_dates=["date"])
sentiment = load_csv("data/sentiment_features.csv", parse_dates=["date"])
market = load_csv("data/csi300_features.csv", parse_dates=["date"])
evaluation = load_csv("outputs/model_evaluation.csv")
predictions = load_csv("outputs/oos_predictions.csv", parse_dates=["date"])
backtest = load_csv("outputs/oos_backtest.csv", parse_dates=["date"])
backtest_metrics = load_csv("outputs/oos_backtest_metrics.csv")
importance = load_csv("outputs/feature_importance.csv")
latest_signal = load_csv("outputs/daily_prediction.csv")

overview_tab, data_tab, model_tab, backtest_tab, signal_tab = st.tabs(
    ["Overview", "Data", "Model evaluation", "OOS backtest", "Latest paper signal"]
)

with overview_tab:
    cols = st.columns(4)
    clean_news_days = len(news) if news is not None else 0
    market_days = len(market) if market is not None else 0
    oos_n = (
        int(evaluation["n"].max())
        if evaluation is not None and not evaluation.empty and "n" in evaluation
        else 0
    )
    latest_market = (
        market["date"].max().strftime("%Y-%m-%d")
        if market is not None and not market.empty
        else "—"
    )
    cols[0].metric("News days", clean_news_days)
    cols[1].metric("CSI 300 observations", market_days)
    cols[2].metric("OOS predictions", oos_n)
    cols[3].metric("Latest market date", latest_market)

    st.markdown(
        """
        The research question is whether daily financial-news sentiment adds
        incremental next-session directional information beyond market-only
        volatility and momentum features. The public evaluation uses expanding
        chronological splits and compares a **market + sentiment** XGBoost model
        against a **market-only** baseline.

        GARCH(1,1) is used separately as a one-step-ahead volatility forecast for
        risk scaling. It is not used to manufacture the directional target.
        """
    )

    if news is not None and not news.empty:
        st.info(
            f"Current committed news sample: {news['date'].min().date()} "
            f"to {news['date'].max().date()}. "
            "The clean history is short, so any generated performance result should be treated as exploratory."
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
                name="Article count",
                opacity=0.35,
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Daily financial-news sentiment and article volume",
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Sentiment", secondary_y=False)
        fig.update_yaxes(title_text="Articles", secondary_y=True)
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
            "No OOS evaluation is available for the current sample. "
            "Run the research pipeline after extending the clean news history."
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
            "No OOS backtest is available for the current sample. "
            "Generated backtest files are intentionally kept out of Git."
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
            "No paper-trading signal has been generated in this checkout. "
            "Run the daily research refresh after a canonical model artifact exists."
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
