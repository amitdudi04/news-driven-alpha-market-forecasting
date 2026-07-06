import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==========================================
# MODULE 19: LIVE STREAMLIT DASHBOARD
# ==========================================
# Objective: Provide a clean, real-time, auto-refreshing institutional 
# interface tracking today's execution signals and historical performance.

# Page Configuration (Wide Layout)
st.set_page_config(
    page_title="News-Driven Alpha Dashboard", 
    layout="wide", 
    page_icon="📈"
)

# Function to load data safely, cached for 60 seconds to enable auto-refresh behavior
@st.cache_data(ttl=60)
def load_data():
    base_dir = os.getcwd()
    try:
        daily_pred = pd.read_csv(os.path.join(base_dir, 'outputs', 'daily_prediction.csv'))
        live_track = pd.read_csv(os.path.join(base_dir, 'outputs', 'live_tracking.csv'))
        # Using final_dataset to reliably grab the latest sentiment feature
        features = pd.read_csv(os.path.join(base_dir, 'data', 'final_dataset.csv'))
        return daily_pred, live_track, features
    except Exception as e:
        return None, None, None

def main():
    st.title("📈 News-Driven Alpha: Live Quantitative Dashboard")
    st.markdown("Real-time monitoring for the CSI 300 Sentiment Trading Strategy.")
    
    daily_pred, live_track, features = load_data()
    
    if daily_pred is None or live_track is None or features is None:
        st.error("Data files not found. Ensure the pipeline has been executed and outputs/data directories exist.")
        return
        
    # --- 1. TOP METRICS ROW (TODAY'S EXECUTION) ---
    latest_pred = daily_pred.iloc[-1]
    latest_feature = features.iloc[-1]
    
    # Custom CSS to style the metrics
    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
            background-color: #1e1e1e;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="📅 Target Execution Date", value=str(latest_pred['date']))
        
    with col2:
        # Determine color direction for position
        pos_val = float(latest_pred['position'])
        if pos_val > 0:
            delta_color = "normal"
            pos_str = f"Leverage: {pos_val:.2f}x"
        elif pos_val < 0:
            delta_color = "inverse"
            pos_str = f"Leverage: {pos_val:.2f}x"
        else:
            delta_color = "off"
            pos_str = "Flat (Cash)"
            
        st.metric(label="⚡ Current Signal", value=str(latest_pred['signal']), delta=pos_str, delta_color=delta_color)
        
    with col3:
        pred_ret_pct = float(latest_pred['predicted_return']) * 100
        st.metric(label="🎯 Predicted T+1 Return", value=f"{pred_ret_pct:+.2f}%")
        
    with col4:
        sentiment_score = float(latest_feature['sentiment_mean'])
        st.metric(label="📰 Latest Sentiment Score", value=f"{sentiment_score:+.3f}")
        
    st.markdown("---")
    
    # Format dates
    live_track['date'] = pd.to_datetime(live_track['date'])
    
    # --- 2. CUMULATIVE RETURNS CHART (MAIN PANEL) ---
    st.subheader("📊 Portfolio Performance (Out-of-Sample)")
    fig1 = go.Figure()
    
    # Strategy
    fig1.add_trace(go.Scatter(
        x=live_track['date'], y=live_track['cum_strategy'], 
        mode='lines', name='News-Driven Strategy (Net)', 
        line=dict(color='#2ca02c', width=3)
    ))
    
    # Benchmark
    fig1.add_trace(go.Scatter(
        x=live_track['date'], y=live_track['cum_benchmark'], 
        mode='lines', name='CSI 300 Benchmark', 
        line=dict(color='#7f7f7f', width=2, dash='dot')
    ))
    
    fig1.update_layout(
        height=450, 
        margin=dict(l=0, r=0, t=30, b=0), 
        hovermode='x unified', 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333')
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # --- 3. RISK METRICS (SUB-PANELS) ---
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🎯 Rolling Sharpe Ratio (20-Day)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=live_track['date'], y=live_track['rolling_sharpe'], 
            mode='lines', name='Rolling Sharpe', 
            line=dict(color='#1f77b4', width=2)
        ))
        # Zero threshold line
        fig2.add_trace(go.Scatter(
            x=[live_track['date'].iloc[0], live_track['date'].iloc[-1]], y=[0, 0], 
            mode='lines', name='Baseline (0.0)', 
            line=dict(color='red', width=1, dash='dash')
        ))
        fig2.update_layout(
            height=300, margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_chart2:
        st.subheader("📉 Underwater Drawdown")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=live_track['date'], y=live_track['drawdown'] * 100, 
            mode='lines', name='Drawdown %', 
            line=dict(color='#d62728', width=1), 
            fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.3)'
        ))
        fig3.update_layout(
            height=300, margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified', yaxis_title='Loss (%)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333')
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    
    # Auto-refresh / Force update mechanism
    col_btn, col_txt = st.columns([1, 10])
    with col_btn:
        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_txt:
        st.caption("Dashboard data caches for 60 seconds to simulate a live UI without overloading disk reads. Click Refresh to force an immediate file pull.")

if __name__ == "__main__":
    main()
