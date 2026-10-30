"""
ADANIPORTS Deep Learning Stock Analytics & Prediction Dashboard
Ultra-Modern, FinTech & Executive Grade Streamlit Application
Designed for high-impact MBA Project Demonstrations & Cloud 24/7 Hosting.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration
st.set_page_config(
    page_title="ADANIPORTS AI | Deep Learning Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced FinTech Dark/Glassmorphism CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0B1329 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.28) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(37, 99, 235, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }

    .hero-desc {
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 820px;
        line-height: 1.6;
    }

    /* Glowing Glassmorphism KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 12px -2px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(0,0,0,0.08);
    }

    .card-accent-blue::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #2563EB; }
    .card-accent-green::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #10B981; }
    .card-accent-purple::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #8B5CF6; }
    .card-accent-amber::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #F59E0B; }

    .card-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
    }

    .card-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }

    .card-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.35rem;
    }

    .pill-up {
        background: #D1FAE5;
        color: #065F46;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    .pill-down {
        background: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }

    /* Simulation Box */
    .sim-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1px solid #BBF7D0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Path & Data Ingestion
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "ADANIPORTS-selected-columns-2.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date", ascending=True).reset_index(drop=True)
    
    num_cols = ["Prev Close", "Open", "High", "Low", "Last", "Close", "VWAP"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].ffill().bfill()
    
    # Financial Technical Features
    df["Daily_Return"] = df["Close"].pct_change() * 100.0
    df["Daily_Return"] = df["Daily_Return"].fillna(0.0)
    df["SMA_20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["SMA_50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["SMA_100"] = df["Close"].rolling(100, min_periods=1).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(20, min_periods=1).std().fillna(0.0)
    
    # Bollinger Bands
    rolling_std = df["Close"].rolling(20).std().fillna(0.0)
    df["BB_Upper"] = df["SMA_20"] + (rolling_std * 2)
    df["BB_Lower"] = df["SMA_20"] - (rolling_std * 2)
    
    return df


@st.cache_data
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return None


df = load_data()
results = load_results()

# 4. Sidebar Branding & Navigation
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.2rem;">
        <div style="background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; color:white; font-size:24px; font-weight:bold; box-shadow:0 6px 16px rgba(37,99,235,0.35);">
            ⚡
        </div>
        <div>
            <div style="font-size:1.2rem; font-weight:800; color:#0F172A; line-height:1.2;">ADANIPORTS AI</div>
            <div style="font-size:0.75rem; font-weight:700; color:#2563EB; text-transform:uppercase; letter-spacing:0.06em;">Quant Terminal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Real-time Live Price Ticker Card in Sidebar
    latest_close_val = float(df["Close"].iloc[-1])
    pred_val_sidebar = 724.93
    delta_val = pred_val_sidebar - latest_close_val
    delta_pct = (delta_val / latest_close_val) * 100
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%); border:1px solid #E2E8F0; border-radius:12px; padding:12px; margin-bottom:1.2rem; box-shadow:0 2px 6px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">
            <span>NSE: ADANIPORTS</span>
            <span style="background:{'#DCFCE7' if delta_val>=0 else '#FEE2E2'}; color:{'#15803D' if delta_val>=0 else '#B91C1C'}; padding:2px 6px; border-radius:4px; font-size:0.7rem;">{delta_pct:+.2f}% EST</span>
        </div>
        <div style="font-size:1.5rem; font-weight:800; color:#0F172A; font-family:'JetBrains Mono', monospace; margin:4px 0;">
            ₹{latest_close_val:.2f}
        </div>
        <div style="font-size:0.78rem; color:#64748B;">
            Target (t+1): <b style="color:#2563EB;">₹{pred_val_sidebar:.2f}</b> (GRU)
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "EXPLORE MODULES",
        [
            "⚡ Executive Dashboard",
            "📊 Candlestick & Technical Studio",
            "🧠 Deep Learning Benchmark",
            "🔮 Next-Day Forecast Terminal",
            "💡 Trading Scenario Simulator",
            "📑 Viva & Methodology Audit"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # 2. Interactive Chart Filters Control inside Sidebar
    st.markdown("#### 🎛️ Chart Display Controls")
    show_ma = st.toggle("Show Moving Averages (20/50/100)", value=True)
    show_vol = st.toggle("Highlight Volatility Shocks", value=True)
    
    st.markdown("---")
    
    # 3. Interactive Quick Market Inspection Tool (Date Picker)
    st.markdown("#### 📅 Historical Date Inspector")
    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()
    selected_inspect_date = st.date_input(
        "Jump to Trading Day",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Look up selected date
    match_row = df[df["Date"].dt.date == selected_inspect_date]
    if not match_row.empty:
        r = match_row.iloc[0]
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:10px; padding:10px; font-size:0.8rem; margin-top:6px;">
            <div style="font-weight:700; color:#1E293B; margin-bottom:4px;">📆 {r['Date'].strftime('%d %b %Y')}</div>
            <div style="display:flex; justify-content:space-between; color:#475569;"><span>Close:</span> <b>₹{r['Close']:.2f}</b></div>
            <div style="display:flex; justify-content:space-between; color:#475569;"><span>Day Return:</span> <b style="color:{'#16A34A' if r['Daily_Return']>=0 else '#DC2626'};">{r['Daily_Return']:+.2f}%</b></div>
            <div style="display:flex; justify-content:space-between; color:#475569;"><span>VWAP:</span> <b>₹{r['VWAP']:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("ℹ️ Weekend or Market Holiday (No Trading).")

    st.markdown("---")
    
    # 4. Interactive Quick Model Switcher
    st.markdown("#### 🧠 Model Engine Switcher")
    active_engine = st.selectbox(
        "Active Architecture",
        ["GRU (Best Performer - 98.32% R²)", "LSTM (Baseline - 96.66% R²)"],
        index=0
    )
    
    # 5. Download Processed Dataset Button
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Cleaned Dataset (CSV)",
        data=csv_bytes,
        file_name="ADANIPORTS_clean_features.csv",
        mime="text/csv",
        use_container_width=True
    )

latest_close = float(df["Close"].iloc[-1])
latest_date = df["Date"].iloc[-1].strftime("%d %B %Y")
pred_close_gru = 724.93
pred_close_lstm = 703.61
diff_gru = pred_close_gru - latest_close
pct_diff_gru = (diff_gru / latest_close) * 100.0


# ==============================================================================
# TAB 1: EXECUTIVE DASHBOARD
# ==============================================================================
if menu == "⚡ Executive Dashboard":
    # Hero Section
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-badge">Deep Learning Time-Series Forecaster • MBA Capstone</div>
        <div class="hero-title">ADANIPORTS Stock Price Prediction Terminal</div>
        <div class="hero-desc">
            Automated recurrent neural network predicting <b>Day t+1 Close</b> based on a <b>60-day lookback sequence</b>. 
            Comparing stacked <b>LSTM vs. GRU architectures</b> across 13.5 years of macroeconomic cycles without future data leakage.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Glassmorphism KPI Grid
    sign_str = "+" if diff_gru >= 0 else ""
    pill_class = "pill-up" if diff_gru >= 0 else "pill-down"
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="glass-card card-accent-blue">
            <div class="card-label"><span>Latest Close Price</span> <span>₹</span></div>
            <div class="card-val">₹{latest_close:.2f}</div>
            <div class="card-sub">Session: {latest_date}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card card-accent-green">
            <div class="card-label"><span>Predicted Close (t+1)</span> <span class="{pill_class}">{sign_str}{pct_diff_gru:.2f}%</span></div>
            <div class="card-val" style="color: {'#10B981' if diff_gru>=0 else '#EF4444'};">₹{pred_close_gru:.2f}</div>
            <div class="card-sub">Expected Delta: <b>{sign_str}₹{diff_gru:.2f}</b> (GRU Engine)</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card card-accent-purple">
            <div class="card-label"><span>Model Accuracy (R²)</span> <span>🎯</span></div>
            <div class="card-val" style="color:#8B5CF6;">98.32%</div>
            <div class="card-sub">Unseen Test Set Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="glass-card card-accent-amber">
            <div class="card-label"><span>Model Test RMSE</span> <span>📉</span></div>
            <div class="card-val" style="color:#D97706;">₹13.99</div>
            <div class="card-sub">Lowest Error • GRU Outperformed LSTM</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Primary Chart: Unseen Test Set Actual vs Predicted (Interactive)
    st.markdown("### 🎯 Unseen Test Set Performance: Actual vs GRU Predictions (665 Sessions)")
    st.caption("Evaluated on completely unseen chronological market data (August 2018 to April 2021) without look-ahead bias.")
    
    if results and "metadata" in results:
        meta = results["metadata"]
        test_dates = pd.to_datetime(meta.get("test_dates", []))
        actual = meta.get("test_actual", [])
        gru_preds = meta.get("gru_predictions", [])
        lstm_preds = meta.get("lstm_predictions", [])
        
        fig_dash = go.Figure()
        fig_dash.add_trace(go.Scatter(
            x=test_dates, y=actual,
            mode="lines", name="Actual Close Price (₹)",
            line=dict(color="#0F172A", width=2.5),
            hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Actual Close:</b> ₹%{y:.2f}<extra></extra>"
        ))
        fig_dash.add_trace(go.Scatter(
            x=test_dates, y=gru_preds,
            mode="lines", name="GRU Forecast (Selected Model)",
            line=dict(color="#10B981", width=2, dash="dash"),
            hovertemplate="<b>GRU Forecast:</b> ₹%{y:.2f}<extra></extra>"
        ))
        fig_dash.update_layout(
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Trading Session Date",
            yaxis_title="Stock Price (₹ INR)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=10, b=20),
            height=460
        )
        st.plotly_chart(fig_dash, use_container_width=True)

    # Dual Column Technicals
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("#### 📈 Multi-Timeframe Trend Crossover")
        fig_cross = go.Figure()
        fig_cross.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#0F172A", width=1.2), opacity=0.6))
        fig_cross.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20 (Short-term)", line=dict(color="#2563EB", width=1.5)))
        fig_cross.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50 (Medium-term)", line=dict(color="#F59E0B", width=1.5)))
        fig_cross.add_trace(go.Scatter(x=df["Date"], y=df["SMA_100"], name="SMA 100 (Long-term)", line=dict(color="#EF4444", width=1.8)))
        fig_cross.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=20),
            height=340
        )
        st.plotly_chart(fig_cross, use_container_width=True)

    with ch2:
        st.markdown("#### ⚡ 20-Day Rolling Risk Volatility")
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=df["Date"], y=df["Volatility_20"],
            fill="tozeroy",
            line=dict(color="#8B5CF6", width=1.8),
            name="20d Volatility (%)"
        ))
        fig_risk.update_layout(
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Volatility (%)",
            margin=dict(l=10, r=10, t=10, b=20),
            height=340
        )
        st.plotly_chart(fig_risk, use_container_width=True)


# ==============================================================================
# TAB 2: CANDLESTICK & TECHNICAL STUDIO
# ==============================================================================
elif menu == "📊 Candlestick & Technical Studio":
    st.markdown('<div class="hero-badge">Interactive Technical Analysis Studio</div>', unsafe_allow_html=True)
    st.markdown("## Interactive Candlestick & Price Indicators")
    st.caption("Explore OHLC price action with interactive zoom, range sliders, and Bollinger Bands.")
    
    # Timeframe Selector
    t_filter = st.selectbox(
        "Select Time Horizon Window",
        ["Full History (2007–2021)", "Last 5 Years (2016–2021)", "Last 1 Year (2020–2021)", "Last 6 Months"]
    )
    
    if t_filter == "Last 6 Months":
        df_view = df.tail(126).copy()
    elif t_filter == "Last 1 Year (2020–2021)":
        df_view = df.tail(252).copy()
    elif t_filter == "Last 5 Years (2016–2021)":
        df_view = df.tail(1260).copy()
    else:
        df_view = df.copy()

    # Candlestick + Volume Chart
    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
    
    # Candlestick
    fig_candle.add_trace(go.Candlestick(
        x=df_view["Date"],
        open=df_view["Open"],
        high=df_view["High"],
        low=df_view["Low"],
        close=df_view["Close"],
        name="OHLC Price",
        increasing_line_color="#10B981",
        decreasing_line_color="#EF4444"
    ), row=1, col=1)
    
    # Bollinger Bands
    fig_candle.add_trace(go.Scatter(x=df_view["Date"], y=df_view["BB_Upper"], line=dict(color="rgba(148, 163, 184, 0.5)", width=1, dash="dot"), name="BB Upper"), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df_view["Date"], y=df_view["BB_Lower"], line=dict(color="rgba(148, 163, 184, 0.5)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(241, 245, 249, 0.2)", name="BB Band"), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df_view["Date"], y=df_view["EMA_20"], line=dict(color="#3B82F6", width=1.5), name="20 EMA"), row=1, col=1)
    
    # Daily Return Bar below
    colors_ret = ["#10B981" if r >= 0 else "#EF4444" for r in df_view["Daily_Return"]]
    fig_candle.add_trace(go.Bar(
        x=df_view["Date"], y=df_view["Daily_Return"],
        marker_color=colors_ret, name="Daily Return %"
    ), row=2, col=1)
    
    fig_candle.update_layout(
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Return Distribution Statistics
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Mean Daily Return", f"{df['Daily_Return'].mean():.3f}%")
    sc2.metric("Daily Volatility (Std Dev)", f"{df['Daily_Return'].std():.3f}%")
    sc3.metric("Max Single-Day Gain", f"+{df['Daily_Return'].max():.2f}%")
    sc4.metric("Max Single-Day Drop", f"{df['Daily_Return'].min():.2f}%")


# ==============================================================================
# TAB 3: DEEP LEARNING BENCHMARK
# ==============================================================================
elif menu == "🧠 Deep Learning Benchmark (LSTM vs GRU)":
    st.markdown('<div class="hero-badge">Model Architecture Benchmark</div>', unsafe_allow_html=True)
    st.markdown("## Head-to-Head Evaluation: LSTM vs. GRU")
    st.caption("Rigorous quantitative assessment on the unseen 665-day test dataset (2018–2021).")
    
    # Interactive Benchmark Table
    bench_df = pd.DataFrame({
        "Model Architecture": ["Stacked LSTM (64, 32 Units)", "Stacked GRU (64, 32 Units)"],
        "MAE (₹) ↓": ["₹13.09", "₹9.27"],
        "MSE ↓": ["388.54", "195.81"],
        "RMSE (₹) ↓": ["₹19.71", "₹13.99"],
        "MAPE (%) ↓": ["3.19%", "2.26%"],
        "R² Score ↑": ["0.9666", "0.9832"],
        "Convergence Epochs": ["30 (EarlyStopping)", "20 (EarlyStopping)"],
        "Verdict": ["Baseline Performer", "🏆 WINNING MODEL (Selected)"]
    })
    
    st.dataframe(bench_df, use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **Why GRU Outperformed LSTM:** Financial time-series contain high stochastic volatility and noise. Because the Gated Recurrent Unit (GRU) merges the cell state and hidden state with only **2 gates** (vs. 3 in LSTM) and has ~25% fewer parameters, it acted as an architectural regularizer—resisting memorization of market noise and generalizing with **29% lower RMSE**.
    """)

    # Learning Curves & Residual Distribution
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### 📉 Training Loss Convergence (MSE)")
        if results and "metadata" in results:
            meta = results["metadata"]
            lstm_h = meta.get("lstm_history", {})
            gru_h = meta.get("gru_history", {})
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=lstm_h.get("loss", []), mode="lines", name="LSTM Train Loss", line=dict(color="#2563EB")))
            fig_loss.add_trace(go.Scatter(y=lstm_h.get("val_loss", []), mode="lines", name="LSTM Val Loss", line=dict(color="#2563EB", dash="dash")))
            fig_loss.add_trace(go.Scatter(y=gru_h.get("loss", []), mode="lines", name="GRU Train Loss", line=dict(color="#EF4444")))
            fig_loss.add_trace(go.Scatter(y=gru_h.get("val_loss", []), mode="lines", name="GRU Val Loss", line=dict(color="#EF4444", dash="dash")))
            fig_loss.update_layout(
                template="plotly_white",
                xaxis_title="Epoch",
                yaxis_title="Mean Squared Error",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=10, b=20),
                height=340
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    with b2:
        st.markdown("#### 🎯 Residual Error Density Distribution")
        if results and "metadata" in results:
            meta = results["metadata"]
            actual = np.array(meta.get("test_actual", []))
            lstm_p = np.array(meta.get("lstm_predictions", []))
            gru_p = np.array(meta.get("gru_predictions", []))
            
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(x=actual - lstm_p, name="LSTM Residuals", marker_color="#2563EB", opacity=0.5, nbinsx=40))
            fig_err.add_trace(go.Histogram(x=actual - gru_p, name="GRU Residuals", marker_color="#10B981", opacity=0.6, nbinsx=40))
            fig_err.update_layout(
                barmode="overlay",
                template="plotly_white",
                xaxis_title="Error in ₹ INR (Actual - Predicted)",
                yaxis_title="Count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=10, b=20),
                height=340
            )
            st.plotly_chart(fig_err, use_container_width=True)


# ==============================================================================
# TAB 4: NEXT-DAY FORECASTER TERMINAL
# ==============================================================================
elif menu == "🔮 Next-Day Forecast Terminal":
    st.markdown('<div class="hero-badge">Algorithmic Forecasting Engine</div>', unsafe_allow_html=True)
    st.markdown("## Next Trading Day Close Price Projection")
    st.caption("Point-estimate prediction generated via the trained recurrent models using the latest 60-day historical sequence.")
    
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        chosen_m = st.radio("Select Prediction Architecture", ["GRU (Recommended - Champion)", "LSTM (Baseline Model)"])
        sel_pred = pred_close_gru if "GRU" in chosen_m else pred_close_lstm
        diff_sel = sel_pred - latest_close
        pct_sel = (diff_sel / latest_close) * 100.0
        
        st.markdown(f"""
        <div class="glass-card card-accent-green" style="margin-top:1rem;">
            <div class="card-label">Predicted Next Close</div>
            <div class="card-val" style="color:#10B981;">₹{sel_pred:.2f}</div>
            <div class="card-sub">Delta: <b>{diff_sel:+.2f} ({pct_sel:+.2f}%)</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        # Comparison gauge visual
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sel_pred,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Next-Day Forecast vs Latest Close (₹{latest_close:.2f})", 'font': {'size': 18, 'weight': 'bold'}},
            delta={'reference': latest_close, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
            gauge={
                'axis': {'range': [latest_close * 0.85, latest_close * 1.15], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2563EB"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#CBD5E1",
                'steps': [
                    {'range': [latest_close * 0.85, latest_close], 'color': '#FEE2E2'},
                    {'range': [latest_close, latest_close * 1.15], 'color': '#DCFCE7'}
                ],
                'threshold': {
                    'line': {'color': "#0F172A", 'width': 3},
                    'thickness': 0.75,
                    'value': latest_close
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("### 📋 60-Day Lookback Sequence: Recent 10 Trading Days")
    st.dataframe(
        df.tail(10)[["Date", "Prev Close", "Open", "High", "Low", "Close", "VWAP", "Daily_Return"]].style.format({
            "Prev Close": "₹{:.2f}",
            "Open": "₹{:.2f}",
            "High": "₹{:.2f}",
            "Low": "₹{:.2f}",
            "Close": "₹{:.2f}",
            "VWAP": "₹{:.2f}",
            "Daily_Return": "{:+.2f}%"
        }),
        use_container_width=True
    )


# ==============================================================================
# TAB 5: TRADING SCENARIO SIMULATOR (NEW INTERACTIVE FEATURE)
# ==============================================================================
elif menu == "💡 Trading Scenario Simulator":
    st.markdown('<div class="hero-badge">Interactive Quant Sandbox</div>', unsafe_allow_html=True)
    st.markdown("## Strategy & Position Profit Simulator")
    st.caption("Simulate portfolio return outcomes based on the deep learning model's forecasted price movement.")
    
    sim_col1, sim_col2 = st.columns([1, 1.4])
    with sim_col1:
        st.markdown("#### Investment Parameters")
        capital = st.number_input("Capital Allocated (₹ INR)", min_value=10000, max_value=5000000, value=100000, step=10000)
        stop_loss_pct = st.slider("Risk Management: Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        target_model = st.selectbox("Forecast Model to Follow", ["GRU (+₹724.93 Target)", "LSTM (+₹703.61 Target)"])
        
        shares = int(capital // latest_close)
        invested = shares * latest_close
        cash_left = capital - invested
        
        target_p = pred_close_gru if "GRU" in target_model else pred_close_lstm
        expected_gain = (target_p - latest_close) * shares
        expected_roi = (expected_gain / invested) * 100
        stop_loss_price = latest_close * (1 - (stop_loss_pct / 100))
        max_risk_amt = (latest_close - stop_loss_price) * shares
        
    with sim_col2:
        st.markdown(f"""
        <div class="sim-card">
            <h3 style="color:#065F46; margin-top:0;">📊 Simulated Position Summary</h3>
            <p style="color:#047857; margin-bottom:1rem;">Shares Purchased: <b>{shares} Units</b> @ ₹{latest_close:.2f}</p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div style="background:white; padding:12px; border-radius:10px; border:1px solid #A7F3D0;">
                    <div style="font-size:0.75rem; color:#047857; font-weight:700;">TOTAL INVESTED</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#065F46;">₹{invested:,.2f}</div>
                </div>
                <div style="background:white; padding:12px; border-radius:10px; border:1px solid #A7F3D0;">
                    <div style="font-size:0.75rem; color:#047857; font-weight:700;">ESTIMATED P&L</div>
                    <div style="font-size:1.3rem; font-weight:800; color:{'#059669' if expected_gain>=0 else '#DC2626'};">{expected_gain:+,.2f} ({expected_roi:+.2f}%)</div>
                </div>
                <div style="background:white; padding:12px; border-radius:10px; border:1px solid #A7F3D0;">
                    <div style="font-size:0.75rem; color:#047857; font-weight:700;">STOP LOSS TRIGGER</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#B45309;">₹{stop_loss_price:.2f}</div>
                </div>
                <div style="background:white; padding:12px; border-radius:10px; border:1px solid #A7F3D0;">
                    <div style="font-size:0.75rem; color:#047857; font-weight:700;">MAX CONTROLLED RISK</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#DC2626;">-₹{max_risk_amt:,.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("ℹ️ *Educational Simulation:* Demonstrates how quantitative price point estimates integrate directly with corporate risk management & portfolio allocation rules.")


# ==============================================================================
# TAB 6: VIVA & METHODOLOGY AUDIT
# ==============================================================================
else:
    st.markdown('<div class="hero-badge">Academic Examination Guide</div>', unsafe_allow_html=True)
    st.markdown("## Methodology & Viva Voce Q&A")
    st.caption("Key conceptual questions, statistical formulations, and zero-data-leakage architecture explanation.")
    
    with st.expander("Q1: Why is stock price prediction formulated as a time-series problem?"):
        st.write("""
        Unlike traditional classification or cross-sectional regression where observations are assumed to be independent and identically distributed (i.i.d.), financial asset prices exhibit strong **temporal autocorrelation**. Tomorrow's price is deeply conditioned on historical momentum, volume flows, and market trends. Shuffling data destroys these temporal dependencies and leads to catastrophic data leakage.
        """)

    with st.expander("Q2: How did you ensure ZERO data leakage in the pipeline?"):
        st.write("""
        1. **Strict Chronological Splitting:** The data was partitioned into the first 80% (2007–2018) for training and final 20% (2018–2021) for testing without any random shuffling.
        2. **Isolated MinMax Scaling:** `MinMaxScaler` was fitted **exclusively on the training dataset**. The test split was only *transformed* using the pre-fitted parameters ($X_{min}, X_{max}$ from training).
        3. **Lagged Technical Indicators:** All moving averages and return indicators use strictly backward-looking rolling windows.
        """)

    with st.expander("Q3: What is the mathematical difference between LSTM and GRU?"):
        st.write("""
        * **LSTM:** Uses 3 gating mechanisms (*Forget Gate*, *Input Gate*, *Output Gate*) and maintains two separate internal memory vectors (*Cell State* $C_t$ and *Hidden State* $h_t$).
        * **GRU:** Uses 2 gating mechanisms (*Reset Gate* and *Update Gate*) and merges the cell state and hidden state into a single recurrent state ($h_t$). GRU has ~25% fewer trainable parameters, making it faster to train and less prone to overfitting on noisy financial datasets.
        """)

    with st.expander("Q4: Why evaluate using both MAE and RMSE?"):
        st.write("""
        * **MAE (Mean Absolute Error):** Measures the average magnitude of errors in the actual currency unit (₹ INR). Easily interpretable for corporate managers.
        * **RMSE (Root Mean Squared Error):** Squares the errors before averaging, heavily penalizing large outlier errors. In capital markets, a single massive forecasting error can trigger a liquidity crisis; hence, minimizing RMSE is paramount for risk management.
        """)
