"""
Visualization Module using Plotly
Generates interactive, responsive financial charts with hover, zoom, and range selectors.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# Dark/Light Modern Financial Dashboard Palette
THEME = {
    "bg_color": "rgba(248, 250, 252, 0.9)",
    "paper_bg": "rgba(255, 255, 255, 0.95)",
    "grid_color": "#E2E8F0",
    "text_color": "#1E293B",
    "primary": "#2563EB",
    "success": "#10B981",
    "danger": "#EF4444",
    "purple": "#8B5CF6",
    "amber": "#F59E0B",
    "slate": "#64748B"
}


def plot_closing_price_history(df: pd.DataFrame, as_html: bool = True):
    """
    Historical Closing Price Interactive Chart with Range Slider.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price (₹)",
        line=dict(color=THEME["primary"], width=2),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> ₹%{y:.2f}<extra></extra>"
    ))
    
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["VWAP"],
            mode="lines",
            name="VWAP (₹)",
            line=dict(color=THEME["purple"], width=1.5, dash="dash"),
            hovertemplate="<b>VWAP:</b> ₹%{y:.2f}<extra></extra>"
        ))
        
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig.to_html(include_plotlyjs=False, full_html=False) if as_html else fig


def plot_moving_averages(df: pd.DataFrame, as_html: bool = True):
    """
    Closing Price with 20-day, 50-day, and 100-day Simple Moving Averages.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Closing Price",
        line=dict(color="#0F172A", width=1.5),
        opacity=0.75,
        hovertemplate="<b>Close:</b> ₹%{y:.2f}<extra></extra>"
    ))
    
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["SMA_20"],
            mode="lines",
            name="20-day SMA",
            line=dict(color=THEME["primary"], width=1.8),
            hovertemplate="<b>SMA 20:</b> ₹%{y:.2f}<extra></extra>"
        ))
        
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["SMA_50"],
            mode="lines",
            name="50-day SMA",
            line=dict(color=THEME["amber"], width=1.8),
            hovertemplate="<b>SMA 50:</b> ₹%{y:.2f}<extra></extra>"
        ))
        
    if "SMA_100" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["SMA_100"],
            mode="lines",
            name="100-day SMA",
            line=dict(color=THEME["danger"], width=2),
            hovertemplate="<b>SMA 100:</b> ₹%{y:.2f}<extra></extra>"
        ))
        
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig.to_html(include_plotlyjs=False, full_html=False) if as_html else fig


def plot_daily_returns_volatility(df: pd.DataFrame, as_html: bool = True):
    """
    Subplots: Daily Percentage Return and Rolling 20-day Volatility.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Daily Percentage Return (%)", "20-day Rolling Volatility (%)")
    )
    
    returns = df["Daily_Return"] if "Daily_Return" in df.columns else df["Close"].pct_change() * 100
    
    # Positive / Negative colors
    colors = [THEME["success"] if r >= 0 else THEME["danger"] for r in returns]
    
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=returns,
            marker_color=colors,
            name="Daily Return %",
            hovertemplate="<b>Return:</b> %{y:.2f}%<extra></extra>"
        ),
        row=1, col=1
    )
    
    if "Volatility_20" in df.columns:
        vol = df["Volatility_20"]
    else:
        vol = returns.rolling(20, min_periods=1).std().fillna(0)
        
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=vol,
            mode="lines",
            line=dict(color=THEME["purple"], width=2),
            name="20d Volatility",
            hovertemplate="<b>Volatility:</b> %{y:.2f}%<extra></extra>"
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    return fig.to_html(include_plotlyjs=False, full_html=False) if as_html else fig


def plot_actual_vs_predicted(
    dates, y_true, y_pred, model_title: str = "Model", color: str = "#2563EB", as_html: bool = True
):
    """
    Actual vs Predicted Test Period Chart.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode="lines",
        name="Actual Close (₹)",
        line=dict(color="#0F172A", width=2),
        hovertemplate="<b>Actual:</b> ₹%{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode="lines",
        name=f"{model_title} Predicted (₹)",
        line=dict(color=color, width=2, dash="dash"),
        hovertemplate=f"<b>{model_title} Pred:</b> ₹%{{y:.2f}}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig.to_html(include_plotlyjs=False, full_html=False) if as_html else fig


def plot_lstm_vs_gru_comparison(dates, y_true, lstm_pred, gru_pred, as_html: bool = True):
    """
    Combined Actual vs LSTM vs GRU comparison chart on the Test Dataset.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode="lines",
        name="Actual Close",
        line=dict(color="#0F172A", width=2.5),
        hovertemplate="<b>Actual:</b> ₹%{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=lstm_pred,
        mode="lines",
        name="LSTM Prediction",
        line=dict(color=THEME["primary"], width=1.8),
        hovertemplate="<b>LSTM:</b> ₹%{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=gru_pred,
        mode="lines",
        name="GRU Prediction",
        line=dict(color=THEME["danger"], width=1.8, dash="dot"),
        hovertemplate="<b>GRU:</b> ₹%{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig.to_html(include_plotlyjs=False, full_html=False) if as_html else fig


def plot_loss_curves(lstm_hist: dict, gru_hist: dict, as_html: bool = True):
    """
    Training vs Validation Loss curves across training epochs for LSTM and GRU.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("LSTM Learning Curve (MSE Loss)", "GRU Learning Curve (MSE Loss)")
    )
    
    # LSTM
    epochs_lstm = list(range(1, len(lstm_hist.get("loss", [])) + 1))
    fig.add_trace(
        go.Scatter(x=epochs_lstm, y=lstm_hist.get("loss", []), mode="lines", name="LSTM Train Loss", line=dict(color=THEME["primary"])),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs_lstm, y=lstm_hist.get("val_loss", []), mode="lines", name="LSTM Val Loss", line=dict(color=THEME["primary"], dash="dash")),
        row=1, col=1
    )
    
    # GRU
    epochs_gru = list(range(1, len(gru_hist.get("loss", [])) + 1))
    fig.add_trace(
        go.Scatter(x=epochs_gru, y=gru_hist.get("loss", []), mode="lines", name="GRU Train Loss", line=dict(color=THEME["danger"])),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs_gru, y=gru_hist.get("val_loss", []), mode="lines", name="GRU Val Loss", line=dict(color=THEME["danger"], dash="dash")),
        row=1, col=2
    )
    
    fig.update_layout(
        title="<b>Deep Learning Convergence: Training vs Validation Loss Curves</b>",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Mean Squared Error", row=1, col=1)
    fig.update_yaxes(title_text="Mean Squared Error", row=1, col=2)
    
    return fig.to_html(include_plotlyjs=False, full_html=False) if as_html else fig
