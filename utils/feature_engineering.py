"""
Feature Engineering Module for Stock Price Prediction
Generates technical indicators and time-series features strictly avoiding future data leakage.
"""

import pandas as pd
import numpy as np


def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs technical features and indicators using only backward-looking data.
    
    Features created:
    - Daily Return (%) = (Close_t - Close_{t-1}) / Close_{t-1} * 100
    - High-Low Difference = High_t - Low_t
    - Open-Close Difference = Open_t - Close_t
    - SMA 20 = 20-day Simple Moving Average of Close
    - SMA 50 = 50-day Simple Moving Average of Close
    - SMA 100 = 100-day Simple Moving Average of Close
    - 20-day Rolling Volatility (std of daily returns)
    
    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame with Date and OHLC prices.
        
    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    data = df.copy()
    
    # Ensure chronological order
    data.sort_values(by="Date", ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Prev Close verification
    if "Prev Close" not in data.columns and "Close" in data.columns:
        data["Prev Close"] = data["Close"].shift(1)
        
    # Price Differences
    if "High" in data.columns and "Low" in data.columns:
        data["High_Low_Diff"] = data["High"] - data["Low"]
        
    if "Open" in data.columns and "Close" in data.columns:
        data["Open_Close_Diff"] = data["Open"] - data["Close"]
        
    # Daily Return (%)
    if "Close" in data.columns:
        data["Daily_Return"] = data["Close"].pct_change() * 100.0
        # First row pct_change is NaN, fill with 0.0
        data["Daily_Return"] = data["Daily_Return"].fillna(0.0)
        
        # Simple Moving Averages
        data["SMA_20"] = data["Close"].rolling(window=20, min_periods=1).mean()
        data["SMA_50"] = data["Close"].rolling(window=50, min_periods=1).mean()
        data["SMA_100"] = data["Close"].rolling(window=100, min_periods=1).mean()
        
        # Exponential Moving Average (EMA 20)
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
        
        # 20-day Rolling Volatility (Standard Deviation of Returns)
        data["Volatility_20"] = data["Daily_Return"].rolling(window=20, min_periods=1).std().fillna(0.0)
        
    return data


def compute_return_statistics(df: pd.DataFrame) -> dict:
    """
    Computes summary statistics for stock returns.
    """
    if "Daily_Return" not in df.columns:
        returns = df["Close"].pct_change() * 100.0
    else:
        returns = df["Daily_Return"]
        
    clean_returns = returns.dropna()
    
    return {
        "mean_return": float(clean_returns.mean()),
        "std_return": float(clean_returns.std()),
        "max_return": float(clean_returns.max()),
        "min_return": float(clean_returns.min()),
        "skewness": float(clean_returns.skew()),
        "kurtosis": float(clean_returns.kurtosis()),
        "cumulative_return": float(((1 + clean_returns / 100).prod() - 1) * 100)
    }
