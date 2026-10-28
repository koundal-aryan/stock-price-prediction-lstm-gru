"""
Sequence Generator Module for Stock Price Prediction
Implements strict chronological train/test splitting, isolated MinMax scaling,
and sequence sliding-window generation without look-ahead bias or data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def create_sequences(data_series: np.ndarray, sequence_length: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a 1D/2D time-series into sliding window sequences:
    Input: [t - sequence_length, ..., t - 1]
    Target: [t] (Next trading day)
    
    Parameters:
        data_series (np.ndarray): Scaled numerical array (N, 1) or (N,).
        sequence_length (int): Sliding window size (default 60 days).
        
    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y)
    """
    X, y = [], []
    for i in range(sequence_length, len(data_series)):
        X.append(data_series[i - sequence_length:i])
        y.append(data_series[i, 0] if data_series.ndim > 1 else data_series[i])
        
    return np.array(X), np.array(y)


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = "Close",
    train_ratio: float = 0.80,
    sequence_length: int = 60,
    scaler_save_path: str = None
) -> dict:
    """
    Performs chronological split, fits MinMaxScaler ONLY on training data,
    transforms both train and test data, and builds sequences.
    
    Parameters:
        df (pd.DataFrame): Preprocessed dataframe sorted chronologically.
        target_col (str): Target column name (default 'Close').
        train_ratio (float): Ratio for chronological training partition (default 0.80).
        sequence_length (int): Length of sliding window (default 60).
        scaler_save_path (str): Optional filepath to save the fitted scaler.
        
    Returns:
        dict: Training and testing sequences, raw arrays, dates, and the fitted scaler.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
        
    total_len = len(df)
    train_size = int(total_len * train_ratio)
    test_size = total_len - train_size
    
    # Chronological partition
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # 1. Fit scaler strictly on training set only (Prevents Data Leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_raw = train_df[[target_col]].values
    scaler.fit(train_raw)
    
    if scaler_save_path:
        joblib.dump(scaler, scaler_save_path)
        print(f"[SequenceGenerator] Scaler saved to: {scaler_save_path}")
        
    # 2. Transform training data
    train_scaled = scaler.transform(train_raw)
    
    # 3. Transform test data with warm-up context (previous sequence_length days from train)
    # This allows predicting every single point in the test period [train_size : total_len]
    test_context_raw = df.iloc[train_size - sequence_length:][[target_col]].values
    test_context_scaled = scaler.transform(test_context_raw)
    
    # 4. Generate sequences
    X_train, y_train = create_sequences(train_scaled, sequence_length=sequence_length)
    X_test, y_test = create_sequences(test_context_scaled, sequence_length=sequence_length)
    
    # Ensure shape is (batch_size, sequence_length, n_features)
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=-1)
    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, axis=-1)
        
    # Extract corresponding dates
    # Train dates for y_train correspond to df['Date'].iloc[sequence_length : train_size]
    train_dates = train_df["Date"].iloc[sequence_length:].reset_index(drop=True)
    # Test dates for y_test correspond exactly to df['Date'].iloc[train_size :]
    test_dates = test_df["Date"].reset_index(drop=True)
    
    # Unscaled test targets for true price evaluation
    y_test_unscaled = test_df[target_col].values
    
    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_unscaled": y_test_unscaled,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "scaler": scaler,
        "train_size": train_size,
        "test_size": test_size,
        "sequence_length": sequence_length
    }
    
    print(f"[SequenceGenerator] Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    print(f"[SequenceGenerator] Target evaluation points: {len(y_test_unscaled)} days ({test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')})")
    
    return result
