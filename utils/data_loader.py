"""
Data Loader Module for Stock Price Prediction
Handles loading, auto-detection of columns, chronological sorting, and basic validation.
"""

import os
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads dataset from CSV, dynamically detects columns,
    converts Date to datetime, and sorts chronologically.
    
    Parameters:
        file_path (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Cleaned and chronologically sorted DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("The provided CSV dataset is empty.")
        
    # Auto-detect Date column (case-insensitive)
    date_col = None
    for col in df.columns:
        if col.strip().lower() == "date":
            date_col = col
            break
            
    if date_col is None:
        raise KeyError("Could not find a 'Date' column in the dataset.")
        
    # Standardize Date column name to 'Date'
    if date_col != "Date":
        df.rename(columns={date_col: "Date"}, inplace=True)
        
    # Convert Date to datetime and sort
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Drop rows with invalid dates if any
    initial_len = len(df)
    df = df.dropna(subset=["Date"])
    if len(df) < initial_len:
        print(f"[DataLoader] Dropped {initial_len - len(df)} rows with invalid dates.")
        
    # Chronological sort
    df.sort_values(by="Date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generates a structured summary of the dataset.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    summary = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "start_date": df["Date"].min().strftime("%Y-%m-%d") if "Date" in df.columns else None,
        "end_date": df["Date"].max().strftime("%Y-%m-%d") if "Date" in df.columns else None,
        "total_trading_days": int(len(df)),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return summary
