"""
Preprocessing Module for Stock Price Prediction
Cleans data, removes duplicates, handles missing values, and enforces time-series integrity.
"""

import pandas as pd
import numpy as np


def preprocess_stock_data(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Cleans and preprocesses the stock market dataset.
    
    Steps:
    1. Removes duplicate rows (by Date).
    2. Enforces numeric datatypes on financial columns.
    3. Handles missing values using forward-fill followed by backward-fill.
    4. Validates price consistency (Price > 0, High >= Low).
    5. Returns cleaned DataFrame and preprocessing report.
    """
    report = {}
    initial_rows = len(df)
    report["initial_rows"] = initial_rows
    report["initial_columns"] = len(df.columns)
    
    # 1. Duplicate handling
    duplicates = int(df.duplicated(subset=["Date"]).sum())
    report["duplicates_found"] = duplicates
    if duplicates > 0:
        df = df.drop_duplicates(subset=["Date"], keep="first").copy()
    else:
        df = df.copy()
        
    # 2. Identify candidate price columns
    potential_numeric = ["Prev Close", "Open", "High", "Low", "Last", "Close", "VWAP", "Volume", "Turnover"]
    existing_numeric = [col for col in potential_numeric if col in df.columns]
    
    # Cast to numeric, invalid to NaN
    for col in existing_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # 3. Missing values report before fill
    missing_before = {k: int(v) for k, v in df[existing_numeric].isnull().sum().to_dict().items()}
    report["missing_values_before"] = missing_before
    
    # Forward fill then backward fill for time series
    df[existing_numeric] = df[existing_numeric].ffill().bfill()
    
    # 4. Anomaly / Invalid Value Detection
    invalid_mask = pd.Series(False, index=df.index)
    if "Close" in df.columns:
        invalid_mask = invalid_mask | (df["Close"] <= 0)
    if "High" in df.columns and "Low" in df.columns:
        invalid_mask = invalid_mask | (df["High"] < df["Low"])
        
    invalid_count = int(invalid_mask.sum())
    report["invalid_price_records_removed"] = invalid_count
    if invalid_count > 0:
        df = df[~invalid_mask].copy()
        
    # Chronological sort guarantee
    df.sort_values(by="Date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    report["final_usable_records"] = len(df)
    report["start_date"] = df["Date"].min().strftime("%Y-%m-%d")
    report["end_date"] = df["Date"].max().strftime("%Y-%m-%d")
    report["columns_retained"] = df.columns.tolist()
    
    if verbose:
        print("=" * 60)
        print("           PREPROCESSING SUMMARY REPORT")
        print("=" * 60)
        print(f" Initial records       : {report['initial_rows']}")
        print(f" Duplicate dates found : {report['duplicates_found']}")
        print(f" Date range            : {report['start_date']} to {report['end_date']}")
        print(f" Invalid rows removed  : {report['invalid_price_records_removed']}")
        print(f" Final usable records  : {report['final_usable_records']}")
        print(f" Columns available     : {', '.join(report['columns_retained'])}")
        print("=" * 60)
        
    return df, report
