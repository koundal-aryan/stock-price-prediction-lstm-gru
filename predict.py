"""
Stock Price Prediction Inference Pipeline
Loads the best saved deep learning model (LSTM or GRU) and scaler,
takes the latest 60 trading days of ADANIPORTS, and forecasts the next trading day's Close price.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.data_loader import load_dataset
from utils.preprocessing import preprocess_stock_data


DEFAULT_DATA_PATH = os.path.join(PROJECT_DIR, "data", "ADANIPORTS-selected-columns-2.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")


def get_model_status() -> dict:
    """
    Checks whether trained models and scaler are available.
    """
    lstm_exists = os.path.exists(os.path.join(MODELS_DIR, "lstm_model.keras"))
    gru_exists = os.path.exists(os.path.join(MODELS_DIR, "gru_model.keras"))
    scaler_exists = os.path.exists(SCALER_PATH)
    results_exists = os.path.exists(RESULTS_PATH)
    
    is_ready = (lstm_exists or gru_exists) and scaler_exists and results_exists
    
    best_model_name = "LSTM"
    if results_exists:
        try:
            with open(RESULTS_PATH, "r") as f:
                res = json.load(f)
                best_model_name = res.get("best_model", "LSTM")
        except Exception:
            pass
            
    return {
        "ready": is_ready,
        "lstm_exists": lstm_exists,
        "gru_exists": gru_exists,
        "scaler_exists": scaler_exists,
        "results_exists": results_exists,
        "best_model": best_model_name
    }


def predict_next_day(
    data_path: str = DEFAULT_DATA_PATH,
    model_choice: str = "best",
    sequence_length: int = 60
) -> dict:
    """
    Predicts the next trading day's closing price using the trained deep learning model.
    
    Parameters:
        data_path (str): Path to historical stock dataset.
        model_choice (str): 'best', 'LSTM', or 'GRU'.
        sequence_length (int): Sliding window length (60 days).
        
    Returns:
        dict: Next-day prediction results and metrics.
    """
    status = get_model_status()
    if not status["ready"]:
        raise FileNotFoundError(
            "Models have not been trained yet. Please run python train.py first."
        )
        
    # Determine which model file to load
    target_model = status["best_model"] if model_choice.lower() == "best" else model_choice.upper()
    model_filename = f"{target_model.lower()}_model.keras"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_filename}' not found at {model_path}.")
        
    # Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load and preprocess dataset
    df = load_dataset(data_path)
    clean_df, _ = preprocess_stock_data(df, verbose=False)
    
    if len(clean_df) < sequence_length:
        raise ValueError(
            f"Dataset has only {len(clean_df)} records, but sequence_length is {sequence_length}."
        )
        
    # Extract latest sequence_length days of Close price
    latest_window = clean_df.tail(sequence_length).copy()
    latest_close_val = float(latest_window["Close"].iloc[-1])
    latest_date = latest_window["Date"].iloc[-1].strftime("%Y-%m-%d")
    
    # Scale sequence using the fitted scaler
    raw_window = latest_window[["Close"]].values
    scaled_window = scaler.transform(raw_window)
    
    # Reshape for Keras: (batch_size=1, sequence_length=60, n_features=1)
    input_sequence = np.reshape(scaled_window, (1, sequence_length, 1))
    
    # Load Model and Predict
    model = tf.keras.models.load_model(model_path)
    scaled_prediction = model.predict(input_sequence, verbose=0)
    
    # Inverse transform to recover INR ₹ price
    unscaled_pred = scaler.inverse_transform(scaled_prediction)
    predicted_close = float(unscaled_pred[0, 0])
    
    # Compute price differences
    abs_diff = predicted_close - latest_close_val
    pct_diff = (abs_diff / latest_close_val) * 100.0
    
    # Load test metrics from results
    results_meta = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                all_results = json.load(f)
                results_meta = all_results.get(target_model, {})
        except Exception:
            pass
            
    prediction_result = {
        "model_used": target_model,
        "is_best_model": (target_model == status["best_model"]),
        "latest_date": latest_date,
        "latest_close": round(latest_close_val, 2),
        "predicted_close": round(predicted_close, 2),
        "absolute_change": round(abs_diff, 2),
        "percentage_change": round(pct_diff, 2),
        "direction": "UP" if abs_diff >= 0 else "DOWN",
        "window_size": sequence_length,
        "model_mae": results_meta.get("MAE", None),
        "model_rmse": results_meta.get("RMSE", None),
        "model_r2": results_meta.get("R2", None),
        "disclaimer": (
            "IMPORTANT: This is an algorithmic deep learning prediction intended strictly "
            "for academic research and educational demonstration. Stock markets involve systematic "
            "financial risk, macroeconomic factors, and unpredictable real-time sentiment that historical "
            "price patterns alone cannot guarantee."
        )
    }
    
    return prediction_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict Next Trading Day Close Price for ADANIPORTS")
    parser.add_argument("--model", type=str, default="best", choices=["best", "lstm", "gru", "LSTM", "GRU"],
                        help="Choose model to evaluate: 'best', 'lstm', or 'gru'")
    args = parser.parse_args()
    
    try:
        res = predict_next_day(model_choice=args.model)
        print("=" * 65)
        print(f"      NEXT TRADING DAY CLOSE PREDICTION ({res['model_used']})")
        print("=" * 65)
        print(f" Latest Trading Date       : {res['latest_date']}")
        print(f" Latest Actual Close Price : ₹{res['latest_close']:.2f}")
        print(f" Predicted Next-Day Close  : ₹{res['predicted_close']:.2f}")
        direction_sign = "+" if res['absolute_change'] >= 0 else ""
        print(f" Expected Absolute Change  : {direction_sign}₹{res['absolute_change']:.2f}")
        print(f" Expected Percentage Change: {direction_sign}{res['percentage_change']:.2f}%")
        print("-" * 65)
        print(f" Model Test RMSE           : ₹{res['model_rmse']:.2f}" if res['model_rmse'] else "")
        print(f" Model Test MAE            : ₹{res['model_mae']:.2f}" if res['model_mae'] else "")
        print("=" * 65)
        print(f" [DISCLAIMER]\n {res['disclaimer']}")
        print("=" * 65)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
