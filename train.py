"""
End-to-End Deep Learning Training Pipeline for Stock Price Prediction
Trains LSTM and GRU models on ADANIPORTS historical stock dataset.
Evaluates both architectures, compares performance, and saves all artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Ensure project root is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.data_loader import load_dataset, get_dataset_summary
from utils.preprocessing import preprocess_stock_data
from utils.feature_engineering import create_technical_features, compute_return_statistics
from utils.sequence_generator import prepare_train_test_data
from utils.models import build_lstm_model, build_gru_model, train_model
from utils.evaluation import calculate_metrics, compare_models


def run_training_pipeline(
    csv_path: str = None,
    epochs: int = 50,
    batch_size: int = 32,
    sequence_length: int = 60,
    train_ratio: float = 0.80
):
    """
    Executes the full pipeline:
    Load -> Clean -> Feature Engineering -> Split -> Scale -> Sequences -> Train LSTM & GRU -> Evaluate -> Save
    """
    print("=" * 70)
    print("  ADANIPORTS STOCK PRICE PREDICTION: DEEP LEARNING TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Resolve paths
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, "data", "ADANIPORTS-selected-columns-2.csv")
        
    models_dir = os.path.join(PROJECT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    lstm_save_path = os.path.join(models_dir, "lstm_model.keras")
    gru_save_path = os.path.join(models_dir, "gru_model.keras")
    scaler_save_path = os.path.join(models_dir, "scaler.pkl")
    results_save_path = os.path.join(models_dir, "model_results.json")
    
    # 2. Load dataset
    print(f"\n[Step 1/8] Loading dataset from: {csv_path}")
    raw_df = load_dataset(csv_path)
    initial_summary = get_dataset_summary(raw_df)
    print(f" Loaded {initial_summary['total_rows']} rows, {initial_summary['total_columns']} columns.")
    print(f" Detected columns: {initial_summary['columns']}")
    
    # 3. Clean and Preprocess
    print("\n[Step 2/8] Preprocessing and validating time-series integrity...")
    clean_df, prep_report = preprocess_stock_data(raw_df, verbose=True)
    
    # 4. Feature Engineering
    print("\n[Step 3/8] Computing technical features (SMA 20/50/100, returns, volatility)...")
    featured_df = create_technical_features(clean_df)
    return_stats = compute_return_statistics(featured_df)
    print(f" Daily Returns Statistics: Mean = {return_stats['mean_return']:.3f}%, Std = {return_stats['std_return']:.3f}%")
    
    # 5. Split, Scale (Train Only), and Generate Sequences
    print(f"\n[Step 4/8] Chronological split (80% Train, 20% Test) & Sequence generation (Window = {sequence_length})...")
    data_dict = prepare_train_test_data(
        df=featured_df,
        target_col="Close",
        train_ratio=train_ratio,
        sequence_length=sequence_length,
        scaler_save_path=scaler_save_path
    )
    
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    y_test_unscaled = data_dict["y_test_unscaled"]
    test_dates = data_dict["test_dates"]
    scaler = data_dict["scaler"]
    
    # 6. Train LSTM Model
    print("\n[Step 5/8] Building and Training LSTM Model...")
    lstm_model = build_lstm_model(input_shape=(sequence_length, X_train.shape[2]))
    lstm_model, lstm_history = train_model(
        model=lstm_model,
        X_train=X_train,
        y_train=y_train,
        model_name="LSTM",
        save_path=lstm_save_path,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.10,
        verbose=1
    )
    
    # 7. Train GRU Model
    print("\n[Step 6/8] Building and Training GRU Model...")
    gru_model = build_gru_model(input_shape=(sequence_length, X_train.shape[2]))
    gru_model, gru_history = train_model(
        model=gru_model,
        X_train=X_train,
        y_train=y_train,
        model_name="GRU",
        save_path=gru_save_path,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.10,
        verbose=1
    )
    
    # 8. Inference on Test Set & Unscaling
    print("\n[Step 7/8] Generating Test Predictions & Inverting MinMax Scaling...")
    lstm_test_scaled_pred = lstm_model.predict(X_test, verbose=0)
    gru_test_scaled_pred = gru_model.predict(X_test, verbose=0)
    
    # Inverse transform to get predicted rupee prices
    lstm_test_pred = scaler.inverse_transform(lstm_test_scaled_pred).flatten()
    gru_test_pred = scaler.inverse_transform(gru_test_scaled_pred).flatten()
    
    # Format test dates as strings
    test_date_strs = [d.strftime("%Y-%m-%d") for d in test_dates]
    
    # 9. Model Comparison & Metrics Saving
    print("\n[Step 8/8] Evaluating regression metrics and determining best model...")
    extra_metadata = {
        "dataset_summary": initial_summary,
        "preprocessing_report": prep_report,
        "return_statistics": return_stats,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "train_ratio": train_ratio,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test))
        },
        "lstm_history": lstm_history,
        "gru_history": gru_history,
        "test_dates": test_date_strs,
        "test_actual": [float(x) for x in y_test_unscaled],
        "lstm_predictions": [float(x) for x in lstm_test_pred],
        "gru_predictions": [float(x) for x in gru_test_pred]
    }
    
    comparison_results = compare_models(
        lstm_true=y_test_unscaled,
        lstm_pred=lstm_test_pred,
        gru_true=y_test_unscaled,
        gru_pred=gru_test_pred,
        save_path=results_save_path,
        extra_metadata=extra_metadata
    )
    
    print("\n Training Pipeline Finished Successfully!")
    print(f" Artifacts Saved:")
    print(f"  - Scaler       : {scaler_save_path}")
    print(f"  - LSTM Model   : {lstm_save_path}")
    print(f"  - GRU Model    : {gru_save_path}")
    print(f"  - Metrics JSON : {results_save_path}")
    
    return comparison_results


if __name__ == "__main__":
    run_training_pipeline()
