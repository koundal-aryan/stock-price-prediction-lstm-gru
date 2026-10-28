"""
Flask Web Application for ADANIPORTS Stock Price Prediction
Production-ready web server serving live dashboards, EDA, model comparisons, and real-time inference.
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Ensure project root is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.data_loader import load_dataset, get_dataset_summary
from utils.preprocessing import preprocess_stock_data
from utils.feature_engineering import create_technical_features, compute_return_statistics
from utils.visualization import (
    plot_closing_price_history,
    plot_moving_averages,
    plot_daily_returns_volatility,
    plot_actual_vs_predicted,
    plot_lstm_vs_gru_comparison,
    plot_loss_curves
)
from predict import predict_next_day, get_model_status

app = Flask(__name__)
app.secret_key = "adaniports-deep-learning-secret-key-mba-project"

CSV_PATH = os.path.join(PROJECT_DIR, "data", "ADANIPORTS-selected-columns-2.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")

# In-memory cached objects
DATA_CACHE = {
    "df_clean": None,
    "df_featured": None,
    "summary": None,
    "prep_report": None,
    "return_stats": None,
    "results": None
}


def load_cached_data():
    """
    Loads and caches preprocessed data and model results in memory
    to ensure instantaneous page loads.
    """
    if DATA_CACHE["df_clean"] is None:
        if os.path.exists(CSV_PATH):
            raw_df = load_dataset(CSV_PATH)
            clean_df, prep_report = preprocess_stock_data(raw_df, verbose=False)
            featured_df = create_technical_features(clean_df)
            summary = get_dataset_summary(clean_df)
            return_stats = compute_return_statistics(featured_df)
            
            DATA_CACHE["df_clean"] = clean_df
            DATA_CACHE["df_featured"] = featured_df
            DATA_CACHE["summary"] = summary
            DATA_CACHE["prep_report"] = prep_report
            DATA_CACHE["return_stats"] = return_stats
            
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                DATA_CACHE["results"] = json.load(f)
        except Exception as e:
            print(f"[App] Error reading results JSON: {e}")
            
    return DATA_CACHE


@app.route("/")
def index():
    """Project Overview & Academic Executive Summary Page"""
    cache = load_cached_data()
    status = get_model_status()
    return render_template(
        "index.html",
        active_page="index",
        summary=cache["summary"],
        results=cache["results"],
        status=status
    )


@app.route("/dashboard")
def dashboard():
    """Main Analytics & Live KPI Dashboard Page"""
    status = get_model_status()
    if not status["ready"]:
        return render_template(
            "index.html",
            active_page="dashboard",
            not_trained=True,
            error_message="Models have not been trained yet. Please run python train.py first."
        )
        
    cache = load_cached_data()
    results = cache["results"]
    df_featured = cache["df_featured"]
    
    # Generate live next-day prediction
    try:
        prediction = predict_next_day(model_choice="best")
    except Exception as e:
        prediction = None
        print(f"[App] Prediction error: {e}")
        
    # Generate interactive charts
    chart_history = plot_closing_price_history(df_featured)
    chart_sma = plot_moving_averages(df_featured)
    chart_returns = plot_daily_returns_volatility(df_featured)
    
    # Actual vs Predicted Charts
    meta = results.get("metadata", {})
    test_dates = pd.to_datetime(meta.get("test_dates", []))
    test_actual = meta.get("test_actual", [])
    best_model_name = results.get("best_model", "GRU")
    best_preds = meta.get(f"{best_model_name.lower()}_predictions", [])
    lstm_preds = meta.get("lstm_predictions", [])
    gru_preds = meta.get("gru_predictions", [])
    
    chart_best_pred = plot_actual_vs_predicted(
        dates=test_dates,
        y_true=test_actual,
        y_pred=best_preds,
        model_title=best_model_name,
        color="#10B981" if best_model_name == "GRU" else "#2563EB"
    )
    
    chart_comparison = plot_lstm_vs_gru_comparison(
        dates=test_dates,
        y_true=test_actual,
        lstm_pred=lstm_preds,
        gru_pred=gru_preds
    )
    
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        results=results,
        prediction=prediction,
        chart_history=chart_history,
        chart_sma=chart_sma,
        chart_returns=chart_returns,
        chart_best_pred=chart_best_pred,
        chart_comparison=chart_comparison
    )


@app.route("/analysis")
def analysis():
    """Exploratory Financial Data Analysis (EDA) Page"""
    cache = load_cached_data()
    df_featured = cache["df_featured"]
    summary = cache["summary"]
    prep_report = cache["prep_report"]
    return_stats = cache["return_stats"]
    
    chart_history = plot_closing_price_history(df_featured)
    chart_sma = plot_moving_averages(df_featured)
    chart_returns = plot_daily_returns_volatility(df_featured)
    
    return render_template(
        "analysis.html",
        active_page="analysis",
        summary=summary,
        prep_report=prep_report,
        return_stats=return_stats,
        chart_history=chart_history,
        chart_sma=chart_sma,
        chart_returns=chart_returns
    )


@app.route("/models")
def models_view():
    """Model Comparison, Architecture Breakdown & Learning Curves Page"""
    status = get_model_status()
    if not status["ready"]:
        return render_template(
            "index.html",
            active_page="models",
            not_trained=True,
            error_message="Models have not been trained yet. Please run python train.py first."
        )
        
    cache = load_cached_data()
    results = cache["results"]
    meta = results.get("metadata", {})
    
    test_dates = pd.to_datetime(meta.get("test_dates", []))
    test_actual = meta.get("test_actual", [])
    lstm_preds = meta.get("lstm_predictions", [])
    gru_preds = meta.get("gru_predictions", [])
    lstm_hist = meta.get("lstm_history", {})
    gru_hist = meta.get("gru_history", {})
    
    chart_loss = plot_loss_curves(lstm_hist, gru_hist)
    chart_lstm = plot_actual_vs_predicted(test_dates, test_actual, lstm_preds, model_title="LSTM", color="#2563EB")
    chart_gru = plot_actual_vs_predicted(test_dates, test_actual, gru_preds, model_title="GRU", color="#EF4444")
    
    return render_template(
        "models.html",
        active_page="models",
        results=results,
        chart_loss=chart_loss,
        chart_lstm=chart_lstm,
        chart_gru=chart_gru
    )


@app.route("/predict")
def prediction_view():
    """Dedicated Next-Day Price Prediction Page with Scenario Selection"""
    status = get_model_status()
    if not status["ready"]:
        return render_template(
            "index.html",
            active_page="predict",
            not_trained=True,
            error_message="Models have not been trained yet. Please run python train.py first."
        )
        
    cache = load_cached_data()
    results = cache["results"]
    df_featured = cache["df_featured"]
    
    selected_model = request.args.get("model", "best")
    try:
        prediction = predict_next_day(model_choice=selected_model)
    except Exception as e:
        flash(f"Prediction error: {str(e)}", "danger")
        prediction = predict_next_day(model_choice="best")
        
    # Get last 10 rows of input sequence for display
    recent_rows = df_featured.tail(10).copy()
    recent_rows["Date"] = recent_rows["Date"].dt.strftime("%Y-%m-%d")
    recent_records = recent_rows.to_dict(orient="records")
    
    return render_template(
        "prediction.html",
        active_page="predict",
        prediction=prediction,
        selected_model=selected_model,
        results=results,
        recent_rows=recent_records
    )


@app.route("/api/predict")
def api_predict():
    """RESTful API endpoint for algorithmic forecasting"""
    model_choice = request.args.get("model", "best")
    try:
        res = predict_next_day(model_choice=model_choice)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    import socket

    def is_port_available(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return True
            except OSError:
                return False

    port = int(os.environ.get("PORT", 5000 if is_port_available(5000) else 5001))
    print(f"Starting Flask server on http://127.0.0.1:{port} (http://localhost:{port}) ...")
    app.run(host="0.0.0.0", port=port, debug=False)
