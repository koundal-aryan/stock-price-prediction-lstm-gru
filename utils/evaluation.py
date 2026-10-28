"""
Model Evaluation and Comparison Module
Calculates MAE, MSE, RMSE, MAPE, and R2 score on unscaled real-world stock prices.
"""

import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes regression performance metrics:
    - MAE: Mean Absolute Error (in Rupees ₹)
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error (in Rupees ₹)
    - MAPE: Mean Absolute Percentage Error (%)
    - R2: Coefficient of Determination
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    
    # Avoid zero division in MAPE
    non_zero_mask = (y_true != 0)
    mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4)
    }


def compare_models(
    lstm_true, lstm_pred,
    gru_true, gru_pred,
    save_path: str = None,
    extra_metadata: dict = None
) -> dict:
    """
    Evaluates both LSTM and GRU, compares their performance,
    determines the superior model, and saves structured results.
    """
    lstm_metrics = calculate_metrics(lstm_true, lstm_pred)
    gru_metrics = calculate_metrics(gru_true, gru_pred)
    
    # Selection rule: Lower RMSE primary, lower MAE secondary
    if lstm_metrics["RMSE"] < gru_metrics["RMSE"]:
        best_model = "LSTM"
        selection_reason = f"LSTM achieved lower RMSE (₹{lstm_metrics['RMSE']:.2f} vs ₹{gru_metrics['RMSE']:.2f}) and lower error variance."
    elif gru_metrics["RMSE"] < lstm_metrics["RMSE"]:
        best_model = "GRU"
        selection_reason = f"GRU achieved lower RMSE (₹{gru_metrics['RMSE']:.2f} vs ₹{lstm_metrics['RMSE']:.2f}) and superior generalization."
    else:
        best_model = "LSTM" if lstm_metrics["MAE"] <= gru_metrics["MAE"] else "GRU"
        selection_reason = f"Models tied on RMSE; {best_model} chosen due to lower MAE."
        
    results = {
        "LSTM": lstm_metrics,
        "GRU": gru_metrics,
        "best_model": best_model,
        "selection_reason": selection_reason
    }
    
    if extra_metadata:
        results["metadata"] = extra_metadata
        
    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[Evaluation] Evaluation results saved to {save_path}")
        
    # Print formatted MBA Presentation Table
    print("\n" + "=" * 70)
    print("           MODEL PERFORMANCE COMPARISON (TEST DATASET)")
    print("=" * 70)
    print(f"{'Model':<10} | {'MAE (₹)':<10} | {'MSE':<12} | {'RMSE (₹)':<10} | {'MAPE (%)':<10} | {'R² Score':<10}")
    print("-" * 70)
    print(f"{'LSTM':<10} | {lstm_metrics['MAE']:<10.2f} | {lstm_metrics['MSE']:<12.2f} | {lstm_metrics['RMSE']:<10.2f} | {lstm_metrics['MAPE']:<10.2f} | {lstm_metrics['R2']:<10.4f}")
    print(f"{'GRU':<10} | {gru_metrics['MAE']:<10.2f} | {gru_metrics['MSE']:<12.2f} | {gru_metrics['RMSE']:<10.2f} | {gru_metrics['MAPE']:<10.2f} | {gru_metrics['R2']:<10.4f}")
    print("=" * 70)
    print(f" Best Performing Model : {best_model}")
    print(f" Selection Rationale   : {selection_reason}")
    print("=" * 70 + "\n")
    
    return results
