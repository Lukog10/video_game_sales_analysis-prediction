# -*- coding: utf-8 -*-
"""
Model Evaluation Module
=======================
Handles evaluation metrics computation for trained models.

What was done:
    - R² Score calculation
    - Mean Absolute Error (MAE) calculation
    - Root Mean Squared Error (RMSE) calculation
    - Comprehensive model comparison

Key methods:
    - sklearn.metrics (r2_score, mean_absolute_error, mean_squared_error)
    - numpy for RMSE calculation

Attributes involved:
    - y_true: Actual Global_Sales values
    - y_pred: Predicted Global_Sales values
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_r2(y_true, y_pred):
    """
    Calculate R² (coefficient of determination) score.
    
    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    float
        R² score (higher is better, max = 1.0).
    """
    return r2_score(y_true, y_pred)


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    float
        MAE score (lower is better).
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    float
        RMSE score (lower is better).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(name, y_true, y_pred):
    """
    Evaluate a model and print all metrics.
    
    Parameters
    ----------
    name : str
        Name of the model.
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    dict
        Dictionary containing all metrics.
    """
    r2 = calculate_r2(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    print(f"\n{name} Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    
    return {'name': name, 'r2': r2, 'mae': mae, 'rmse': rmse}


def evaluate_all_models(results_dict, y_true):
    """
    Evaluate all models and create a comparison summary.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary from train_all_models containing predictions.
    y_true : array-like
        Actual test values.
    
    Returns
    -------
    list
        List of evaluation dictionaries for each model.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    evaluations = []
    
    for name, data in results_dict.items():
        eval_result = evaluate_model(name, y_true, data['predictions'])
        evaluations.append(eval_result)
    
    return evaluations


def print_comparison_table(evaluations):
    """
    Print a formatted comparison table of all models.
    
    Parameters
    ----------
    evaluations : list
        List of evaluation dictionaries.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'R² Score':>12} {'MAE':>12} {'RMSE':>12}")
    print("-" * 63)
    
    for eval_result in evaluations:
        print(f"{eval_result['name']:<25} {eval_result['r2']:>12.4f} {eval_result['mae']:>12.4f} {eval_result['rmse']:>12.4f}")
    
    # Find best model by R² score
    best_model = max(evaluations, key=lambda x: x['r2'])
    print("-" * 63)
    print(f"\n🏆 Best Model: {best_model['name']} (R² = {best_model['r2']:.4f})")


if __name__ == "__main__":
    # Test the module
    import numpy as np
    
    # Simulate predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    print("Testing evaluation functions:")
    print(f"R² Score: {calculate_r2(y_true, y_pred):.4f}")
    print(f"MAE: {calculate_mae(y_true, y_pred):.4f}")
    print(f"RMSE: {calculate_rmse(y_true, y_pred):.4f}")
