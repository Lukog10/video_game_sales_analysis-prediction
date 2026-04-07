# -*- coding: utf-8 -*-
"""
Prediction & Output Module
==========================
Handles final predictions and output formatting.

What was done:
    - Global Sales Prediction generation
    - Model comparison output
    - Results export functionality

Key methods:
    - Model predict() methods
    - Pandas for output formatting

Attributes involved:
    - Predicted Global_Sales values
    - Model performance metrics
"""

import pandas as pd
import numpy as np


def generate_predictions(model, X, model_name="Model"):
    """
    Generate predictions using a trained model.
    
    Parameters
    ----------
    model : estimator
        Trained sklearn/xgboost model.
    X : pd.DataFrame
        Feature matrix for prediction.
    model_name : str
        Name of the model for display.
    
    Returns
    -------
    np.ndarray
        Predicted values.
    """
    predictions = model.predict(X)
    print(f"\n[Prediction] {model_name}: Generated {len(predictions)} predictions")
    return predictions


def create_prediction_dataframe(X_test, y_true, predictions_dict):
    """
    Create a DataFrame comparing actual vs predicted values.
    
    Parameters
    ----------
    X_test : pd.DataFrame
        Test features.
    y_true : pd.Series
        Actual target values.
    predictions_dict : dict
        Dictionary of model_name: predictions pairs.
    
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame.
    """
    result_df = X_test.copy()
    result_df['Actual_Global_Sales'] = y_true.values
    
    for model_name, predictions in predictions_dict.items():
        col_name = f'Predicted_{model_name.replace(" ", "_")}'
        result_df[col_name] = predictions
    
    return result_df


def print_sample_predictions(y_true, predictions_dict, n_samples=10):
    """
    Print a sample of predictions from all models.
    
    Parameters
    ----------
    y_true : array-like
        Actual values.
    predictions_dict : dict
        Dictionary of model_name: predictions pairs.
    n_samples : int
        Number of samples to display.
    """
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Create comparison table
    print(f"\n{'Index':<8} {'Actual':>10}", end="")
    for name in predictions_dict.keys():
        print(f" {name[:12]:>12}", end="")
    print()
    print("-" * (20 + 13 * len(predictions_dict)))
    
    for i in range(min(n_samples, len(y_true))):
        print(f"{i:<8} {y_true.iloc[i]:>10.2f}", end="")
        for predictions in predictions_dict.values():
            print(f" {predictions[i]:>12.2f}", end="")
        print()


def export_predictions(result_df, filename="predictions_output.csv"):
    """
    Export predictions to a CSV file.
    
    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame with predictions.
    filename : str
        Output filename.
    """
    result_df.to_csv(filename, index=False)
    print(f"\n[Export] Predictions saved to: {filename}")


def print_final_summary(evaluations):
    """
    Print final summary with best model recommendation.
    
    Parameters
    ----------
    evaluations : list
        List of evaluation dictionaries from model_evaluation.
    """
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    best_r2 = max(evaluations, key=lambda x: x['r2'])
    best_mae = min(evaluations, key=lambda x: x['mae'])
    best_rmse = min(evaluations, key=lambda x: x['rmse'])
    
    print(f"\n📊 Best R² Score:  {best_r2['name']} ({best_r2['r2']:.4f})")
    print(f"📉 Lowest MAE:     {best_mae['name']} ({best_mae['mae']:.4f})")
    print(f"📉 Lowest RMSE:    {best_rmse['name']} ({best_rmse['rmse']:.4f})")
    
    # Overall recommendation (based on R²)
    print(f"\n🏆 RECOMMENDED MODEL: {best_r2['name']}")
    print("   (Based on highest R² score)")


if __name__ == "__main__":
    # Test the module
    import numpy as np
    
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions_dict = {
        'Random Forest': np.array([1.1, 2.2, 2.9, 4.1, 4.8]),
        'XGBoost': np.array([1.0, 2.1, 3.1, 3.9, 5.1]),
        'Hybrid': np.array([1.05, 2.15, 3.0, 4.0, 4.95])
    }
    
    print_sample_predictions(y_true, predictions_dict)
