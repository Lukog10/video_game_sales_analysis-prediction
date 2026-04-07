# -*- coding: utf-8 -*-
"""
Video Game Sales Analysis & Prediction
=======================================
Main orchestration script that executes the complete pipeline.

Pipeline Flow:
    1. Data Collection     → Load dataset
    2. Data Preprocessing  → Clean and filter data
    3. EDA                 → Exploratory analysis (optional)
    4. Feature Encoding    → Label encode categoricals
    5. Model Training      → Train RF, XGBoost, Hybrid
    6. Model Evaluation    → Calculate metrics
    7. Prediction & Output → Generate and display results

Author: Refactored for academic presentation quality
"""

# =============================================================================
# IMPORTS
# =============================================================================
from src.data_collection import load_data, display_data_info
from src.data_preprocessing import preprocess_pipeline
from src.exploratory_data_analysis import run_full_eda
from src.feature_encoding import encode_categorical
from src.model_training import (
    prepare_features_target, 
    split_data, 
    train_all_models
)
from src.model_evaluation import evaluate_all_models, print_comparison_table
from src.prediction import print_sample_predictions, print_final_summary

from config import DATA_PATH


def main(run_eda=False, data_path=None):
    """
    Execute the complete Video Game Sales Analysis & Prediction pipeline.
    
    Parameters
    ----------
    run_eda : bool, optional
        Whether to run full EDA visualizations. Default is False.
    data_path : str, optional
        Path to the dataset. Defaults to DATA_PATH from config.
    """
    print("╔" + "═" * 58 + "╗")
    print("║" + " VIDEO GAME SALES ANALYSIS & PREDICTION ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # =========================================================================
    # STEP 1: DATA COLLECTION
    # =========================================================================
    print("\n" + "▶ STEP 1: DATA COLLECTION".center(60, "─"))
    
    if data_path is None:
        data_path = DATA_PATH
    
    df_raw = load_data(data_path)
    display_data_info(df_raw)
    
    # =========================================================================
    # STEP 2: DATA PREPROCESSING
    # =========================================================================
    print("\n" + "▶ STEP 2: DATA PREPROCESSING".center(60, "─"))
    
    df_processed = preprocess_pipeline(df_raw)
    
    # =========================================================================
    # STEP 3: EXPLORATORY DATA ANALYSIS (Optional)
    # =========================================================================
    if run_eda:
        print("\n" + "▶ STEP 3: EXPLORATORY DATA ANALYSIS".center(60, "─"))
        run_full_eda(df_processed)
    else:
        print("\n" + "▶ STEP 3: EDA".center(60, "─"))
        print("[EDA] Skipped. Set run_eda=True to generate visualizations.")
    
    # =========================================================================
    # STEP 4: FEATURE ENCODING
    # =========================================================================
    print("\n" + "▶ STEP 4: FEATURE ENCODING".center(60, "─"))
    
    df_encoded, encoders = encode_categorical(df_processed)
    
    # =========================================================================
    # STEP 5: MODEL TRAINING
    # =========================================================================
    print("\n" + "▶ STEP 5: MODEL TRAINING".center(60, "─"))
    
    X, y = prepare_features_target(df_encoded)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    results = train_all_models(X_train, X_test, y_train)
    
    # =========================================================================
    # STEP 6: MODEL EVALUATION
    # =========================================================================
    print("\n" + "▶ STEP 6: MODEL EVALUATION".center(60, "─"))
    
    evaluations = evaluate_all_models(results, y_test)
    print_comparison_table(evaluations)
    
    # =========================================================================
    # STEP 7: PREDICTION & OUTPUT
    # =========================================================================
    print("\n" + "▶ STEP 7: PREDICTION & OUTPUT".center(60, "─"))
    
    predictions_dict = {name: data['predictions'] for name, data in results.items()}
    print_sample_predictions(y_test, predictions_dict)
    print_final_summary(evaluations)
    
    print("\n" + "═" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("═" * 60)
    
    return results, evaluations


if __name__ == "__main__":
    # Run the complete pipeline
    # Set run_eda=True to generate all visualizations
    results, evaluations = main(run_eda=False)
