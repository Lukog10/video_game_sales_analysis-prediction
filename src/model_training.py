# -*- coding: utf-8 -*-
"""
Model Training Module
=====================
Handles training of machine learning models for sales prediction.

What was done:
    - Random Forest Regressor training
    - XGBoost Regressor training
    - Hybrid Ensemble Model (average of RF + XGBoost)

Key methods:
    - sklearn.ensemble.RandomForestRegressor
    - xgboost.XGBRegressor
    - Ensemble averaging

Attributes involved:
    - Features: Platform, Year, Genre, Publisher, Regional Sales
    - Target: Global_Sales
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import TEST_SIZE, RANDOM_STATE, RF_PARAMS, XGB_PARAMS, TARGET_COLUMN


def prepare_features_target(df, target_column=None):
    """
    Split dataframe into features (X) and target (y).
    
    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataset.
    target_column : str, optional
        Name of target column. Defaults to TARGET_COLUMN from config.
    
    Returns
    -------
    tuple
        (X, y) - Features DataFrame and target Series
    """
    if target_column is None:
        target_column = TARGET_COLUMN
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\n[Features] X shape: {X.shape}")
    print(f"[Target] y shape: {y.shape}")
    
    return X, y


def split_data(X, y, test_size=None, random_state=None):
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float, optional
        Proportion of data for testing. Defaults to TEST_SIZE from config.
    random_state : int, optional
        Random seed. Defaults to RANDOM_STATE from config.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n[Split] Training set: {X_train.shape[0]} samples")
    print(f"[Split] Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest Regressor model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : dict, optional
        Model hyperparameters. Defaults to RF_PARAMS from config.
    
    Returns
    -------
    RandomForestRegressor
        Trained model.
    """
    if params is None:
        params = RF_PARAMS
    
    print("\n" + "-" * 40)
    print("Training Random Forest Regressor...")
    print(f"Parameters: {params}")
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    print("Random Forest training complete!")
    
    return model


def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost Regressor model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : dict, optional
        Model hyperparameters. Defaults to XGB_PARAMS from config.
    
    Returns
    -------
    XGBRegressor
        Trained model.
    """
    if params is None:
        params = XGB_PARAMS
    
    print("\n" + "-" * 40)
    print("Training XGBoost Regressor...")
    print(f"Parameters: {params}")
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    print("XGBoost training complete!")
    
    return model


def create_hybrid_predictions(rf_predictions, xgb_predictions):
    """
    Create hybrid ensemble predictions by averaging RF and XGBoost.
    
    Parameters
    ----------
    rf_predictions : np.ndarray
        Predictions from Random Forest model.
    xgb_predictions : np.ndarray
        Predictions from XGBoost model.
    
    Returns
    -------
    np.ndarray
        Averaged hybrid predictions.
    """
    print("\n" + "-" * 40)
    print("Creating Hybrid Ensemble (RF + XGBoost average)...")
    
    hybrid_pred = (rf_predictions + xgb_predictions) / 2
    
    print("Hybrid predictions created!")
    
    return hybrid_pred


def train_all_models(X_train, X_test, y_train):
    """
    Train all models and generate predictions.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target.
    
    Returns
    -------
    dict
        Dictionary containing models and their predictions.
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    # Create Hybrid predictions
    hybrid_pred = create_hybrid_predictions(rf_pred, xgb_pred)
    
    results = {
        'Random Forest': {'model': rf_model, 'predictions': rf_pred},
        'XGBoost': {'model': xgb_model, 'predictions': xgb_pred},
        'Hybrid (RF + XGB)': {'model': None, 'predictions': hybrid_pred}
    }
    
    print("\n[Training] All models trained successfully!")
    
    return results


if __name__ == "__main__":
    # Test the module
    from data_collection import load_data
    from data_preprocessing import preprocess_pipeline
    from feature_encoding import encode_categorical
    
    df = load_data()
    df_processed = preprocess_pipeline(df)
    df_encoded, _ = encode_categorical(df_processed)
    
    X, y = prepare_features_target(df_encoded)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    results = train_all_models(X_train, X_test, y_train)
    print(f"\nModels trained: {list(results.keys())}")
