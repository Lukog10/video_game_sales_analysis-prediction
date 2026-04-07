# -*- coding: utf-8 -*-
"""
Configuration Module
====================
Central configuration file containing all constants, hyperparameters,
and settings for the Video Game Sales Analysis & Prediction project.
"""

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_PATH = "data/vgsales(Original data).csv"  # Update with your dataset path
MAX_YEAR = 2015  # Filter data up to this year

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================
# Target variable
TARGET_COLUMN = "Global_Sales"

# Columns to drop during preprocessing
COLUMNS_TO_DROP = ["Name"]

# Categorical columns for encoding
CATEGORICAL_COLUMNS = ["Platform", "Genre", "Publisher"]

# Sales columns for analysis
SALES_COLUMNS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]

# Regional sales columns (excluding global)
REGIONAL_SALES_COLUMNS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Train-test split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "random_state": RANDOM_STATE
}

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
FIGURE_SIZE_LARGE = (15, 10)
FIGURE_SIZE_XLARGE = (30, 15)
FIGURE_SIZE_MEDIUM = (12, 8)
