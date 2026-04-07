# -*- coding: utf-8 -*-
"""
Data Preprocessing Module
=========================
Handles all data cleaning and preprocessing operations.

What was done:
    - Missing value removal
    - Year filtering (data up to 2015)
    - Column removal (Name column)
    - Data type corrections

Key methods:
    - dropna() for missing value removal
    - DataFrame filtering and dropping

Attributes involved:
    - Year (filtering)
    - Name (removal)
    - All columns (null handling)
"""

import pandas as pd
from config import MAX_YEAR, COLUMNS_TO_DROP


def filter_by_year(df, max_year=None):
    """
    Filter dataset to include only records up to a specified year.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    max_year : int, optional
        Maximum year to include. Defaults to MAX_YEAR from config.
    
    Returns
    -------
    pd.DataFrame
        Filtered dataset.
    """
    if max_year is None:
        max_year = MAX_YEAR
    
    drop_row_index = df[df['Year'] > max_year].index
    df_filtered = df.drop(drop_row_index)
    
    print(f"[Preprocessing] Filtered data to year <= {max_year}")
    print(f"               Shape after filtering: {df_filtered.shape}")
    
    return df_filtered


def remove_missing_values(df):
    """
    Remove rows with missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    
    Returns
    -------
    pd.DataFrame
        Dataset with missing values removed.
    """
    initial_rows = len(df)
    df_clean = df.dropna()
    removed_rows = initial_rows - len(df_clean)
    
    print(f"[Preprocessing] Removed {removed_rows} rows with missing values")
    print(f"               Shape after cleaning: {df_clean.shape}")
    
    return df_clean


def drop_columns(df, columns=None):
    """
    Drop specified columns from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list, optional
        Columns to drop. Defaults to COLUMNS_TO_DROP from config.
    
    Returns
    -------
    pd.DataFrame
        Dataset with specified columns removed.
    """
    if columns is None:
        columns = COLUMNS_TO_DROP
    
    df_reduced = df.drop(columns=columns)
    
    print(f"[Preprocessing] Dropped columns: {columns}")
    print(f"               Remaining columns: {list(df_reduced.columns)}")
    
    return df_reduced


def preprocess_pipeline(df):
    """
    Execute the complete preprocessing pipeline.
    
    Pipeline Steps:
        1. Filter by year
        2. Remove missing values
        3. Drop non-useful columns
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.
    
    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataset.
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Filter by year
    df = filter_by_year(df)
    
    # Step 2: Remove missing values
    df = remove_missing_values(df)
    
    # Step 3: Drop non-useful columns
    df = drop_columns(df)
    
    print("\n[Preprocessing] Pipeline complete!")
    print(f"               Final shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Test the module
    from data_collection import load_data
    
    df = load_data()
    df_processed = preprocess_pipeline(df)
    print("\nProcessed Data Sample:")
    print(df_processed.head())
