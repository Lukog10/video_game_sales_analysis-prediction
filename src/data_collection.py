# -*- coding: utf-8 -*-
"""
Data Collection Module
======================
Handles loading and initial inspection of the video game sales dataset.

What was done:
    - Load dataset from CSV file
    - Display basic information about the data

Key methods:
    - pandas.read_csv() for data loading
    - DataFrame inspection methods

Attributes involved:
    - All columns from vgsales.csv (Rank, Name, Platform, Year, Genre, 
      Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales)
"""

import pandas as pd
from config import DATA_PATH


def load_data(path=None):
    """
    Load the video game sales dataset from CSV.
    
    Parameters
    ----------
    path : str, optional
        Path to the CSV file. Defaults to DATA_PATH from config.
    
    Returns
    -------
    pd.DataFrame
        Raw dataset as a pandas DataFrame.
    """
    if path is None:
        path = DATA_PATH
    
    data = pd.read_csv(path)
    return data


def display_data_info(df):
    """
    Display basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.
    """
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n--- Data Types ---")
    print(df.info())
    
    print("\n--- Numerical Summary ---")
    print(df.describe())
    
    print("\n--- Categorical Summary ---")
    print(df.describe(include=['object', 'bool']))
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())


if __name__ == "__main__":
    # Test the module
    df = load_data()
    display_data_info(df)
