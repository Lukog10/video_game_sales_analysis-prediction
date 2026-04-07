# -*- coding: utf-8 -*-
"""
Feature Encoding Module
=======================
Handles categorical feature encoding for machine learning models.

What was done:
    - Label Encoding of categorical features
    - Platform → Numeric
    - Genre → Numeric
    - Publisher → Numeric

Key methods:
    - sklearn.preprocessing.LabelEncoder

Attributes involved:
    - Platform, Genre, Publisher (categorical to numeric)
"""

from sklearn.preprocessing import LabelEncoder
from config import CATEGORICAL_COLUMNS


def encode_categorical(df, columns=None):
    """
    Apply Label Encoding to categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with categorical columns.
    columns : list, optional
        Columns to encode. Defaults to CATEGORICAL_COLUMNS from config.
    
    Returns
    -------
    tuple
        (encoded_df, encoders_dict)
        - encoded_df: DataFrame with encoded columns
        - encoders_dict: Dictionary of fitted LabelEncoder objects
    """
    if columns is None:
        columns = CATEGORICAL_COLUMNS
    
    df_encoded = df.copy()
    encoders = {}
    
    print("\n" + "=" * 60)
    print("FEATURE ENCODING")
    print("=" * 60)
    
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        print(f"[Encoding] {col}: {len(le.classes_)} unique values → Numeric (0 to {len(le.classes_) - 1})")
    
    print("\n[Encoding] Complete!")
    
    return df_encoded, encoders


def get_encoding_mapping(encoder, column_name):
    """
    Get the mapping from original values to encoded values.
    
    Parameters
    ----------
    encoder : LabelEncoder
        Fitted LabelEncoder object.
    column_name : str
        Name of the encoded column.
    
    Returns
    -------
    dict
        Mapping from original values to encoded integers.
    """
    mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
    return mapping


if __name__ == "__main__":
    # Test the module
    from data_collection import load_data
    from data_preprocessing import preprocess_pipeline
    
    df = load_data()
    df_processed = preprocess_pipeline(df)
    df_encoded, encoders = encode_categorical(df_processed)
    
    print("\nEncoded Data Sample:")
    print(df_encoded.head())
    
    print("\nEncoding Mappings:")
    for col, enc in encoders.items():
        print(f"\n{col}: {list(enc.classes_[:5])}...")  # Show first 5
