"""
Utility functions for data cleaning and preparation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def clean_numeric_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean and convert columns that look numeric but are stored as strings.
    Handles comma separators, currency symbols, percentages.
    
    Args:
        df: DataFrame to clean
        columns: Specific columns to clean (defaults to all object columns)
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Try to detect if column should be numeric
        sample = df[col].dropna().head(10)
        if len(sample) == 0:
            continue
        
        # Check if values look numeric
        sample_str = sample.astype(str)
        
        # Remove common formatting characters and check if numeric
        cleaned = sample_str.str.replace(r'[$€£,\s%]', '', regex=True)
        cleaned = cleaned.str.replace(r'^\(', '-', regex=True)  # Handle (123) as negative
        cleaned = cleaned.str.replace(r'\)$', '', regex=True)
        
        # Check if most values are numeric after cleaning
        try:
            numeric_count = pd.to_numeric(cleaned, errors='coerce').notna().sum()
            if numeric_count / len(sample) > 0.8:
                # Clean the full column
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(r'[$€£,\s]', '', regex=True)
                df[col] = df[col].str.replace(r'^\(', '-', regex=True)
                df[col] = df[col].str.replace(r'\)$', '', regex=True)
                df[col] = df[col].str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    
    return df


def auto_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically clean a DataFrame:
    - Convert string numbers to numeric
    - Parse dates
    - Strip whitespace from strings
    - Standardize column names
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Clean column names
    df.columns = (df.columns
                  .str.strip()
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.replace(r'[^\w\s$%()]', '', regex=True))
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert numeric-looking columns
    df = clean_numeric_columns(df)
    
    # Try to parse date columns
    date_keywords = ['date', 'datum', 'time', 'created', 'updated']
    for col in df.select_dtypes(include=['object']).columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() / len(df) > 0.8:
                    df[col] = parsed
            except:
                pass
    
    return df


def prepare_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for analysis by cleaning and converting data types.
    Wrapper around auto_clean_dataframe with additional checks.
    """
    # Clean the data
    df = auto_clean_dataframe(df)
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def detect_column_roles(df: pd.DataFrame) -> dict:
    """
    Detect the likely role of each column in ecom analysis.
    
    Returns:
        Dict with suggested roles for columns
    """
    roles = {
        'id_columns': [],
        'date_columns': [],
        'revenue_columns': [],
        'quantity_columns': [],
        'category_columns': [],
        'measure_columns': [],
        'dimension_columns': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        dtype = df[col].dtype
        
        # Date columns
        if pd.api.types.is_datetime64_any_dtype(dtype):
            roles['date_columns'].append(col)
            continue
        
        # ID columns
        id_keywords = ['id', '_id', 'code', 'key', 'nummer', 'number']
        if any(kw in col_lower for kw in id_keywords):
            roles['id_columns'].append(col)
            continue
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(dtype):
            # Revenue/monetary
            revenue_keywords = ['revenue', 'sales', 'amount', 'total', 'price', 'cost', 'profit', 'umsatz']
            if any(kw in col_lower for kw in revenue_keywords):
                roles['revenue_columns'].append(col)
            # Quantity
            elif any(kw in col_lower for kw in ['qty', 'quantity', 'units', 'count', 'menge']):
                roles['quantity_columns'].append(col)
            else:
                roles['measure_columns'].append(col)
        
        # Categorical columns
        elif dtype == 'object':
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_ratio < 0.5 or df[col].nunique() < 20:
                if any(kw in col_lower for kw in ['category', 'type', 'channel', 'region', 'segment']):
                    roles['category_columns'].append(col)
                else:
                    roles['dimension_columns'].append(col)
            else:
                roles['dimension_columns'].append(col)
    
    return roles
