"""
Data validation and type inference utilities.
Automatically detects column types and validates data quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ColumnType(Enum):
    """Semantic column types for ecom analysis."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str
    dtype: str
    semantic_type: ColumnType
    null_count: int
    null_percent: float
    unique_count: int
    unique_percent: float
    sample_values: List[Any]
    stats: Optional[Dict[str, float]] = None  # For numeric columns


@dataclass
class DataProfile:
    """Complete profile of a dataset."""
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    quality_score: float  # 0-100
    issues: List[str]
    recommendations: List[str]


def infer_column_type(series: pd.Series) -> ColumnType:
    """
    Infer semantic type of a column based on content analysis.
    Goes beyond pandas dtype to understand business meaning.
    """
    col_name = series.name.lower() if series.name else ""
    dtype = series.dtype
    
    # Drop nulls for analysis
    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnType.UNKNOWN
    
    # Check for boolean first
    unique_vals = set(non_null.unique())
    bool_patterns = [
        {True, False},
        {1, 0},
        {'yes', 'no'},
        {'ja', 'nein'},
        {'true', 'false'},
        {'y', 'n'},
        {1.0, 0.0}
    ]
    if unique_vals in bool_patterns or (len(unique_vals) <= 2 and 
        any(str(v).lower() in ['yes', 'no', 'true', 'false', 'ja', 'nein'] for v in unique_vals)):
        return ColumnType.BOOLEAN
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return ColumnType.DATETIME
    
    # Try parsing as datetime if object type
    if dtype == 'object':
        try:
            parsed = pd.to_datetime(non_null, errors='coerce')
            if parsed.notna().sum() / len(non_null) > 0.8:
                return ColumnType.DATETIME
        except:
            pass
    
    # Check for identifiers (high cardinality, often strings/ints)
    id_keywords = ['id', '_id', 'code', 'key', 'nummer', 'number', 'uuid', 'guid']
    if any(kw in col_name for kw in id_keywords):
        return ColumnType.IDENTIFIER
    
    # Check numeric types
    if pd.api.types.is_numeric_dtype(dtype):
        # Check for currency patterns in column name
        currency_keywords = ['price', 'cost', 'revenue', 'amount', 'total', 'sum', 
                            'preis', 'umsatz', 'betrag', 'euro', 'eur', 'usd', 'value']
        if any(kw in col_name for kw in currency_keywords):
            return ColumnType.CURRENCY
        
        # Check for percentage
        pct_keywords = ['rate', 'percent', 'pct', '%', 'ratio', 'quote', 'anteil']
        if any(kw in col_name for kw in pct_keywords):
            return ColumnType.PERCENTAGE
        
        # Check if values look like percentages (0-1 or 0-100 range)
        if non_null.min() >= 0 and non_null.max() <= 1:
            return ColumnType.PERCENTAGE
        if non_null.min() >= 0 and non_null.max() <= 100 and 'rate' in col_name:
            return ColumnType.PERCENTAGE
            
        return ColumnType.NUMERIC
    
    # String analysis
    if dtype == 'object':
        avg_length = non_null.astype(str).str.len().mean()
        unique_ratio = len(non_null.unique()) / len(non_null)
        
        # Long text (descriptions, comments)
        if avg_length > 50:
            return ColumnType.TEXT
        
        # High cardinality strings are likely identifiers
        if unique_ratio > 0.9 and len(non_null) > 20:
            return ColumnType.IDENTIFIER
        
        # Low cardinality = categorical
        if unique_ratio < 0.5 or len(non_null.unique()) < 20:
            return ColumnType.CATEGORICAL
        
        return ColumnType.TEXT
    
    return ColumnType.UNKNOWN


def profile_column(series: pd.Series) -> ColumnProfile:
    """Generate detailed profile for a single column."""
    non_null = series.dropna()
    total = len(series)
    null_count = series.isna().sum()
    
    # Get sample values (up to 5 unique)
    try:
        samples = list(non_null.unique()[:5]) if len(non_null) > 0 else []
        # Convert unhashable types to strings for display
        samples = [str(s) if isinstance(s, (dict, list)) else s for s in samples]
    except (TypeError, ValueError):
        # Fallback for unhashable types
        samples = [str(v) for v in non_null.head(5).tolist()]
    
    # Calculate stats for numeric columns
    stats = None
    if pd.api.types.is_numeric_dtype(series.dtype) and len(non_null) > 0:
        stats = {
            'min': float(non_null.min()),
            'max': float(non_null.max()),
            'mean': float(non_null.mean()),
            'median': float(non_null.median()),
            'std': float(non_null.std()) if len(non_null) > 1 else 0.0
        }
    
    return ColumnProfile(
        name=series.name,
        dtype=str(series.dtype),
        semantic_type=infer_column_type(series),
        null_count=null_count,
        null_percent=round(null_count / total * 100, 2) if total > 0 else 0,
        unique_count=len(non_null.unique()),
        unique_percent=round(len(non_null.unique()) / total * 100, 2) if total > 0 else 0,
        sample_values=samples,
        stats=stats
    )


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """
    Generate complete profile of a dataframe.
    Identifies quality issues and provides recommendations.
    """
    issues = []
    recommendations = []
    
    # Profile each column
    columns = {}
    for col in df.columns:
        columns[col] = profile_column(df[col])
    
    # Check for quality issues
    for name, profile in columns.items():
        # High null percentage
        if profile.null_percent > 50:
            issues.append(f"Column '{name}' has {profile.null_percent}% missing values")
        elif profile.null_percent > 20:
            recommendations.append(f"Column '{name}' has {profile.null_percent}% missing values - consider imputation")
        
        # Single value columns (no variance)
        if profile.unique_count == 1 and profile.null_count < len(df):
            issues.append(f"Column '{name}' has only one unique value - may be useless for analysis")
        
        # Potential ID column being treated as numeric
        if profile.semantic_type == ColumnType.NUMERIC and profile.unique_percent > 95:
            recommendations.append(f"Column '{name}' might be an identifier rather than a measure")
    
    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"Dataset contains {dup_count} duplicate rows ({round(dup_count/len(df)*100, 1)}%)")
    
    # Calculate quality score
    quality_score = 100
    quality_score -= len(issues) * 10  # -10 per issue
    quality_score -= len(recommendations) * 3  # -3 per recommendation
    
    # Penalize for high overall null rate
    total_nulls = df.isna().sum().sum()
    total_cells = df.size
    null_rate = total_nulls / total_cells if total_cells > 0 else 0
    quality_score -= null_rate * 30
    
    quality_score = max(0, min(100, quality_score))
    
    return DataProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        quality_score=round(quality_score, 1),
        issues=issues,
        recommendations=recommendations
    )


def validate_for_analysis(df: pd.DataFrame, required_types: List[ColumnType] = None) -> Tuple[bool, List[str]]:
    """
    Validate if dataframe is suitable for analysis.
    
    Args:
        df: DataFrame to validate
        required_types: List of required column types (e.g., need at least one DATETIME for trends)
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if df.empty:
        errors.append("Dataset is empty")
        return False, errors
    
    if len(df.columns) == 0:
        errors.append("Dataset has no columns")
        return False, errors
    
    if len(df) < 2:
        errors.append("Dataset needs at least 2 rows for analysis")
        return False, errors
    
    # Check for required types
    if required_types:
        profile = profile_dataframe(df)
        found_types = {col.semantic_type for col in profile.columns.values()}
        
        for req_type in required_types:
            if req_type not in found_types:
                errors.append(f"Analysis requires a {req_type.value} column but none found")
    
    return len(errors) == 0, errors


def suggest_ecom_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Suggest which columns map to common ecom concepts.
    Returns dict with suggested column names for each concept.
    """
    profile = profile_dataframe(df)
    
    suggestions = {
        'date_column': None,
        'revenue_column': None,
        'quantity_column': None,
        'customer_id': None,
        'product_id': None,
        'category_column': None,
        'channel_column': None,
    }
    
    for name, col_profile in profile.columns.items():
        name_lower = name.lower()
        
        # Date column
        if col_profile.semantic_type == ColumnType.DATETIME:
            if suggestions['date_column'] is None:
                suggestions['date_column'] = name
        
        # Revenue
        if col_profile.semantic_type == ColumnType.CURRENCY:
            if suggestions['revenue_column'] is None:
                suggestions['revenue_column'] = name
        elif col_profile.semantic_type == ColumnType.NUMERIC:
            if any(kw in name_lower for kw in ['revenue', 'umsatz', 'sales', 'total', 'amount']):
                suggestions['revenue_column'] = name
        
        # Quantity
        if col_profile.semantic_type == ColumnType.NUMERIC:
            if any(kw in name_lower for kw in ['qty', 'quantity', 'menge', 'anzahl', 'count', 'units']):
                suggestions['quantity_column'] = name
        
        # Customer ID
        if col_profile.semantic_type == ColumnType.IDENTIFIER:
            if any(kw in name_lower for kw in ['customer', 'kunde', 'user', 'client', 'buyer']):
                suggestions['customer_id'] = name
        
        # Product ID
        if col_profile.semantic_type == ColumnType.IDENTIFIER:
            if any(kw in name_lower for kw in ['product', 'produkt', 'item', 'sku', 'article', 'artikel']):
                suggestions['product_id'] = name
        
        # Category
        if col_profile.semantic_type == ColumnType.CATEGORICAL:
            if any(kw in name_lower for kw in ['category', 'kategorie', 'type', 'typ', 'group', 'segment']):
                suggestions['category_column'] = name
        
        # Channel
        if col_profile.semantic_type == ColumnType.CATEGORICAL:
            if any(kw in name_lower for kw in ['channel', 'kanal', 'source', 'quelle', 'medium', 'platform']):
                suggestions['channel_column'] = name
    
    return suggestions
