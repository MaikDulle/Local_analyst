"""
Summary statistics and dataset overview.
Provides quick insights into any dataset - the foundation for all other analyses.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class NumericSummary:
    """Summary statistics for a numeric column."""
    column: str
    count: int
    missing: int
    missing_pct: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    has_outliers: bool
    outlier_count: int
    sum: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'column': self.column,
            'count': self.count,
            'missing': self.missing,
            'missing_pct': self.missing_pct,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'q25': self.q25,
            'q75': self.q75,
            'iqr': self.iqr,
            'skewness': self.skewness,
            'has_outliers': self.has_outliers,
            'outlier_count': self.outlier_count,
            'sum': self.sum
        }


@dataclass
class CategoricalSummary:
    """Summary statistics for a categorical column."""
    column: str
    count: int
    missing: int
    missing_pct: float
    unique_count: int
    top_value: Any
    top_count: int
    top_pct: float
    value_counts: Dict[Any, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'column': self.column,
            'count': self.count,
            'missing': self.missing,
            'missing_pct': self.missing_pct,
            'unique_count': self.unique_count,
            'top_value': self.top_value,
            'top_count': self.top_count,
            'top_pct': self.top_pct,
            'value_counts': self.value_counts
        }


@dataclass
class DatasetSummary:
    """Complete summary of a dataset."""
    row_count: int
    column_count: int
    memory_mb: float
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    numeric_summaries: Dict[str, NumericSummary]
    categorical_summaries: Dict[str, CategoricalSummary]
    datetime_range: Optional[Dict[str, Any]]
    total_missing: int
    total_missing_pct: float
    duplicate_rows: int
    insights: List[str] = field(default_factory=list)


def summarize_numeric(series: pd.Series) -> NumericSummary:
    """Generate summary statistics for a numeric column."""
    non_null = series.dropna()
    total = len(series)
    missing = series.isna().sum()
    
    if len(non_null) == 0:
        return NumericSummary(
            column=series.name,
            count=0, missing=missing, missing_pct=100.0,
            mean=0, median=0, std=0, min=0, max=0,
            q25=0, q75=0, iqr=0, skewness=0,
            has_outliers=False, outlier_count=0, sum=0
        )
    
    q25 = float(non_null.quantile(0.25))
    q75 = float(non_null.quantile(0.75))
    iqr = q75 - q25
    
    # Detect outliers using IQR method
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
    
    # Calculate skewness
    try:
        skewness = float(non_null.skew())
    except:
        skewness = 0.0
    
    return NumericSummary(
        column=series.name,
        count=len(non_null),
        missing=missing,
        missing_pct=round(missing / total * 100, 2) if total > 0 else 0,
        mean=round(float(non_null.mean()), 4),
        median=round(float(non_null.median()), 4),
        std=round(float(non_null.std()), 4) if len(non_null) > 1 else 0,
        min=round(float(non_null.min()), 4),
        max=round(float(non_null.max()), 4),
        q25=round(q25, 4),
        q75=round(q75, 4),
        iqr=round(iqr, 4),
        skewness=round(skewness, 4),
        has_outliers=len(outliers) > 0,
        outlier_count=len(outliers),
        sum=round(float(non_null.sum()), 4)
    )


def summarize_categorical(series: pd.Series, top_n: int = 10) -> CategoricalSummary:
    """Generate summary statistics for a categorical column."""
    non_null = series.dropna()
    total = len(series)
    missing = series.isna().sum()
    
    if len(non_null) == 0:
        return CategoricalSummary(
            column=series.name,
            count=0, missing=missing, missing_pct=100.0,
            unique_count=0, top_value=None, top_count=0, top_pct=0,
            value_counts={}
        )
    
    value_counts = non_null.value_counts()
    top_value = value_counts.index[0] if len(value_counts) > 0 else None
    top_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
    
    return CategoricalSummary(
        column=series.name,
        count=len(non_null),
        missing=missing,
        missing_pct=round(missing / total * 100, 2) if total > 0 else 0,
        unique_count=len(value_counts),
        top_value=top_value,
        top_count=top_count,
        top_pct=round(top_count / len(non_null) * 100, 2) if len(non_null) > 0 else 0,
        value_counts=dict(value_counts.head(top_n))
    )


def get_datetime_range(df: pd.DataFrame, datetime_cols: List[str]) -> Optional[Dict[str, Any]]:
    """Get the date range for datetime columns."""
    if not datetime_cols:
        return None
    
    # Use the first datetime column
    col = datetime_cols[0]
    series = pd.to_datetime(df[col], errors='coerce').dropna()
    
    if len(series) == 0:
        return None
    
    return {
        'column': col,
        'min': series.min().isoformat(),
        'max': series.max().isoformat(),
        'range_days': (series.max() - series.min()).days,
        'unique_dates': series.dt.date.nunique()
    }


def generate_insights(summary: DatasetSummary) -> List[str]:
    """Generate automatic insights from the summary."""
    insights = []
    
    # Dataset size insight
    if summary.row_count < 100:
        insights.append(f"⚠️ Small dataset ({summary.row_count} rows) - statistical significance may be limited")
    elif summary.row_count > 100000:
        insights.append(f"📊 Large dataset ({summary.row_count:,} rows) - consider sampling for exploratory analysis")
    
    # Missing data insights
    if summary.total_missing_pct > 20:
        insights.append(f"⚠️ High missing data rate ({summary.total_missing_pct:.1f}%) - data quality may affect analysis")
    
    # Duplicate insights
    if summary.duplicate_rows > 0:
        dup_pct = summary.duplicate_rows / summary.row_count * 100
        insights.append(f"🔄 Found {summary.duplicate_rows} duplicate rows ({dup_pct:.1f}%)")
    
    # Numeric column insights
    for name, num_summary in summary.numeric_summaries.items():
        # High outlier count
        if num_summary.outlier_count > 0:
            outlier_pct = num_summary.outlier_count / num_summary.count * 100 if num_summary.count > 0 else 0
            if outlier_pct > 5:
                insights.append(f"📈 '{name}' has {num_summary.outlier_count} outliers ({outlier_pct:.1f}%)")
        
        # High skewness
        if abs(num_summary.skewness) > 2:
            direction = "right" if num_summary.skewness > 0 else "left"
            insights.append(f"📊 '{name}' is highly skewed {direction} (skewness: {num_summary.skewness:.2f})")
        
        # High missing rate for specific column
        if num_summary.missing_pct > 30:
            insights.append(f"⚠️ '{name}' has {num_summary.missing_pct:.1f}% missing values")
    
    # Categorical column insights
    for name, cat_summary in summary.categorical_summaries.items():
        # Dominant category
        if cat_summary.top_pct > 80:
            insights.append(f"📌 '{name}' is dominated by '{cat_summary.top_value}' ({cat_summary.top_pct:.1f}%)")
        
        # High cardinality
        if cat_summary.unique_count > 100:
            insights.append(f"🔢 '{name}' has high cardinality ({cat_summary.unique_count} unique values)")
    
    # Date range insight
    if summary.datetime_range:
        days = summary.datetime_range['range_days']
        if days < 7:
            insights.append(f"📅 Data spans only {days} days - limited for trend analysis")
        elif days > 365:
            years = days / 365
            insights.append(f"📅 Data spans {years:.1f} years - good for seasonal analysis")
    
    return insights


def summarize_dataset(
    df: pd.DataFrame,
    include_value_counts: bool = True,
    top_n_categories: int = 10
) -> DatasetSummary:
    """
    Generate comprehensive summary of a dataset.
    
    Args:
        df: DataFrame to summarize
        include_value_counts: Whether to include value counts for categorical columns
        top_n_categories: Number of top categories to include in value counts
    
    Returns:
        DatasetSummary with all statistics and insights
    """
    if df.empty:
        return DatasetSummary(
            row_count=0, column_count=0, memory_mb=0,
            numeric_columns=[], categorical_columns=[], datetime_columns=[],
            numeric_summaries={}, categorical_summaries={},
            datetime_range=None, total_missing=0, total_missing_pct=0,
            duplicate_rows=0, insights=["Dataset is empty"]
        )
    
    # Classify columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Generate summaries
    numeric_summaries = {col: summarize_numeric(df[col]) for col in numeric_cols}
    categorical_summaries = {col: summarize_categorical(df[col], top_n_categories) for col in categorical_cols}
    
    # Calculate totals
    total_missing = df.isna().sum().sum()
    total_cells = df.size
    
    # Create summary
    summary = DatasetSummary(
        row_count=len(df),
        column_count=len(df.columns),
        memory_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        numeric_summaries=numeric_summaries,
        categorical_summaries=categorical_summaries,
        datetime_range=get_datetime_range(df, datetime_cols),
        total_missing=total_missing,
        total_missing_pct=round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
        duplicate_rows=df.duplicated().sum()
    )
    
    # Generate insights
    summary.insights = generate_insights(summary)
    
    return summary


def quick_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick one-liner statistics for a dataset.
    Useful for dashboards or quick overviews.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'missing_pct': round(df.isna().sum().sum() / df.size * 100, 2) if df.size > 0 else 0,
        'duplicates': df.duplicated().sum(),
    }
    
    # Add totals for numeric columns (useful for revenue, quantity, etc.)
    for col in numeric_cols:
        col_clean = col.lower().replace(' ', '_').replace('($)', '').replace('(%)', '').strip('_')
        stats[f'total_{col_clean}'] = df[col].sum()
        stats[f'avg_{col_clean}'] = df[col].mean()
    
    return stats


def compare_periods(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: str = 'M'  # M=month, W=week, Q=quarter, Y=year
) -> pd.DataFrame:
    """
    Compare values across time periods.
    
    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column to compare
        period: Period for grouping (M, W, Q, Y)
    
    Returns:
        DataFrame with period comparisons
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Group by period
    grouped = df.groupby(df[date_col].dt.to_period(period))[value_col].agg(['sum', 'mean', 'count'])
    grouped.columns = ['total', 'average', 'count']
    
    # Calculate period-over-period change
    grouped['prev_total'] = grouped['total'].shift(1)
    grouped['change'] = grouped['total'] - grouped['prev_total']
    grouped['change_pct'] = (grouped['change'] / grouped['prev_total'] * 100).round(2)
    
    # Reset index for cleaner output
    grouped = grouped.reset_index()
    grouped[date_col] = grouped[date_col].astype(str)
    
    return grouped


def top_performers(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Get top/bottom performers by a grouping column.
    
    Args:
        df: DataFrame
        group_col: Column to group by (e.g., 'product', 'region')
        value_col: Column to rank by (e.g., 'revenue', 'quantity')
        n: Number of results
        ascending: If True, return bottom performers
    
    Returns:
        DataFrame with top performers and their stats
    """
    grouped = df.groupby(group_col)[value_col].agg(['sum', 'mean', 'count'])
    grouped.columns = ['total', 'average', 'count']
    
    # Calculate share of total
    total_sum = grouped['total'].sum()
    grouped['share_pct'] = (grouped['total'] / total_sum * 100).round(2)
    
    # Sort and get top n
    grouped = grouped.sort_values('total', ascending=ascending).head(n)
    
    # Add rank
    grouped['rank'] = range(1, len(grouped) + 1)
    
    return grouped.reset_index()


def distribution_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Get distribution statistics for a numeric series.
    Useful for understanding value spread.
    """
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return {'error': 'No valid data'}
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    
    return {
        'count': len(non_null),
        'mean': round(float(non_null.mean()), 2),
        'std': round(float(non_null.std()), 2) if len(non_null) > 1 else 0,
        'min': round(float(non_null.min()), 2),
        'max': round(float(non_null.max()), 2),
        'range': round(float(non_null.max() - non_null.min()), 2),
        'percentiles': {f'p{p}': round(float(non_null.quantile(p/100)), 2) for p in percentiles},
        'skewness': round(float(non_null.skew()), 2),
        'kurtosis': round(float(non_null.kurtosis()), 2) if len(non_null) > 3 else 0
    }
