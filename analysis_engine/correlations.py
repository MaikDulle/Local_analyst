"""
Correlation and relationship analysis.
Find patterns and relationships between variables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""
    var1: str
    var2: str
    correlation: float
    p_value: Optional[float]
    strength: str
    direction: str


def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: DataFrame
        columns: Specific columns to include (defaults to all numeric)
        method: 'pearson', 'spearman', or 'kendall'
    
    Returns:
        Correlation matrix as DataFrame
    """
    if columns:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    return numeric_df.corr(method=method).round(4)


def find_strong_correlations(
    df: pd.DataFrame,
    threshold: float = 0.5,
    method: str = 'pearson'
) -> List[CorrelationResult]:
    """
    Find pairs of variables with strong correlations.
    
    Args:
        df: DataFrame
        threshold: Minimum absolute correlation to include (default 0.5)
        method: Correlation method
    
    Returns:
        List of CorrelationResult for strong correlations
    """
    corr_matrix = correlation_matrix(df, method=method)
    
    if corr_matrix.empty:
        return []
    
    results = []
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]
            
            if abs(corr) >= threshold:
                # Determine strength
                abs_corr = abs(corr)
                if abs_corr >= 0.9:
                    strength = 'very strong'
                elif abs_corr >= 0.7:
                    strength = 'strong'
                elif abs_corr >= 0.5:
                    strength = 'moderate'
                else:
                    strength = 'weak'
                
                direction = 'positive' if corr > 0 else 'negative'
                
                results.append(CorrelationResult(
                    var1=cols[i],
                    var2=cols[j],
                    correlation=round(corr, 4),
                    p_value=None,  # Would need scipy for p-values
                    strength=strength,
                    direction=direction
                ))
    
    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x.correlation), reverse=True)
    
    return results


def correlation_with_target(
    df: pd.DataFrame,
    target_col: str,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation of all numeric columns with a target variable.
    
    Args:
        df: DataFrame
        target_col: Target variable to correlate against
        method: Correlation method
    
    Returns:
        DataFrame with correlations sorted by absolute value
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' is not numeric")
    
    correlations = numeric_df.corr(method=method)[target_col].drop(target_col)
    
    result = pd.DataFrame({
        'variable': correlations.index,
        'correlation': correlations.values,
        'abs_correlation': correlations.abs().values
    })
    
    # Add interpretation
    def interpret_correlation(corr):
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = 'Strong'
        elif abs_corr >= 0.4:
            strength = 'Moderate'
        elif abs_corr >= 0.2:
            strength = 'Weak'
        else:
            strength = 'Very weak'
        
        direction = 'positive' if corr > 0 else 'negative'
        return f"{strength} {direction}"
    
    result['interpretation'] = result['correlation'].apply(interpret_correlation)
    
    return result.sort_values('abs_correlation', ascending=False)


def cross_tabulation(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: Optional[str] = None,
    aggfunc: str = 'count',
    normalize: Optional[str] = None
) -> pd.DataFrame:
    """
    Create cross-tabulation (pivot table) of two categorical variables.
    
    Args:
        df: DataFrame
        row_col: Column for rows
        col_col: Column for columns
        value_col: Column to aggregate (optional)
        aggfunc: Aggregation function ('count', 'sum', 'mean')
        normalize: 'index', 'columns', 'all', or None
    
    Returns:
        Cross-tabulation DataFrame
    """
    if value_col and value_col in df.columns:
        ct = pd.crosstab(
            df[row_col], 
            df[col_col], 
            values=df[value_col],
            aggfunc=aggfunc,
            normalize=normalize
        )
    else:
        ct = pd.crosstab(
            df[row_col],
            df[col_col],
            normalize=normalize
        )
    
    # Add totals
    if normalize is None:
        ct['Total'] = ct.sum(axis=1)
        ct.loc['Total'] = ct.sum(axis=0)
    
    return ct


def group_comparison(
    df: pd.DataFrame,
    group_col: str,
    value_col: str
) -> Dict[str, Any]:
    """
    Compare a numeric variable across groups.
    
    Args:
        df: DataFrame
        group_col: Categorical column to group by
        value_col: Numeric column to compare
    
    Returns:
        Dict with group statistics and comparison metrics
    """
    grouped = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max', 'median'])
    grouped.columns = ['count', 'mean', 'std', 'min', 'max', 'median']
    
    # Overall statistics
    overall_mean = df[value_col].mean()
    overall_std = df[value_col].std()
    
    # Deviation from overall mean
    grouped['diff_from_overall'] = grouped['mean'] - overall_mean
    grouped['diff_pct'] = (grouped['diff_from_overall'] / overall_mean * 100).round(2)
    
    # Coefficient of variation per group
    grouped['cv'] = (grouped['std'] / grouped['mean']).round(3)
    
    # Effect size (Cohen's d) relative to overall
    grouped['effect_size'] = (grouped['diff_from_overall'] / overall_std).round(3)
    
    return {
        'group_stats': grouped.round(2).reset_index(),
        'overall_mean': round(overall_mean, 2),
        'overall_std': round(overall_std, 2),
        'n_groups': len(grouped),
        'range_of_means': round(grouped['mean'].max() - grouped['mean'].min(), 2)
    }


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Detect outliers in a numeric column.
    
    Args:
        df: DataFrame
        column: Column to analyze
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: IQR multiplier (default 1.5) or z-score threshold (default 3)
    
    Returns:
        Dict with outlier information
    """
    series = df[column].dropna()
    
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        z_scores = (series - mean) / std
        outliers = series[z_scores.abs() > threshold]
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'column': column,
        'method': method,
        'threshold': threshold,
        'total_values': len(series),
        'outlier_count': len(outliers),
        'outlier_pct': round(len(outliers) / len(series) * 100, 2),
        'lower_bound': round(lower_bound, 4),
        'upper_bound': round(upper_bound, 4),
        'outlier_values': outliers.tolist() if len(outliers) <= 20 else outliers.head(20).tolist(),
        'outlier_indices': outliers.index.tolist() if len(outliers) <= 20 else outliers.head(20).index.tolist()
    }


def relationship_strength(
    df: pd.DataFrame,
    x_col: str,
    y_col: str
) -> Dict[str, Any]:
    """
    Analyze the relationship between two variables (numeric or categorical).
    
    Automatically selects appropriate method based on variable types.
    """
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
    
    result = {
        'x_col': x_col,
        'y_col': y_col,
        'x_type': 'numeric' if x_is_numeric else 'categorical',
        'y_type': 'numeric' if y_is_numeric else 'categorical'
    }
    
    if x_is_numeric and y_is_numeric:
        # Both numeric: use correlation
        corr = df[[x_col, y_col]].corr().iloc[0, 1]
        result['method'] = 'pearson_correlation'
        result['value'] = round(corr, 4)
        result['interpretation'] = _interpret_correlation(corr)
        
    elif not x_is_numeric and not y_is_numeric:
        # Both categorical: use Cramér's V
        ct = pd.crosstab(df[x_col], df[y_col])
        chi2 = _chi_square_stat(ct)
        n = ct.sum().sum()
        min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        result['method'] = 'cramers_v'
        result['value'] = round(cramers_v, 4)
        result['interpretation'] = _interpret_cramers_v(cramers_v)
        
    else:
        # Mixed: use point-biserial or eta squared
        numeric_col = x_col if x_is_numeric else y_col
        cat_col = y_col if x_is_numeric else x_col
        
        # Calculate eta squared (proportion of variance explained)
        groups = df.groupby(cat_col)[numeric_col]
        overall_mean = df[numeric_col].mean()
        
        ss_between = sum(len(g) * (g.mean() - overall_mean)**2 for _, g in groups)
        ss_total = ((df[numeric_col] - overall_mean)**2).sum()
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result['method'] = 'eta_squared'
        result['value'] = round(eta_squared, 4)
        result['interpretation'] = _interpret_eta_squared(eta_squared)
    
    return result


def _chi_square_stat(contingency_table: pd.DataFrame) -> float:
    """Calculate chi-square statistic from contingency table."""
    observed = contingency_table.values
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    
    expected = np.outer(row_totals, col_totals) / total
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.where(expected > 0, (observed - expected)**2 / expected, 0)
    
    return chi2.sum()


def _interpret_correlation(corr: float) -> str:
    """Interpret Pearson correlation coefficient."""
    abs_corr = abs(corr)
    direction = 'positive' if corr > 0 else 'negative'
    
    if abs_corr >= 0.7:
        return f"Strong {direction} relationship"
    elif abs_corr >= 0.4:
        return f"Moderate {direction} relationship"
    elif abs_corr >= 0.2:
        return f"Weak {direction} relationship"
    else:
        return "Very weak or no relationship"


def _interpret_cramers_v(v: float) -> str:
    """Interpret Cramér's V statistic."""
    if v >= 0.5:
        return "Strong association"
    elif v >= 0.3:
        return "Moderate association"
    elif v >= 0.1:
        return "Weak association"
    else:
        return "Very weak or no association"


def _interpret_eta_squared(eta2: float) -> str:
    """Interpret eta squared (effect size)."""
    if eta2 >= 0.14:
        return "Large effect size"
    elif eta2 >= 0.06:
        return "Medium effect size"
    elif eta2 >= 0.01:
        return "Small effect size"
    else:
        return "Negligible effect"


def multi_variable_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze relationship of multiple variables with a target.
    
    Args:
        df: DataFrame
        target_col: Target variable
        feature_cols: Features to analyze (defaults to all other columns)
    
    Returns:
        DataFrame with relationship strength for each feature
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    
    results = []
    
    for col in feature_cols:
        try:
            rel = relationship_strength(df, col, target_col)
            results.append({
                'feature': col,
                'method': rel['method'],
                'strength': rel['value'],
                'interpretation': rel['interpretation']
            })
        except Exception as e:
            results.append({
                'feature': col,
                'method': 'error',
                'strength': None,
                'interpretation': str(e)
            })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('strength', ascending=False, na_position='last')
    
    return result_df
