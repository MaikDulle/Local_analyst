"""
Enhanced correlation analysis that handles mixed data types.
Add this to analysis_engine/correlations.py or create as separate file.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MixedCorrelationResult:
    """Results from correlation analysis with mixed types."""
    numeric_correlations: pd.DataFrame
    categorical_associations: List[Dict]
    numeric_categorical_analysis: List[Dict]
    data_summary: Dict


def analyze_mixed_correlations(
    df: pd.DataFrame,
    max_categories: int = 10
) -> MixedCorrelationResult:
    """
    Comprehensive correlation analysis for mixed data types.
    
    Handles:
    - Numeric vs Numeric: Pearson correlation
    - Categorical vs Categorical: Cramér's V
    - Numeric vs Categorical: ANOVA / Point-biserial
    
    Args:
        df: DataFrame to analyze
        max_categories: Max unique values to treat column as categorical
        
    Returns:
        MixedCorrelationResult with all correlation types
    """
    
    # Separate columns by type
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Identify categorical columns
    categorical_cols = []
    for col in df.columns:
        if col not in numeric_cols:
            # Explicitly non-numeric
            categorical_cols.append(col)
        elif df[col].nunique() <= max_categories:
            # Numeric but few unique values - treat as categorical
            categorical_cols.append(col)
            numeric_cols.remove(col)
    
    # 1. NUMERIC vs NUMERIC - Pearson Correlation
    numeric_corr = pd.DataFrame()
    if len(numeric_cols) >= 2:
        numeric_corr = df[numeric_cols].corr(method='pearson')
    
    # 2. CATEGORICAL vs CATEGORICAL - Cramér's V
    categorical_associations = []
    if len(categorical_cols) >= 2:
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                cramers_v = calculate_cramers_v(df[col1], df[col2])
                if cramers_v > 0.1:  # Only show meaningful associations
                    categorical_associations.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'cramers_v': cramers_v,
                        'strength': interpret_cramers_v(cramers_v)
                    })
    
    # 3. NUMERIC vs CATEGORICAL - ANOVA F-statistic
    numeric_categorical = []
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            # Skip if categorical has too many categories
            if df[cat_col].nunique() > 20:
                continue
            
            # Perform ANOVA
            groups = [df[df[cat_col] == cat][num_col].dropna() 
                     for cat in df[cat_col].unique()]
            
            # Need at least 2 groups with data
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue
            
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                
                if p_value < 0.05:  # Significant relationship
                    # Calculate eta-squared (effect size)
                    grand_mean = df[num_col].mean()
                    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
                    ss_total = ((df[num_col] - grand_mean)**2).sum()
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    numeric_categorical.append({
                        'numeric_var': num_col,
                        'categorical_var': cat_col,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'eta_squared': eta_squared,
                        'strength': interpret_eta_squared(eta_squared)
                    })
            except:
                # Skip if ANOVA fails
                continue
    
    # Data summary
    summary = {
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'numeric_correlations_found': (numeric_corr.abs() > 0.3).sum().sum() // 2 if not numeric_corr.empty else 0,
        'categorical_associations_found': len(categorical_associations),
        'numeric_categorical_relationships': len(numeric_categorical)
    }
    
    return MixedCorrelationResult(
        numeric_correlations=numeric_corr,
        categorical_associations=categorical_associations,
        numeric_categorical_analysis=numeric_categorical,
        data_summary=summary
    )


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Cramér's V for categorical association.
    
    Returns value between 0 (no association) and 1 (perfect association).
    """
    # Create contingency table
    confusion_matrix = pd.crosstab(x, y)
    
    # Calculate chi-square
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    
    # Calculate Cramér's V
    min_dim = min(confusion_matrix.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    return cramers_v


def interpret_cramers_v(v: float) -> str:
    """Interpret Cramér's V strength."""
    if v > 0.5:
        return "Strong"
    elif v > 0.3:
        return "Moderate"
    elif v > 0.1:
        return "Weak"
    else:
        return "Very weak"


def interpret_eta_squared(eta_sq: float) -> str:
    """Interpret eta-squared (effect size) strength."""
    if eta_sq > 0.14:
        return "Large"
    elif eta_sq > 0.06:
        return "Medium"
    elif eta_sq > 0.01:
        return "Small"
    else:
        return "Very small"


def find_all_relationships(
    df: pd.DataFrame,
    threshold: float = 0.3
) -> Dict[str, List[Dict]]:
    """
    Find all types of relationships in data.
    
    Args:
        df: DataFrame to analyze
        threshold: Minimum correlation/association strength
        
    Returns:
        Dict with 'numeric', 'categorical', and 'mixed' relationships
    """
    result = analyze_mixed_correlations(df)
    
    # Build numeric correlation pairs — ALL pairs always included
    numeric_relationships = []
    numeric_relationships_all = []   # unfiltered, for heatmap display
    if not result.numeric_correlations.empty:
        for i in range(len(result.numeric_correlations.columns)):
            for j in range(i+1, len(result.numeric_correlations.columns)):
                col1 = result.numeric_correlations.columns[i]
                col2 = result.numeric_correlations.columns[j]
                corr = result.numeric_correlations.iloc[i, j]

                entry = {
                    'var1': col1,
                    'var2': col2,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'type': 'positive' if corr > 0 else 'negative'
                }
                numeric_relationships_all.append(entry)
                if abs(corr) >= threshold:
                    numeric_relationships.append(entry)
    
    # Filter categorical associations
    categorical_relationships = [
        assoc for assoc in result.categorical_associations 
        if assoc['cramers_v'] >= threshold
    ]
    
    # Mixed relationships (already filtered in analyze_mixed_correlations)
    mixed_relationships = result.numeric_categorical_analysis
    
    return {
        'numeric': numeric_relationships,           # filtered by threshold
        'numeric_all': numeric_relationships_all,  # all pairs, for heatmap
        'numeric_matrix': result.numeric_correlations,  # raw DataFrame
        'categorical': categorical_relationships,
        'mixed': mixed_relationships,
        'summary': result.data_summary
    }


# Export
__all__ = [
    'MixedCorrelationResult',
    'analyze_mixed_correlations',
    'calculate_cramers_v',
    'find_all_relationships'
]
