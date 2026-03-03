"""
Business-Focused Anomaly Detection for Local Analyst.

Detects THREE types of anomalies:
1. VALUE ANOMALIES - Unusual numbers (spikes, drops)
2. PATTERN ANOMALIES - Unusual co-occurrences (things that don't normally happen together)
3. SEQUENCE ANOMALIES - Break in normal patterns over time

Focus: Business-relevant anomalies only (not statistical noise)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Anomaly:
    """A detected anomaly with business context."""
    anomaly_type: str  # 'value', 'pattern', 'sequence'
    severity: str  # 'critical', 'high', 'medium', 'low'
    title: str
    description: str
    detected_value: Any
    expected_value: Any
    deviation_pct: float
    business_impact: str
    recommendations: List[str]
    context: Dict[str, Any]


# ==================== VALUE ANOMALIES ====================

def detect_value_anomalies(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    sensitivity: float = 2.5,  # Higher = less sensitive
    min_impact_pct: float = 20  # Minimum % change to be business-relevant
) -> List[Anomaly]:
    """
    Detect unusual values in numeric columns.
    
    Args:
        df: DataFrame to analyze
        numeric_cols: Columns to check (auto-detect if None)
        sensitivity: Z-score threshold (2.5 = top/bottom 1%)
        min_impact_pct: Minimum % deviation to report
        
    Returns:
        List of Anomaly objects
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    anomalies = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        if len(data) < 10:
            continue  # Need enough data
        
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            continue  # No variation
        
        # Calculate z-scores
        z_scores = np.abs((data - mean) / std)
        
        # Find outliers
        outliers = data[z_scores > sensitivity]
        
        for idx, value in outliers.items():
            # Calculate deviation percentage
            deviation_pct = abs((value - mean) / mean * 100) if mean != 0 else 0
            
            # Only report if business-meaningful (>20% deviation)
            if deviation_pct < min_impact_pct:
                continue
            
            # Determine severity
            if deviation_pct > 100:
                severity = 'critical'
            elif deviation_pct > 50:
                severity = 'high'
            elif deviation_pct > 30:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Business context
            is_spike = value > mean
            
            if is_spike:
                title = f"Unusual Spike in {col}"
                impact = f"Value is {deviation_pct:.0f}% above average - investigate for data quality or genuine business event"
                recs = [
                    "✓ Verify data accuracy - could this be a data entry error?",
                    "✓ Check for legitimate business event (promotion, seasonality, etc.)",
                    "✓ If valid, understand what drove this spike to replicate success"
                ]
            else:
                title = f"Unusual Drop in {col}"
                impact = f"Value is {deviation_pct:.0f}% below average - potential issue requiring attention"
                recs = [
                    "✓ Investigate cause immediately - could indicate a problem",
                    "✓ Check for system issues or data collection problems",
                    "✓ Review recent changes that might explain the drop"
                ]
            
            anomalies.append(Anomaly(
                anomaly_type='value',
                severity=severity,
                title=title,
                description=f"{col} = {value:.2f} (expected: {mean:.2f} ± {std:.2f})",
                detected_value=value,
                expected_value=mean,
                deviation_pct=deviation_pct,
                business_impact=impact,
                recommendations=recs,
                context={
                    'column': col,
                    'row_index': idx,
                    'z_score': z_scores[idx],
                    'is_spike': is_spike
                }
            ))
    
    return anomalies


# ==================== PATTERN ANOMALIES ====================

def detect_pattern_anomalies(
    df: pd.DataFrame,
    max_categorical_values: int = 20,
    min_support: float = 0.01  # Minimum 1% occurrence to be meaningful
) -> List[Anomaly]:
    """
    Detect unusual co-occurrences (things that rarely happen together).
    
    Example: High revenue but low session count (unusual pattern)
    
    Args:
        df: DataFrame to analyze
        max_categorical_values: Max unique values for categorical columns
        min_support: Minimum frequency to be business-relevant
        
    Returns:
        List of pattern anomalies
    """
    anomalies = []
    
    # Get categorical and numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [col for col in df.columns 
                       if df[col].nunique() <= max_categorical_values 
                       and col not in numeric_cols]
    
    # PATTERN 1: Unusual combinations of categorical variables
    if len(categorical_cols) >= 2:
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                # Create contingency table
                crosstab = pd.crosstab(df[col1], df[col2], normalize='all')
                
                # Find rare combinations that appear
                rare_combos = []
                for idx in crosstab.index:
                    for col in crosstab.columns:
                        freq = crosstab.loc[idx, col]
                        if freq > 0 and freq < min_support:
                            # This combination is rare but exists
                            count = len(df[(df[col1] == idx) & (df[col2] == col)])
                            rare_combos.append({
                                'combo': f"{col1}={idx}, {col2}={col}",
                                'frequency': freq,
                                'count': count
                            })
                
                # Report top unusual combinations
                for combo in sorted(rare_combos, key=lambda x: x['frequency'])[:3]:
                    anomalies.append(Anomaly(
                        anomaly_type='pattern',
                        severity='medium',
                        title=f"Unusual Combination: {combo['combo']}",
                        description=f"This combination occurs in only {combo['frequency']*100:.2f}% of data ({combo['count']} times)",
                        detected_value=combo['frequency'],
                        expected_value=min_support,
                        deviation_pct=(min_support - combo['frequency']) / min_support * 100,
                        business_impact="Rare pattern detected - could indicate niche segment or data quality issue",
                        recommendations=[
                            "✓ Investigate if this represents a unique customer segment",
                            "✓ Verify data quality for these specific combinations",
                            "✓ Consider if this pattern has business value (niche opportunity)"
                        ],
                        context={
                            'col1': col1,
                            'col2': col2,
                            'combination': combo['combo'],
                            'occurrences': combo['count']
                        }
                    ))
    
    # PATTERN 2: Numeric values that don't match categorical expectations
    for cat_col in categorical_cols[:5]:  # Limit to avoid too many checks
        for num_col in numeric_cols[:5]:
            # Calculate average numeric value per category
            category_avgs = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])
            
            # Find categories with unusual patterns
            for category, row in category_avgs.iterrows():
                if row['count'] < 5:  # Need enough samples
                    continue
                
                # Compare to overall average
                overall_mean = df[num_col].mean()
                overall_std = df[num_col].std()
                
                if overall_std == 0:
                    continue
                
                # Z-score for this category's average
                z_score = abs((row['mean'] - overall_mean) / overall_std)
                
                if z_score > 2.5:  # Unusual pattern
                    deviation = abs((row['mean'] - overall_mean) / overall_mean * 100)
                    
                    if deviation > 30:  # Business-meaningful
                        is_high = row['mean'] > overall_mean
                        
                        anomalies.append(Anomaly(
                            anomaly_type='pattern',
                            severity='high' if deviation > 50 else 'medium',
                            title=f"Unusual Pattern: {cat_col}={category} has {'high' if is_high else 'low'} {num_col}",
                            description=f"{num_col} for {category} is {row['mean']:.2f} vs overall average of {overall_mean:.2f}",
                            detected_value=row['mean'],
                            expected_value=overall_mean,
                            deviation_pct=deviation,
                            business_impact=f"This segment behaves very differently - {deviation:.0f}% {'above' if is_high else 'below'} average",
                            recommendations=[
                                f"✓ Deep-dive analysis on {category} segment",
                                "✓ Understand what makes this segment unique",
                                "✓ Consider targeted strategies for this group" if is_high else "✓ Investigate why performance is lower"
                            ],
                            context={
                                'category_column': cat_col,
                                'category_value': category,
                                'numeric_column': num_col,
                                'sample_size': row['count']
                            }
                        ))
    
    return anomalies


# ==================== SEQUENCE ANOMALIES ====================

def detect_sequence_anomalies(
    df: pd.DataFrame,
    date_col: str,
    metric_cols: List[str],
    window_size: int = 7,  # Look at 7-day windows
    min_change_pct: float = 30  # Minimum % change to report
) -> List[Anomaly]:
    """
    Detect breaks in normal time-series patterns.
    
    Finds: Sudden spikes, drops, trend reversals
    
    Args:
        df: DataFrame with time series data
        date_col: Date column name
        metric_cols: Metrics to analyze
        window_size: Days to compare (recent vs previous)
        min_change_pct: Minimum % change to be business-relevant
        
    Returns:
        List of sequence anomalies
    """
    anomalies = []
    
    # Ensure date column is datetime
    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')
    df_sorted = df_sorted.dropna(subset=[date_col])
    df_sorted = df_sorted.sort_values(date_col)
    
    if len(df_sorted) < window_size * 2:
        return []  # Need enough data
    
    for col in metric_cols:
        if col not in df_sorted.columns:
            continue
        
        data = df_sorted[[date_col, col]].copy()
        data = data.dropna(subset=[col])
        
        if len(data) < window_size * 2:
            continue
        
        # Compare recent window to previous window
        recent_window = data.tail(window_size)
        previous_window = data.tail(window_size * 2).head(window_size)
        
        recent_avg = recent_window[col].mean()
        previous_avg = previous_window[col].mean()
        
        if previous_avg == 0:
            continue
        
        change_pct = ((recent_avg - previous_avg) / previous_avg) * 100
        
        # Only report significant changes
        if abs(change_pct) < min_change_pct:
            continue
        
        # Determine severity
        if abs(change_pct) > 100:
            severity = 'critical'
        elif abs(change_pct) > 50:
            severity = 'high'
        else:
            severity = 'medium'
        
        is_increase = change_pct > 0
        
        # Check if this is acceleration (trend is strengthening)
        if len(data) >= window_size * 3:
            older_window = data.tail(window_size * 3).head(window_size)
            older_avg = older_window[col].mean()
            
            prev_change = ((previous_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0
            
            is_accelerating = (is_increase and change_pct > prev_change) or (not is_increase and change_pct < prev_change)
            pattern = "accelerating" if is_accelerating else "new"
        else:
            pattern = "sudden"
        
        if is_increase:
            title = f"Sudden Increase in {col}"
            impact = f"{pattern.capitalize()} {abs(change_pct):.0f}% increase detected - verify if expected"
            recs = [
                "✓ Verify data accuracy - is this a real increase?",
                "✓ Identify what changed (marketing campaign, seasonality, etc.)",
                "✓ Ensure systems can handle increased volume",
                "✓ Capitalize on positive momentum if trend is genuine"
            ]
        else:
            title = f"Sudden Decrease in {col}"
            impact = f"{pattern.capitalize()} {abs(change_pct):.0f}% decrease detected - requires immediate attention"
            recs = [
                "🚨 URGENT: Investigate cause of decline",
                "✓ Check for system issues or data collection problems",
                "✓ Review recent changes (deployments, campaigns ended, etc.)",
                "✓ Implement corrective actions immediately"
            ]
        
        anomalies.append(Anomaly(
            anomaly_type='sequence',
            severity=severity,
            title=title,
            description=f"Recent {window_size}-day average ({recent_avg:.2f}) vs previous period ({previous_avg:.2f})",
            detected_value=recent_avg,
            expected_value=previous_avg,
            deviation_pct=abs(change_pct),
            business_impact=impact,
            recommendations=recs,
            context={
                'metric': col,
                'recent_period_avg': recent_avg,
                'previous_period_avg': previous_avg,
                'change_pct': change_pct,
                'pattern': pattern,
                'recent_period_start': recent_window[date_col].min(),
                'recent_period_end': recent_window[date_col].max()
            }
        ))
    
    return anomalies


# ==================== COMPREHENSIVE ANOMALY DETECTION ====================

def detect_all_anomalies(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    sensitivity: str = 'medium'  # 'low', 'medium', 'high'
) -> Dict[str, List[Anomaly]]:
    """
    Detect all types of anomalies in one call.
    
    Args:
        df: DataFrame to analyze
        date_col: Date column (optional, for sequence anomalies)
        sensitivity: Detection sensitivity
        
    Returns:
        Dict with 'value', 'pattern', 'sequence' anomaly lists
    """
    # Set sensitivity parameters
    sensitivity_params = {
        'low': {'z_threshold': 3.0, 'min_impact': 50},
        'medium': {'z_threshold': 2.5, 'min_impact': 30},
        'high': {'z_threshold': 2.0, 'min_impact': 20}
    }
    
    params = sensitivity_params.get(sensitivity, sensitivity_params['medium'])
    
    # Detect value anomalies
    value_anomalies = detect_value_anomalies(
        df,
        sensitivity=params['z_threshold'],
        min_impact_pct=params['min_impact']
    )
    
    # Detect pattern anomalies
    pattern_anomalies = detect_pattern_anomalies(df)
    
    # Detect sequence anomalies (if date column provided)
    sequence_anomalies = []
    if date_col and date_col in df.columns:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        sequence_anomalies = detect_sequence_anomalies(
            df,
            date_col,
            numeric_cols[:10],  # Limit to avoid too many checks
            min_change_pct=params['min_impact']
        )
    
    return {
        'value': value_anomalies,
        'pattern': pattern_anomalies,
        'sequence': sequence_anomalies
    }


def prioritize_anomalies(anomalies: List[Anomaly]) -> List[Anomaly]:
    """Sort anomalies by business priority."""
    severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    
    return sorted(
        anomalies,
        key=lambda x: (
            severity_order.get(x.severity, 0),
            x.deviation_pct
        ),
        reverse=True
    )


# Export
__all__ = [
    'Anomaly',
    'detect_value_anomalies',
    'detect_pattern_anomalies',
    'detect_sequence_anomalies',
    'detect_all_anomalies',
    'prioritize_anomalies'
]
