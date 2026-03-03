"""
Automated insights generation for Local Analyst.
Identifies notable patterns and anomalies in data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Insight:
    """A single data insight."""
    title: str
    description: str
    insight_type: str  # 'trend', 'anomaly', 'pattern', 'correlation'
    severity: str  # 'high', 'medium', 'low'
    metrics: Dict[str, Any]


def generate_insights_from_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    metric_name: str = "metric"
) -> List[Insight]:
    """
    Generate insights from time series data.
    
    Args:
        df: DataFrame with time series
        date_col: Date column
        value_col: Value column
        metric_name: Name of metric being analyzed
        
    Returns:
        List of Insights
    """
    insights = []
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    values = df[value_col].values
    
    # Trend insight
    if len(values) >= 7:
        recent_avg = values[-7:].mean()
        earlier_avg = values[:7].mean()
        change_pct = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg != 0 else 0
        
        if abs(change_pct) > 10:
            direction = "increasing" if change_pct > 0 else "decreasing"
            insights.append(Insight(
                title=f"{metric_name.capitalize()} is {direction}",
                description=f"Recent average ({recent_avg:.1f}) is {abs(change_pct):.1f}% {'higher' if change_pct > 0 else 'lower'} than earlier period ({earlier_avg:.1f})",
                insight_type='trend',
                severity='high' if abs(change_pct) > 30 else 'medium',
                metrics={'change_pct': change_pct, 'recent_avg': recent_avg, 'earlier_avg': earlier_avg}
            ))
    
    # Anomaly detection (simple)
    mean = values.mean()
    std = values.std()
    
    anomalies = []
    for i, val in enumerate(values):
        z_score = (val - mean) / std if std > 0 else 0
        if abs(z_score) > 2:
            anomalies.append((i, val, z_score))
    
    if anomalies:
        insights.append(Insight(
            title=f"Anomalies detected in {metric_name}",
            description=f"Found {len(anomalies)} data points significantly different from average",
            insight_type='anomaly',
            severity='high' if len(anomalies) > len(values) * 0.1 else 'low',
            metrics={'anomaly_count': len(anomalies), 'anomaly_indices': [a[0] for a in anomalies]}
        ))
    
    # Volatility insight
    if len(values) >= 30:
        recent_vol = values[-30:].std()
        earlier_vol = values[:30].std()
        
        if recent_vol > earlier_vol * 1.5:
            insights.append(Insight(
                title=f"{metric_name.capitalize()} volatility increased",
                description=f"Recent volatility ({recent_vol:.1f}) is {(recent_vol/earlier_vol-1)*100:.0f}% higher than earlier period",
                insight_type='pattern',
                severity='medium',
                metrics={'recent_volatility': recent_vol, 'earlier_volatility': earlier_vol}
            ))
    
    return insights


def generate_insights_from_segments(
    segment_data: pd.DataFrame,
    segment_col: str,
    value_col: str,
    metric_name: str = "metric"
) -> List[Insight]:
    """
    Generate insights from segmented data.
    
    Args:
        segment_data: DataFrame with segments
        segment_col: Segment column
        value_col: Value column
        metric_name: Metric name
        
    Returns:
        List of Insights
    """
    insights = []
    
    # Top performers
    top_segments = segment_data.nlargest(3, value_col)
    
    if len(top_segments) > 0:
        top_segment = top_segments.iloc[0]
        insights.append(Insight(
            title=f"Top performing segment: {top_segment[segment_col]}",
            description=f"{top_segment[segment_col]} leads with {top_segment[value_col]:.1f} {metric_name}",
            insight_type='pattern',
            severity='medium',
            metrics={'segment': top_segment[segment_col], 'value': top_segment[value_col]}
        ))
    
    # Concentration
    total = segment_data[value_col].sum()
    top_3_pct = (top_segments[value_col].sum() / total * 100) if total > 0 else 0
    
    if top_3_pct > 60:
        insights.append(Insight(
            title="High concentration in top segments",
            description=f"Top 3 segments represent {top_3_pct:.0f}% of total {metric_name}",
            insight_type='pattern',
            severity='high',
            metrics={'concentration_pct': top_3_pct}
        ))
    
    # Underperformers
    bottom_segments = segment_data.nsmallest(3, value_col)
    if len(bottom_segments) > 0:
        bottom_segment = bottom_segments.iloc[0]
        avg = segment_data[value_col].mean()
        
        if bottom_segment[value_col] < avg * 0.3:
            insights.append(Insight(
                title=f"Underperforming segment: {bottom_segment[segment_col]}",
                description=f"{bottom_segment[segment_col]} is {((avg - bottom_segment[value_col])/avg*100):.0f}% below average",
                insight_type='pattern',
                severity='medium',
                metrics={'segment': bottom_segment[segment_col], 'value': bottom_segment[value_col]}
            ))
    
    return insights


def generate_insights_from_correlation(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.7
) -> List[Insight]:
    """
    Generate insights from correlation analysis.
    
    Args:
        correlation_matrix: Correlation matrix
        threshold: Correlation threshold
        
    Returns:
        List of Insights
    """
    insights = []
    
    # Find strong correlations
    strong_corrs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr = correlation_matrix.iloc[i, j]
            
            if abs(corr) >= threshold:
                strong_corrs.append((var1, var2, corr))
    
    if strong_corrs:
        for var1, var2, corr in strong_corrs[:3]:  # Top 3
            direction = "positive" if corr > 0 else "negative"
            insights.append(Insight(
                title=f"Strong {direction} correlation",
                description=f"{var1} and {var2} are strongly correlated (r={corr:.2f})",
                insight_type='correlation',
                severity='medium',
                metrics={'var1': var1, 'var2': var2, 'correlation': corr}
            ))
    
    return insights


def prioritize_insights(insights: List[Insight]) -> List[Insight]:
    """
    Sort insights by priority.
    
    Args:
        insights: List of insights
        
    Returns:
        Sorted list (highest priority first)
    """
    severity_order = {'high': 3, 'medium': 2, 'low': 1}
    
    return sorted(
        insights,
        key=lambda x: severity_order.get(x.severity, 0),
        reverse=True
    )


# Export
__all__ = [
    'Insight',
    'generate_insights_from_timeseries',
    'generate_insights_from_segments',
    'generate_insights_from_correlation',
    'prioritize_insights'
]
