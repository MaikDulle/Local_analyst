"""
Campaign performance tracking for Local Analyst.
Year-over-year comparisons, period analysis, and campaign metrics.
Integrates with existing analysis_engine pattern.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CampaignMetrics:
    """Campaign performance metrics."""
    period: str
    impressions: int
    clicks: int
    conversions: int
    revenue: float
    cost: float
    ctr: float  # Click-through rate
    cvr: float  # Conversion rate
    cpc: float  # Cost per click
    cpa: float  # Cost per acquisition
    roas: float  # Return on ad spend


def campaign_performance_summary(
    df: pd.DataFrame,
    date_col: str,
    metric_cols: List[str],
    campaign_col: Optional[str] = None,
    channel_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate campaign performance summary.
    
    Args:
        df: Campaign data DataFrame
        date_col: Date column name
        metric_cols: List of metric columns to summarize
        campaign_col: Optional campaign name column
        channel_col: Optional channel/source column
        
    Returns:
        Summary DataFrame with key metrics
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    group_cols = []
    if campaign_col and campaign_col in df.columns:
        group_cols.append(campaign_col)
    if channel_col and channel_col in df.columns:
        group_cols.append(channel_col)
    
    if not group_cols:
        # Overall summary
        summary = df[metric_cols].agg(['sum', 'mean', 'median', 'std', 'min', 'max'])
        return summary.T
    
    # Group by campaign/channel
    summary = df.groupby(group_cols)[metric_cols].agg(['sum', 'mean', 'count'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    return summary.reset_index()


def year_over_year_comparison(
    df: pd.DataFrame,
    date_col: str,
    metric_cols: List[str],
    current_year: int,
    previous_year: int,
    period: str = 'M'  # M=monthly, W=weekly, D=daily
) -> pd.DataFrame:
    """
    Compare metrics year-over-year.
    
    Args:
        df: Campaign data
        date_col: Date column
        metric_cols: Metrics to compare
        current_year: Current year to analyze
        previous_year: Previous year to compare against
        period: Aggregation period ('D', 'W', 'M', 'Q')
        
    Returns:
        DataFrame with YoY comparison
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['period'] = df[date_col].dt.to_period(period)
    
    # Filter to relevant years
    current_data = df[df['year'] == current_year].copy()
    previous_data = df[df['year'] == previous_year].copy()
    
    # Aggregate by period
    current_agg = current_data.groupby('period')[metric_cols].sum()
    previous_agg = previous_data.groupby('period')[metric_cols].sum()
    
    # Align periods (same month/week/day in different years)
    current_agg.index = current_agg.index.to_timestamp()
    previous_agg.index = previous_agg.index.to_timestamp()
    
    # Create comparison
    comparison = pd.DataFrame()
    
    for col in metric_cols:
        comparison[f'{col}_current'] = current_agg[col]
        comparison[f'{col}_previous'] = previous_agg[col]
        comparison[f'{col}_change'] = current_agg[col] - previous_agg[col]
        comparison[f'{col}_change_pct'] = (
            (current_agg[col] - previous_agg[col]) / previous_agg[col] * 100
        ).replace([np.inf, -np.inf], 0).fillna(0)
    
    return comparison.reset_index()


def wave_season_comparison(
    df: pd.DataFrame,
    date_col: str,
    metric_cols: List[str],
    wave_start_1: str,
    wave_end_1: str,
    wave_start_2: str,
    wave_end_2: str,
    wave_1_name: str = "Wave 2024/25",
    wave_2_name: str = "Wave 2025/26"
) -> Dict[str, Any]:
    """
    Compare two wave seasons (like Wave Season 2024/25 vs 2025/26).
    
    Designed specifically for seasonal campaign tracking like TUI's Wave Season.
    
    Args:
        df: Campaign data
        date_col: Date column
        metric_cols: Metrics to compare
        wave_start_1: Start date of first wave (YYYY-MM-DD)
        wave_end_1: End date of first wave
        wave_start_2: Start date of second wave
        wave_end_2: End date of second wave
        wave_1_name: Name for first wave
        wave_2_name: Name for second wave
        
    Returns:
        Dict with detailed comparison
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter to wave periods
    wave_1 = df[(df[date_col] >= wave_start_1) & (df[date_col] <= wave_end_1)]
    wave_2 = df[(df[date_col] >= wave_start_2) & (df[date_col] <= wave_end_2)]
    
    # Calculate totals
    totals_1 = wave_1[metric_cols].sum()
    totals_2 = wave_2[metric_cols].sum()
    
    # Daily averages
    days_1 = (pd.to_datetime(wave_end_1) - pd.to_datetime(wave_start_1)).days + 1
    days_2 = (pd.to_datetime(wave_end_2) - pd.to_datetime(wave_start_2)).days + 1
    
    daily_avg_1 = totals_1 / days_1
    daily_avg_2 = totals_2 / days_2
    
    # Calculate changes
    absolute_change = totals_2 - totals_1
    percent_change = ((totals_2 - totals_1) / totals_1 * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    return {
        'wave_1_name': wave_1_name,
        'wave_2_name': wave_2_name,
        'wave_1_days': days_1,
        'wave_2_days': days_2,
        'wave_1_totals': totals_1.to_dict(),
        'wave_2_totals': totals_2.to_dict(),
        'wave_1_daily_avg': daily_avg_1.to_dict(),
        'wave_2_daily_avg': daily_avg_2.to_dict(),
        'absolute_change': absolute_change.to_dict(),
        'percent_change': percent_change.to_dict()
    }


def calculate_campaign_kpis(
    df: pd.DataFrame,
    impressions_col: Optional[str] = None,
    clicks_col: Optional[str] = None,
    conversions_col: Optional[str] = None,
    revenue_col: Optional[str] = None,
    cost_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate derived KPIs from campaign data.
    
    Args:
        df: Campaign data with raw metrics
        impressions_col: Impressions column name
        clicks_col: Clicks column name
        conversions_col: Conversions column name
        revenue_col: Revenue column name
        cost_col: Cost column name
        
    Returns:
        DataFrame with additional KPI columns
    """
    df = df.copy()
    
    # CTR (Click-through rate)
    if impressions_col and clicks_col:
        df['CTR'] = (df[clicks_col] / df[impressions_col] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # CVR (Conversion rate)
    if clicks_col and conversions_col:
        df['CVR'] = (df[conversions_col] / df[clicks_col] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # CPC (Cost per click)
    if cost_col and clicks_col:
        df['CPC'] = (df[cost_col] / df[clicks_col]).replace([np.inf, -np.inf], 0).fillna(0)
    
    # CPA (Cost per acquisition)
    if cost_col and conversions_col:
        df['CPA'] = (df[cost_col] / df[conversions_col]).replace([np.inf, -np.inf], 0).fillna(0)
    
    # ROAS (Return on ad spend)
    if revenue_col and cost_col:
        df['ROAS'] = (df[revenue_col] / df[cost_col]).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Revenue per click
    if revenue_col and clicks_col:
        df['Revenue_per_Click'] = (df[revenue_col] / df[clicks_col]).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def identify_top_campaigns(
    df: pd.DataFrame,
    campaign_col: str,
    metric_col: str,
    top_n: int = 10,
    min_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Identify top-performing campaigns.
    
    Args:
        df: Campaign data
        campaign_col: Campaign identifier column
        metric_col: Metric to rank by (e.g., 'revenue', 'conversions')
        top_n: Number of top campaigns to return
        min_threshold: Optional minimum value for metric
        
    Returns:
        DataFrame with top campaigns
    """
    campaign_totals = df.groupby(campaign_col)[metric_col].sum().sort_values(ascending=False)
    
    if min_threshold:
        campaign_totals = campaign_totals[campaign_totals >= min_threshold]
    
    top_campaigns = campaign_totals.head(top_n)
    
    # Calculate percentage of total
    total = campaign_totals.sum()
    result = pd.DataFrame({
        'campaign': top_campaigns.index,
        'total': top_campaigns.values,
        'percentage_of_total': (top_campaigns.values / total * 100).round(2)
    })
    
    return result.reset_index(drop=True)


def campaign_performance_by_device(
    df: pd.DataFrame,
    device_col: str,
    metric_cols: List[str]
) -> pd.DataFrame:
    """
    Analyze campaign performance by device type.
    
    Args:
        df: Campaign data
        device_col: Device column (Desktop, Mobile, Tablet)
        metric_cols: Metrics to analyze
        
    Returns:
        DataFrame with device-level metrics
    """
    device_performance = df.groupby(device_col)[metric_cols].sum()
    
    # Calculate percentages
    for col in metric_cols:
        total = device_performance[col].sum()
        device_performance[f'{col}_pct'] = (device_performance[col] / total * 100).round(2)
    
    return device_performance.reset_index()


def detect_campaign_anomalies(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    std_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalies in campaign performance using statistical methods.
    
    Args:
        df: Campaign data
        date_col: Date column
        metric_col: Metric to check for anomalies
        std_threshold: Number of standard deviations for anomaly detection
        
    Returns:
        DataFrame with anomalies flagged
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Calculate rolling statistics
    df['rolling_mean'] = df[metric_col].rolling(window=7, min_periods=1).mean()
    df['rolling_std'] = df[metric_col].rolling(window=7, min_periods=1).std()
    
    # Calculate z-score
    df['z_score'] = ((df[metric_col] - df['rolling_mean']) / df['rolling_std']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Flag anomalies
    df['is_anomaly'] = abs(df['z_score']) > std_threshold
    df['anomaly_type'] = np.where(
        df['z_score'] > std_threshold, 'spike',
        np.where(df['z_score'] < -std_threshold, 'drop', 'normal')
    )
    
    return df[[date_col, metric_col, 'rolling_mean', 'z_score', 'is_anomaly', 'anomaly_type']]


# Example usage - Wave Season comparison
if __name__ == "__main__":
    # Generate sample Wave Season data
    np.random.seed(42)
    
    # Wave 2024/25
    dates_2024 = pd.date_range('2024-11-01', '2025-01-31')
    wave_2024 = pd.DataFrame({
        'date': dates_2024,
        'impressions': np.random.poisson(50000, len(dates_2024)),
        'clicks': np.random.poisson(1500, len(dates_2024)),
        'conversions': np.random.poisson(45, len(dates_2024)),
        'revenue': np.random.normal(5000, 1000, len(dates_2024))
    })
    
    # Wave 2025/26 (with slight improvement)
    dates_2025 = pd.date_range('2025-11-01', '2026-01-31')
    wave_2025 = pd.DataFrame({
        'date': dates_2025,
        'impressions': np.random.poisson(52000, len(dates_2025)),
        'clicks': np.random.poisson(1600, len(dates_2025)),
        'conversions': np.random.poisson(50, len(dates_2025)),
        'revenue': np.random.normal(5500, 1000, len(dates_2025))
    })
    
    df = pd.concat([wave_2024, wave_2025], ignore_index=True)
    
    # Compare waves
    comparison = wave_season_comparison(
        df,
        date_col='date',
        metric_cols=['impressions', 'clicks', 'conversions', 'revenue'],
        wave_start_1='2024-11-01',
        wave_end_1='2025-01-31',
        wave_start_2='2025-11-01',
        wave_end_2='2026-01-31'
    )
    
    print("Wave Season Comparison:")
    print(f"{comparison['wave_1_name']} vs {comparison['wave_2_name']}")
    print("\nPercent Changes:")
    for metric, change in comparison['percent_change'].items():
        print(f"  {metric}: {change:+.2f}%")
