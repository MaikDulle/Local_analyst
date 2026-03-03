"""
Revenue and sales analysis for ecom.
Covers totals, trends, period comparisons, and growth metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RevenueMetrics:
    """Core revenue metrics."""
    total_revenue: float
    avg_order_value: float
    total_orders: int
    total_quantity: int
    revenue_per_unit: float
    period_start: Optional[str]
    period_end: Optional[str]


def calculate_revenue_metrics(
    df: pd.DataFrame,
    revenue_col: str,
    quantity_col: Optional[str] = None,
    order_id_col: Optional[str] = None,
    date_col: Optional[str] = None
) -> RevenueMetrics:
    """
    Calculate basic revenue metrics.
    
    Args:
        df: DataFrame with transaction data
        revenue_col: Column containing revenue values
        quantity_col: Column containing quantity (optional)
        order_id_col: Column containing order IDs (optional)
        date_col: Column containing dates (optional)
    """
    total_revenue = df[revenue_col].sum()
    
    # Calculate order count
    if order_id_col and order_id_col in df.columns:
        total_orders = df[order_id_col].nunique()
    else:
        total_orders = len(df)
    
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    # Calculate quantity metrics
    if quantity_col and quantity_col in df.columns:
        total_quantity = df[quantity_col].sum()
        revenue_per_unit = total_revenue / total_quantity if total_quantity > 0 else 0
    else:
        total_quantity = total_orders
        revenue_per_unit = avg_order_value
    
    # Get date range
    period_start = None
    period_end = None
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if len(dates) > 0:
            period_start = dates.min().strftime('%Y-%m-%d')
            period_end = dates.max().strftime('%Y-%m-%d')
    
    return RevenueMetrics(
        total_revenue=round(total_revenue, 2),
        avg_order_value=round(avg_order_value, 2),
        total_orders=int(total_orders),
        total_quantity=int(total_quantity),
        revenue_per_unit=round(revenue_per_unit, 2),
        period_start=period_start,
        period_end=period_end
    )


def revenue_by_period(
    df: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    period: str = 'M',
    include_growth: bool = True
) -> pd.DataFrame:
    """
    Aggregate revenue by time period with growth calculations.
    
    Args:
        df: DataFrame with transaction data
        date_col: Column containing dates
        revenue_col: Column containing revenue
        period: 'D'=day, 'W'=week, 'M'=month, 'Q'=quarter, 'Y'=year
        include_growth: Whether to calculate period-over-period growth
    
    Returns:
        DataFrame with period, revenue, and growth metrics
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, revenue_col])
    
    # Group by period
    grouped = df.groupby(df[date_col].dt.to_period(period)).agg({
        revenue_col: ['sum', 'mean', 'count']
    })
    grouped.columns = ['revenue', 'avg_transaction', 'transactions']
    grouped = grouped.reset_index()
    grouped[date_col] = grouped[date_col].astype(str)
    
    if include_growth and len(grouped) > 1:
        # Period-over-period growth
        grouped['prev_revenue'] = grouped['revenue'].shift(1)
        grouped['growth'] = grouped['revenue'] - grouped['prev_revenue']
        grouped['growth_pct'] = (grouped['growth'] / grouped['prev_revenue'] * 100).round(2)
        
        # Cumulative revenue
        grouped['cumulative_revenue'] = grouped['revenue'].cumsum()
        
        # Moving average (3 periods)
        grouped['ma_3'] = grouped['revenue'].rolling(window=3, min_periods=1).mean().round(2)
    
    return grouped


def revenue_by_dimension(
    df: pd.DataFrame,
    dimension_col: str,
    revenue_col: str,
    quantity_col: Optional[str] = None,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Break down revenue by a dimension (region, product, channel, etc.).
    
    Args:
        df: DataFrame with transaction data
        dimension_col: Column to group by
        revenue_col: Column containing revenue
        quantity_col: Column containing quantity (optional)
        top_n: Limit to top N results (optional)
    
    Returns:
        DataFrame with dimension breakdown
    """
    agg_dict = {revenue_col: 'sum'}
    if quantity_col and quantity_col in df.columns:
        agg_dict[quantity_col] = 'sum'
    
    grouped = df.groupby(dimension_col).agg(agg_dict)
    grouped.columns = ['revenue'] + (['quantity'] if quantity_col else [])
    
    # Calculate metrics
    grouped['transactions'] = df.groupby(dimension_col).size()
    grouped['avg_transaction'] = (grouped['revenue'] / grouped['transactions']).round(2)
    
    # Share of total
    total_revenue = grouped['revenue'].sum()
    grouped['share_pct'] = (grouped['revenue'] / total_revenue * 100).round(2)
    
    # Cumulative share (for Pareto analysis)
    grouped = grouped.sort_values('revenue', ascending=False)
    grouped['cumulative_share_pct'] = grouped['share_pct'].cumsum().round(2)
    
    # Add rank
    grouped['rank'] = range(1, len(grouped) + 1)
    
    if top_n:
        grouped = grouped.head(top_n)
    
    return grouped.reset_index()


def compare_periods_yoy(
    df: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    current_year: Optional[int] = None,
    previous_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Year-over-year comparison by month.
    
    Args:
        df: DataFrame with transaction data
        date_col: Column containing dates
        revenue_col: Column containing revenue
        current_year: Year to compare (defaults to latest in data)
        previous_year: Year to compare against (defaults to current - 1)
    
    Returns:
        DataFrame with YoY comparison by month
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    
    # Determine years to compare
    if current_year is None:
        current_year = df['year'].max()
    if previous_year is None:
        previous_year = current_year - 1
    
    # Filter to relevant years
    df_filtered = df[df['year'].isin([current_year, previous_year])]
    
    # Pivot to get years as columns
    monthly = df_filtered.groupby(['month', 'year'])[revenue_col].sum().unstack(fill_value=0)
    
    if previous_year in monthly.columns and current_year in monthly.columns:
        monthly['change'] = monthly[current_year] - monthly[previous_year]
        monthly['change_pct'] = (monthly['change'] / monthly[previous_year] * 100).round(2)
        monthly['change_pct'] = monthly['change_pct'].replace([np.inf, -np.inf], np.nan)
    
    monthly = monthly.reset_index()
    monthly['month_name'] = pd.to_datetime(monthly['month'], format='%m').dt.strftime('%B')
    
    return monthly


def growth_metrics(
    df: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    period: str = 'M'
) -> Dict[str, Any]:
    """
    Calculate growth metrics over time.
    
    Returns:
        Dict with various growth metrics
    """
    periodic = revenue_by_period(df, date_col, revenue_col, period, include_growth=True)
    
    if len(periodic) < 2:
        return {'error': 'Not enough periods for growth calculation'}
    
    # Get latest complete periods
    latest = periodic.iloc[-1]
    previous = periodic.iloc[-2]
    first = periodic.iloc[0]
    
    # Calculate various growth metrics
    mom_growth = latest.get('growth_pct', np.nan)  # Month-over-month
    
    # Calculate CAGR if we have enough data
    n_periods = len(periodic)
    first_revenue = first['revenue']
    latest_revenue = latest['revenue']
    
    if first_revenue > 0 and n_periods > 1:
        cagr = ((latest_revenue / first_revenue) ** (1 / (n_periods - 1)) - 1) * 100
    else:
        cagr = np.nan
    
    # Average growth rate
    avg_growth = periodic['growth_pct'].mean() if 'growth_pct' in periodic.columns else np.nan
    
    # Trend (are we accelerating or decelerating?)
    if 'growth_pct' in periodic.columns and len(periodic) >= 3:
        recent_growth = periodic['growth_pct'].tail(3).mean()
        earlier_growth = periodic['growth_pct'].head(3).mean()
        trend = 'accelerating' if recent_growth > earlier_growth else 'decelerating'
    else:
        trend = 'unknown'
    
    return {
        'latest_period': latest[date_col],
        'latest_revenue': round(latest['revenue'], 2),
        'previous_revenue': round(previous['revenue'], 2),
        'period_growth': round(mom_growth, 2) if not pd.isna(mom_growth) else None,
        'avg_growth_rate': round(avg_growth, 2) if not pd.isna(avg_growth) else None,
        'cagr': round(cagr, 2) if not pd.isna(cagr) else None,
        'total_periods': n_periods,
        'trend': trend,
        'total_revenue': round(periodic['revenue'].sum(), 2),
        'avg_period_revenue': round(periodic['revenue'].mean(), 2)
    }


def pareto_analysis(
    df: pd.DataFrame,
    dimension_col: str,
    revenue_col: str,
    threshold: float = 80.0
) -> Dict[str, Any]:
    """
    Pareto (80/20) analysis - which items drive most of the revenue?
    
    Args:
        df: DataFrame
        dimension_col: Column to analyze (e.g., 'product', 'customer')
        revenue_col: Revenue column
        threshold: Percentage threshold for "vital few" (default 80%)
    
    Returns:
        Dict with Pareto analysis results
    """
    breakdown = revenue_by_dimension(df, dimension_col, revenue_col)
    
    total_items = len(breakdown)
    total_revenue = breakdown['revenue'].sum()
    
    # Find items that make up the threshold
    vital_few = breakdown[breakdown['cumulative_share_pct'] <= threshold]
    
    # If none qualify, take at least the top one
    if len(vital_few) == 0:
        vital_few = breakdown.head(1)
    
    vital_few_count = len(vital_few)
    vital_few_pct = vital_few_count / total_items * 100 if total_items > 0 else 0
    vital_few_revenue = vital_few['revenue'].sum()
    vital_few_revenue_pct = vital_few_revenue / total_revenue * 100 if total_revenue > 0 else 0
    
    return {
        'total_items': total_items,
        'total_revenue': round(total_revenue, 2),
        'vital_few_count': vital_few_count,
        'vital_few_pct': round(vital_few_pct, 2),
        'vital_few_revenue': round(vital_few_revenue, 2),
        'vital_few_revenue_pct': round(vital_few_revenue_pct, 2),
        'vital_few_items': vital_few[dimension_col].tolist(),
        'concentration_ratio': round(vital_few_revenue_pct / vital_few_pct, 2) if vital_few_pct > 0 else 0,
        'is_concentrated': vital_few_pct < 30 and vital_few_revenue_pct > 70
    }


def revenue_forecast_simple(
    df: pd.DataFrame,
    date_col: str,
    revenue_col: str,
    periods_ahead: int = 3,
    method: str = 'moving_average'
) -> pd.DataFrame:
    """
    Simple revenue forecast using moving average or linear trend.
    
    Args:
        df: DataFrame with historical data
        date_col: Date column
        revenue_col: Revenue column
        periods_ahead: Number of periods to forecast
        method: 'moving_average' or 'linear_trend'
    
    Returns:
        DataFrame with historical data + forecast
    """
    periodic = revenue_by_period(df, date_col, revenue_col, 'M', include_growth=False)
    
    if len(periodic) < 3:
        return periodic
    
    if method == 'moving_average':
        # Use 3-period moving average
        last_ma = periodic['revenue'].tail(3).mean()
        forecasts = [last_ma] * periods_ahead
        
    elif method == 'linear_trend':
        # Simple linear regression
        x = np.arange(len(periodic))
        y = periodic['revenue'].values
        
        # Calculate slope and intercept
        slope = np.polyfit(x, y, 1)[0]
        last_value = periodic['revenue'].iloc[-1]
        
        forecasts = [last_value + slope * (i + 1) for i in range(periods_ahead)]
    
    else:
        forecasts = [periodic['revenue'].mean()] * periods_ahead
    
    # Create forecast rows
    last_period = pd.Period(periodic[date_col].iloc[-1])
    forecast_periods = [(last_period + i + 1).strftime('%Y-%m') for i in range(periods_ahead)]
    
    forecast_df = pd.DataFrame({
        date_col: forecast_periods,
        'revenue': forecasts,
        'avg_transaction': [np.nan] * periods_ahead,
        'transactions': [np.nan] * periods_ahead,
        'is_forecast': [True] * periods_ahead
    })
    
    periodic['is_forecast'] = False
    
    return pd.concat([periodic, forecast_df], ignore_index=True)
