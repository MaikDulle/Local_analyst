"""
Customer analysis and segmentation for ecom.
Includes RFM scoring, customer value tiers, and retention analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CustomerMetrics:
    """Metrics for a single customer."""
    customer_id: str
    total_revenue: float
    total_orders: int
    total_quantity: int
    avg_order_value: float
    first_order: str
    last_order: str
    customer_lifetime_days: int
    orders_per_month: float


def customer_summary(
    df: pd.DataFrame,
    customer_col: str,
    revenue_col: str,
    date_col: Optional[str] = None,
    quantity_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate customer-level summary statistics.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        revenue_col: Revenue column
        date_col: Date column (optional, for tenure calculation)
        quantity_col: Quantity column (optional)
    """
    agg_dict = {
        revenue_col: ['sum', 'mean', 'count']
    }
    
    if quantity_col and quantity_col in df.columns:
        agg_dict[quantity_col] = 'sum'
    
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        agg_dict[date_col] = ['min', 'max']
    
    grouped = df.groupby(customer_col).agg(agg_dict)
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    
    # Rename columns
    rename_map = {
        f'{revenue_col}_sum': 'total_revenue',
        f'{revenue_col}_mean': 'avg_order_value',
        f'{revenue_col}_count': 'total_orders'
    }
    
    if quantity_col:
        rename_map[f'{quantity_col}_sum'] = 'total_quantity'
    
    if date_col:
        rename_map[f'{date_col}_min'] = 'first_order'
        rename_map[f'{date_col}_max'] = 'last_order'
    
    grouped = grouped.rename(columns=rename_map)
    
    # Calculate tenure
    if date_col and 'first_order' in grouped.columns and 'last_order' in grouped.columns:
        grouped['customer_lifetime_days'] = (grouped['last_order'] - grouped['first_order']).dt.days
        grouped['months_active'] = (grouped['customer_lifetime_days'] / 30).round(1)
        grouped['orders_per_month'] = (grouped['total_orders'] / grouped['months_active'].replace(0, 1)).round(2)
    
    # Revenue share
    total_revenue = grouped['total_revenue'].sum()
    grouped['revenue_share_pct'] = (grouped['total_revenue'] / total_revenue * 100).round(2)
    
    return grouped.sort_values('total_revenue', ascending=False).reset_index()


def rfm_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    revenue_col: str,
    reference_date: Optional[str] = None,
    r_bins: int = 5,
    f_bins: int = 5,
    m_bins: int = 5
) -> pd.DataFrame:
    """
    RFM (Recency, Frequency, Monetary) segmentation.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Transaction date column
        revenue_col: Revenue column
        reference_date: Date to calculate recency from (defaults to max date in data)
        r_bins: Number of recency bins (default 5)
        f_bins: Number of frequency bins (default 5)
        m_bins: Number of monetary bins (default 5)
    
    Returns:
        DataFrame with RFM scores and segments per customer
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, customer_col, revenue_col])
    
    # Reference date
    if reference_date:
        ref_date = pd.to_datetime(reference_date)
    else:
        ref_date = df[date_col].max() + timedelta(days=1)
    
    # Calculate RFM values
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (ref_date - x.max()).days,  # Recency
        revenue_col: ['count', 'sum']  # Frequency and Monetary
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary']
    
    # Score each dimension (5 = best)
    # Recency: lower is better, so reverse the labels
    rfm['r_score'] = pd.qcut(rfm['recency'], q=r_bins, labels=range(r_bins, 0, -1), duplicates='drop').astype(int)
    
    # Frequency: higher is better
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=f_bins, labels=range(1, f_bins + 1), duplicates='drop').astype(int)
    
    # Monetary: higher is better
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=m_bins, labels=range(1, m_bins + 1), duplicates='drop').astype(int)
    
    # Combined RFM score
    rfm['rfm_score'] = rfm['r_score'] * 100 + rfm['f_score'] * 10 + rfm['m_score']
    rfm['rfm_sum'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    
    # Segment assignment
    rfm['segment'] = rfm.apply(_assign_rfm_segment, axis=1)
    
    return rfm.reset_index()


def _assign_rfm_segment(row) -> str:
    """Assign customer segment based on RFM scores."""
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    
    # Champions: high on all dimensions
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    
    # Loyal Customers: high frequency and monetary
    if f >= 4 and m >= 4:
        return 'Loyal Customers'
    
    # Potential Loyalists: recent, decent frequency
    if r >= 4 and f >= 2:
        return 'Potential Loyalists'
    
    # New Customers: very recent, low frequency
    if r >= 4 and f <= 2:
        return 'New Customers'
    
    # At Risk: used to be good, haven't purchased recently
    if r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    
    # Can\'t Lose: high value but churning
    if r <= 2 and f >= 4 and m >= 4:
        return "Can't Lose"
    
    # Hibernating: low on all dimensions
    if r <= 2 and f <= 2:
        return 'Hibernating'
    
    # Need Attention: medium scores
    if r >= 3 and f >= 3:
        return 'Need Attention'
    
    # About to Sleep: below average recency
    if r <= 3 and f <= 3:
        return 'About to Sleep'
    
    return 'Other'


def customer_value_tiers(
    df: pd.DataFrame,
    customer_col: str,
    revenue_col: str,
    tiers: int = 4,
    tier_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Segment customers into value tiers based on total revenue.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        revenue_col: Revenue column
        tiers: Number of tiers (default 4)
        tier_names: Custom tier names (e.g., ['Bronze', 'Silver', 'Gold', 'Platinum'])
    """
    # Calculate total revenue per customer
    customer_revenue = df.groupby(customer_col)[revenue_col].sum().reset_index()
    customer_revenue.columns = [customer_col, 'total_revenue']
    
    # Default tier names
    if tier_names is None:
        if tiers == 4:
            tier_names = ['Bronze', 'Silver', 'Gold', 'Platinum']
        elif tiers == 3:
            tier_names = ['Low', 'Medium', 'High']
        else:
            tier_names = [f'Tier {i+1}' for i in range(tiers)]
    
    # Assign tiers based on quantiles
    customer_revenue['tier'] = pd.qcut(
        customer_revenue['total_revenue'].rank(method='first'),
        q=tiers,
        labels=tier_names
    )
    
    # Tier statistics
    tier_stats = customer_revenue.groupby('tier').agg({
        customer_col: 'count',
        'total_revenue': ['sum', 'mean', 'min', 'max']
    })
    tier_stats.columns = ['customer_count', 'total_revenue', 'avg_revenue', 'min_revenue', 'max_revenue']
    
    total_customers = tier_stats['customer_count'].sum()
    total_revenue = tier_stats['total_revenue'].sum()
    
    tier_stats['customer_share_pct'] = (tier_stats['customer_count'] / total_customers * 100).round(2)
    tier_stats['revenue_share_pct'] = (tier_stats['total_revenue'] / total_revenue * 100).round(2)
    
    return {
        'customers': customer_revenue,
        'tier_summary': tier_stats.reset_index()
    }


def customer_cohort_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    revenue_col: str,
    cohort_period: str = 'M'
) -> pd.DataFrame:
    """
    Cohort analysis - track customer behavior by acquisition cohort.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Transaction date column
        revenue_col: Revenue column
        cohort_period: 'M' for monthly, 'Q' for quarterly, 'W' for weekly
    
    Returns:
        Cohort retention matrix
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Determine first transaction (acquisition) date per customer
    first_purchase = df.groupby(customer_col)[date_col].min().reset_index()
    first_purchase.columns = [customer_col, 'cohort_date']
    
    # Merge cohort date back
    df = df.merge(first_purchase, on=customer_col)
    
    # Create period columns
    df['cohort_period'] = df['cohort_date'].dt.to_period(cohort_period)
    df['transaction_period'] = df[date_col].dt.to_period(cohort_period)
    
    # Calculate period index (0 = acquisition period, 1 = next period, etc.)
    df['period_index'] = (df['transaction_period'] - df['cohort_period']).apply(lambda x: x.n if hasattr(x, 'n') else 0)
    
    # Create cohort matrix - count unique customers
    cohort_matrix = df.groupby(['cohort_period', 'period_index'])[customer_col].nunique().unstack(fill_value=0)
    
    # Calculate retention rates
    cohort_sizes = cohort_matrix[0]
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
    
    # Clean up column names
    retention_matrix.columns = [f'Period_{i}' for i in retention_matrix.columns]
    retention_matrix.index = retention_matrix.index.astype(str)
    
    return retention_matrix.round(1)


def customer_lifetime_value(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    revenue_col: str,
    time_horizon_months: int = 12
) -> pd.DataFrame:
    """
    Calculate Customer Lifetime Value (CLV) using historical data.
    
    Simple CLV = Average Order Value × Purchase Frequency × Customer Lifespan
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Date column
        revenue_col: Revenue column
        time_horizon_months: Prediction horizon in months
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Customer summary
    customer_data = df.groupby(customer_col).agg({
        revenue_col: ['sum', 'mean', 'count'],
        date_col: ['min', 'max']
    })
    customer_data.columns = ['total_revenue', 'avg_order_value', 'order_count', 'first_order', 'last_order']
    
    # Calculate customer tenure
    customer_data['tenure_days'] = (customer_data['last_order'] - customer_data['first_order']).dt.days
    customer_data['tenure_months'] = (customer_data['tenure_days'] / 30).clip(lower=1)
    
    # Purchase frequency (orders per month)
    customer_data['purchase_frequency'] = customer_data['order_count'] / customer_data['tenure_months']
    
    # Projected CLV
    customer_data['projected_clv'] = (
        customer_data['avg_order_value'] * 
        customer_data['purchase_frequency'] * 
        time_horizon_months
    ).round(2)
    
    # Historical CLV (actual)
    customer_data['historical_clv'] = customer_data['total_revenue']
    
    return customer_data.reset_index()


def churn_risk_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    revenue_col: str,
    inactive_days_threshold: int = 90
) -> pd.DataFrame:
    """
    Identify customers at risk of churning based on inactivity.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Date column
        revenue_col: Revenue column
        inactive_days_threshold: Days since last purchase to flag as at-risk
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    reference_date = df[date_col].max()
    
    # Customer activity summary
    customer_activity = df.groupby(customer_col).agg({
        date_col: ['max', 'count'],
        revenue_col: 'sum'
    })
    customer_activity.columns = ['last_purchase', 'order_count', 'total_revenue']
    
    # Days since last purchase
    customer_activity['days_since_purchase'] = (reference_date - customer_activity['last_purchase']).dt.days
    
    # Churn risk categorization
    def categorize_risk(days):
        if days <= 30:
            return 'Active'
        elif days <= 60:
            return 'Cooling'
        elif days <= inactive_days_threshold:
            return 'At Risk'
        else:
            return 'Churned'
    
    customer_activity['churn_risk'] = customer_activity['days_since_purchase'].apply(categorize_risk)
    
    # Value at risk
    at_risk = customer_activity[customer_activity['churn_risk'].isin(['At Risk', 'Churned'])]
    
    summary = {
        'total_customers': len(customer_activity),
        'active': len(customer_activity[customer_activity['churn_risk'] == 'Active']),
        'cooling': len(customer_activity[customer_activity['churn_risk'] == 'Cooling']),
        'at_risk': len(customer_activity[customer_activity['churn_risk'] == 'At Risk']),
        'churned': len(customer_activity[customer_activity['churn_risk'] == 'Churned']),
        'revenue_at_risk': at_risk['total_revenue'].sum()
    }
    
    return {
        'customer_risk': customer_activity.reset_index(),
        'summary': summary
    }
