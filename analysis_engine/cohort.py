"""
Cohort analysis for Local Analyst.
Track customer retention and revenue over time by cohort.
Integrates with existing analysis_engine pattern.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CohortAnalysisResult:
    """Results from cohort analysis."""
    retention_matrix: pd.DataFrame
    revenue_matrix: pd.DataFrame
    cohort_sizes: pd.Series
    avg_retention_by_period: pd.Series
    avg_revenue_by_period: pd.Series
    metadata: Dict[str, Any]


def cohort_retention_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    period: str = 'M',
    value_col: Optional[str] = None
) -> CohortAnalysisResult:
    """
    Perform cohort retention analysis.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Transaction date column
        period: Cohort period ('D'=daily, 'W'=weekly, 'M'=monthly, 'Q'=quarterly, 'Y'=yearly)
        value_col: Optional revenue/value column for revenue cohorts
        
    Returns:
        CohortAnalysisResult with retention and revenue matrices
    """
    # Prepare data
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create cohort period (first purchase period for each customer)
    df['cohort'] = df.groupby(customer_col)[date_col].transform('min')
    df['cohort_period'] = df['cohort'].dt.to_period(period)
    
    # Create transaction period
    df['transaction_period'] = df[date_col].dt.to_period(period)
    
    # Calculate periods since cohort
    df['periods_since_cohort'] = (df['transaction_period'] - df['cohort_period']).apply(lambda x: x.n)
    
    # Cohort sizes (number of unique customers in each cohort)
    cohort_sizes = df.groupby('cohort_period')[customer_col].nunique()
    
    # --- RETENTION MATRIX ---
    # Count unique customers per cohort per period
    retention_counts = df.groupby(['cohort_period', 'periods_since_cohort'])[customer_col].nunique().unstack(fill_value=0)
    
    # Convert to percentages
    retention_matrix = retention_counts.divide(cohort_sizes, axis=0) * 100
    
    # --- REVENUE MATRIX ---
    if value_col and value_col in df.columns:
        # Sum revenue per cohort per period
        revenue_matrix = df.groupby(['cohort_period', 'periods_since_cohort'])[value_col].sum().unstack(fill_value=0)
    else:
        revenue_matrix = pd.DataFrame()
    
    # --- AGGREGATED METRICS ---
    # Average retention by period (across all cohorts)
    avg_retention_by_period = retention_matrix.mean(axis=0)
    
    # Average revenue by period (if available)
    if not revenue_matrix.empty:
        avg_revenue_by_period = revenue_matrix.mean(axis=0)
    else:
        avg_revenue_by_period = pd.Series()
    
    # Metadata
    metadata = {
        'total_cohorts': len(cohort_sizes),
        'period': period,
        'period_range': f"{retention_matrix.columns.min()} to {retention_matrix.columns.max()}",
        'avg_cohort_size': cohort_sizes.mean(),
        'total_customers': cohort_sizes.sum(),
        'has_revenue_data': not revenue_matrix.empty
    }
    
    return CohortAnalysisResult(
        retention_matrix=retention_matrix,
        revenue_matrix=revenue_matrix,
        cohort_sizes=cohort_sizes,
        avg_retention_by_period=avg_retention_by_period,
        avg_revenue_by_period=avg_revenue_by_period,
        metadata=metadata
    )


def cohort_ltv_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    value_col: str,
    period: str = 'M',
    periods_to_analyze: int = 12
) -> pd.DataFrame:
    """
    Calculate cohort lifetime value progression.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Transaction date column
        value_col: Revenue/value column
        period: Cohort period
        periods_to_analyze: Number of periods to track (default 12)
        
    Returns:
        DataFrame with cumulative LTV by cohort and period
    """
    result = cohort_retention_analysis(df, customer_col, date_col, period, value_col)
    
    # Calculate cumulative revenue per customer
    revenue_matrix = result.revenue_matrix
    cohort_sizes = result.cohort_sizes
    
    # Revenue per customer
    revenue_per_customer = revenue_matrix.divide(cohort_sizes, axis=0)
    
    # Cumulative LTV
    cumulative_ltv = revenue_per_customer.cumsum(axis=1)
    
    # Limit to specified periods
    if periods_to_analyze and periods_to_analyze < len(cumulative_ltv.columns):
        cumulative_ltv = cumulative_ltv.iloc[:, :periods_to_analyze]
    
    return cumulative_ltv


def cohort_retention_rate_by_segment(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    segment_col: str,
    period: str = 'M'
) -> Dict[str, pd.DataFrame]:
    """
    Compare retention rates across different customer segments.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Transaction date column
        segment_col: Segment column (e.g., 'acquisition_channel', 'customer_tier')
        period: Cohort period
        
    Returns:
        Dict mapping segment names to retention matrices
    """
    segments = df[segment_col].unique()
    results = {}
    
    for segment in segments:
        segment_df = df[df[segment_col] == segment]
        result = cohort_retention_analysis(segment_df, customer_col, date_col, period)
        results[segment] = result.retention_matrix
    
    return results


def identify_high_retention_cohorts(
    retention_matrix: pd.DataFrame,
    period: int = 3,
    threshold: float = 50.0
) -> List[str]:
    """
    Identify cohorts with retention above threshold at specified period.
    
    Args:
        retention_matrix: Retention matrix from cohort_retention_analysis
        period: Period number to check (e.g., 3 for month 3)
        threshold: Minimum retention percentage
        
    Returns:
        List of cohort names with high retention
    """
    if period not in retention_matrix.columns:
        return []
    
    high_retention = retention_matrix[retention_matrix[period] >= threshold]
    return high_retention.index.astype(str).tolist()


def cohort_churn_analysis(
    retention_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate churn rates from retention matrix.
    
    Args:
        retention_matrix: Retention matrix from cohort_retention_analysis
        
    Returns:
        DataFrame with churn rates (100 - retention)
    """
    return 100 - retention_matrix


def cohort_reactivation_rate(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    inactive_days: int = 90,
    period: str = 'M'
) -> pd.DataFrame:
    """
    Calculate reactivation rates for customers who became inactive.
    
    Args:
        df: Transaction DataFrame
        customer_col: Customer ID column
        date_col: Transaction date column
        inactive_days: Days of inactivity to count as churned
        period: Analysis period
        
    Returns:
        DataFrame with reactivation rates by cohort
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])
    
    # Calculate days since last purchase for each transaction
    df['days_since_last'] = df.groupby(customer_col)[date_col].diff().dt.days
    
    # Mark reactivations (came back after being inactive)
    df['was_inactive'] = df['days_since_last'] > inactive_days
    df['reactivated'] = df['was_inactive'] & (df['days_since_last'].notna())
    
    # Create cohort
    df['cohort_period'] = df.groupby(customer_col)[date_col].transform('min').dt.to_period(period)
    df['transaction_period'] = df[date_col].dt.to_period(period)
    df['periods_since_cohort'] = (df['transaction_period'] - df['cohort_period']).apply(lambda x: x.n)
    
    # Count reactivations by cohort and period
    reactivations = df.groupby(['cohort_period', 'periods_since_cohort'])['reactivated'].sum().unstack(fill_value=0)
    
    # Count total inactive customers
    inactives = df.groupby(['cohort_period', 'periods_since_cohort'])['was_inactive'].sum().unstack(fill_value=0)
    
    # Reactivation rate
    reactivation_rate = (reactivations / inactives * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    return reactivation_rate


def compare_cohort_trends(
    retention_matrix: pd.DataFrame,
    recent_cohorts: int = 3,
    early_cohorts: int = 3
) -> Dict[str, Any]:
    """
    Compare retention trends between recent and early cohorts.
    
    Args:
        retention_matrix: Retention matrix from cohort_retention_analysis
        recent_cohorts: Number of most recent cohorts to analyze
        early_cohorts: Number of earliest cohorts to analyze
        
    Returns:
        Dict with comparison metrics
    """
    # Get recent and early cohorts
    recent = retention_matrix.tail(recent_cohorts).mean(axis=0)
    early = retention_matrix.head(early_cohorts).mean(axis=0)
    
    # Calculate difference
    difference = recent - early
    
    return {
        'recent_cohorts_avg_retention': recent.to_dict(),
        'early_cohorts_avg_retention': early.to_dict(),
        'retention_difference': difference.to_dict(),
        'improving_retention': (difference > 0).sum(),
        'declining_retention': (difference < 0).sum(),
        'trend': 'improving' if difference.mean() > 0 else 'declining'
    }


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    transactions = []
    for customer_id in range(1, 1001):
        # First purchase
        first_purchase = np.random.choice(dates[:300])
        
        # Generate repeat purchases with decay
        n_purchases = np.random.poisson(5)
        for i in range(n_purchases):
            days_offset = int(np.random.exponential(30) * i)
            purchase_date = first_purchase + pd.Timedelta(days=days_offset)
            
            if purchase_date <= dates[-1]:
                transactions.append({
                    'customer_id': f'CUST_{customer_id:04d}',
                    'date': purchase_date,
                    'revenue': np.random.lognormal(4, 0.5)
                })
    
    df = pd.DataFrame(transactions)
    
    # Run cohort analysis
    result = cohort_retention_analysis(
        df,
        customer_col='customer_id',
        date_col='date',
        period='M',
        value_col='revenue'
    )
    
    print("Cohort Retention Matrix (%):")
    print(result.retention_matrix.round(1))
    print(f"\nAverage retention by period:")
    print(result.avg_retention_by_period.round(1))
    print(f"\nMetadata:")
    print(result.metadata)
