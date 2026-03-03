"""
Conversion funnel analysis for Local Analyst.
Multi-stage funnel tracking with drop-off rates.
Integrates with existing analysis_engine pattern.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FunnelResult:
    """Results from funnel analysis."""
    stages: List[str]
    counts: List[int]
    conversion_rates: List[float]
    drop_off_rates: List[float]
    overall_conversion: float
    metadata: Dict[str, Any]


def analyze_funnel(
    df: pd.DataFrame,
    stage_columns: List[str],
    stage_names: Optional[List[str]] = None
) -> FunnelResult:
    """
    Analyze conversion funnel from binary stage columns.
    
    Args:
        df: DataFrame with funnel data
        stage_columns: List of binary column names (1=completed, 0=not completed)
        stage_names: Optional custom names for stages
        
    Returns:
        FunnelResult with funnel metrics
    """
    if not stage_names:
        stage_names = stage_columns
    
    counts = []
    conversion_rates = []
    drop_off_rates = []
    
    total_users = len(df)
    prev_count = total_users
    
    for col in stage_columns:
        # Count users who reached this stage
        count = df[col].sum()
        counts.append(int(count))
        
        # Conversion rate from previous stage
        if prev_count > 0:
            conv_rate = (count / prev_count) * 100
        else:
            conv_rate = 0
        conversion_rates.append(conv_rate)
        
        # Drop-off rate
        drop_off = ((prev_count - count) / prev_count * 100) if prev_count > 0 else 0
        drop_off_rates.append(drop_off)
        
        prev_count = count
    
    # Overall conversion (first to last stage)
    if total_users > 0:
        overall_conversion = (counts[-1] / total_users) * 100
    else:
        overall_conversion = 0
    
    metadata = {
        'total_users': total_users,
        'final_conversions': counts[-1] if counts else 0,
        'stages_count': len(stage_columns)
    }
    
    return FunnelResult(
        stages=stage_names,
        counts=counts,
        conversion_rates=conversion_rates,
        drop_off_rates=drop_off_rates,
        overall_conversion=overall_conversion,
        metadata=metadata
    )


def analyze_funnel_by_cohort(
    df: pd.DataFrame,
    stage_columns: List[str],
    cohort_col: str,
    stage_names: Optional[List[str]] = None
) -> Dict[str, FunnelResult]:
    """
    Analyze funnel separately for different cohorts.
    
    Args:
        df: DataFrame with funnel data
        stage_columns: List of stage columns
        cohort_col: Column to group by (e.g., 'device', 'channel')
        stage_names: Optional stage names
        
    Returns:
        Dict mapping cohort names to FunnelResults
    """
    results = {}
    
    for cohort in df[cohort_col].unique():
        cohort_df = df[df[cohort_col] == cohort]
        result = analyze_funnel(cohort_df, stage_columns, stage_names)
        results[cohort] = result
    
    return results


def identify_bottlenecks(
    funnel_result: FunnelResult,
    threshold: float = 30.0
) -> List[Dict[str, Any]]:
    """
    Identify funnel bottlenecks (stages with high drop-off).
    
    Args:
        funnel_result: FunnelResult from analyze_funnel
        threshold: Drop-off percentage threshold
        
    Returns:
        List of bottleneck stages
    """
    bottlenecks = []
    
    for i, (stage, drop_off) in enumerate(zip(funnel_result.stages, funnel_result.drop_off_rates)):
        if drop_off >= threshold:
            bottlenecks.append({
                'stage': stage,
                'stage_index': i,
                'drop_off_rate': drop_off,
                'users_lost': funnel_result.counts[i-1] - funnel_result.counts[i] if i > 0 else 0
            })
    
    return bottlenecks


def compare_funnels(
    funnel_a: FunnelResult,
    funnel_b: FunnelResult,
    funnel_a_name: str = "Funnel A",
    funnel_b_name: str = "Funnel B"
) -> pd.DataFrame:
    """
    Compare two funnels side-by-side.
    
    Args:
        funnel_a: First funnel
        funnel_b: Second funnel
        funnel_a_name: Name for first funnel
        funnel_b_name: Name for second funnel
        
    Returns:
        DataFrame with comparison
    """
    comparison = pd.DataFrame({
        'stage': funnel_a.stages,
        f'{funnel_a_name}_count': funnel_a.counts,
        f'{funnel_b_name}_count': funnel_b.counts,
        f'{funnel_a_name}_rate': [f"{r:.1f}%" for r in funnel_a.conversion_rates],
        f'{funnel_b_name}_rate': [f"{r:.1f}%" for r in funnel_b.conversion_rates],
        'difference': [b - a for a, b in zip(funnel_a.counts, funnel_b.counts)]
    })
    
    return comparison


def calculate_funnel_velocity(
    df: pd.DataFrame,
    stage_columns: List[str],
    date_col: str,
    time_window_days: int = 7
) -> Dict[str, float]:
    """
    Calculate average time between funnel stages.
    
    Args:
        df: DataFrame with funnel data and timestamps
        stage_columns: List of stage columns
        date_col: Date column name
        time_window_days: Maximum days to consider
        
    Returns:
        Dict with average days between stages
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    velocities = {}
    
    for i in range(len(stage_columns) - 1):
        current_stage = stage_columns[i]
        next_stage = stage_columns[i + 1]
        
        # Users who completed both stages
        completed_both = df[(df[current_stage] == 1) & (df[next_stage] == 1)]
        
        if len(completed_both) > 0:
            # Calculate time difference (simplified - assumes sequential completion)
            avg_days = time_window_days / 2  # Placeholder - real implementation would track timestamps
            velocities[f"{current_stage}_to_{next_stage}"] = avg_days
        else:
            velocities[f"{current_stage}_to_{next_stage}"] = 0
    
    return velocities


def segment_funnel_users(
    df: pd.DataFrame,
    stage_columns: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Segment users by where they dropped off.
    
    Args:
        df: DataFrame with funnel data
        stage_columns: List of stage columns
        
    Returns:
        Dict mapping drop-off stages to user DataFrames
    """
    segments = {}
    
    for i, stage in enumerate(stage_columns):
        if i == 0:
            # Users who didn't start
            segments['did_not_start'] = df[df[stage] == 0]
        else:
            prev_stage = stage_columns[i - 1]
            # Users who completed previous but not current
            dropped = df[(df[prev_stage] == 1) & (df[stage] == 0)]
            segments[f'dropped_at_{stage}'] = dropped
    
    # Users who completed entire funnel
    all_completed = df[df[stage_columns].all(axis=1)]
    segments['completed'] = all_completed
    
    return segments


# Example usage
if __name__ == "__main__":
    # Generate sample funnel data
    np.random.seed(42)
    
    n = 10000
    df = pd.DataFrame({
        'user_id': range(n),
        'visited_page': [1] * n,
        'viewed_product': np.random.binomial(1, 0.7, n),
        'added_to_cart': np.random.binomial(1, 0.4, n),
        'started_checkout': np.random.binomial(1, 0.3, n),
        'completed_purchase': np.random.binomial(1, 0.15, n)
    })
    
    # Ensure funnel logic (can't complete later stage without earlier ones)
    for i in range(1, 5):
        cols = df.columns[1:i+2]
        df[cols[i]] = df[cols[i]] * df[cols[i-1]]
    
    # Analyze funnel
    result = analyze_funnel(
        df,
        stage_columns=['visited_page', 'viewed_product', 'added_to_cart', 'started_checkout', 'completed_purchase'],
        stage_names=['Visit', 'View Product', 'Add to Cart', 'Checkout', 'Purchase']
    )
    
    print("Funnel Analysis:")
    for stage, count, rate, drop_off in zip(result.stages, result.counts, result.conversion_rates, result.drop_off_rates):
        print(f"{stage}: {count:,} users ({rate:.1f}% conversion, {drop_off:.1f}% drop-off)")
    
    print(f"\nOverall conversion: {result.overall_conversion:.2f}%")
    
    # Identify bottlenecks
    bottlenecks = identify_bottlenecks(result, threshold=40)
    print(f"\nBottlenecks (>40% drop-off):")
    for bn in bottlenecks:
        print(f"  {bn['stage']}: {bn['drop_off_rate']:.1f}% drop-off ({bn['users_lost']:,} users lost)")
