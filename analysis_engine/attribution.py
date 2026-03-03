"""
Attribution modeling for Local Analyst.
Multi-touch attribution models for marketing channel analysis.
Integrates with existing analysis_engine pattern.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class AttributionResult:
    """Attribution analysis result."""
    model_name: str
    channel_attribution: pd.DataFrame
    total_conversions: float
    methodology: str


def last_touch_attribution(
    df: pd.DataFrame,
    customer_col: str,
    channel_col: str,
    conversion_col: str,
    date_col: str
) -> AttributionResult:
    """
    Last-touch attribution model.
    Assigns 100% credit to the last touchpoint before conversion.
    
    Args:
        df: DataFrame with touchpoint data
        customer_col: Customer ID column
        channel_col: Marketing channel column
        conversion_col: Conversion column (1/0 or value)
        date_col: Date column
        
    Returns:
        AttributionResult with channel attribution
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])
    
    # Filter to conversions only
    conversions = df[df[conversion_col] > 0].copy()
    
    # Get last channel before each conversion
    conversions['last_channel'] = conversions.groupby(customer_col)[channel_col].transform('last')
    
    # Attribute conversions
    attribution = conversions.groupby('last_channel')[conversion_col].sum()
    
    result_df = pd.DataFrame({
        'channel': attribution.index,
        'attributed_conversions': attribution.values,
        'attribution_percentage': (attribution.values / attribution.sum() * 100).round(2)
    })
    
    return AttributionResult(
        model_name='Last Touch',
        channel_attribution=result_df.sort_values('attributed_conversions', ascending=False).reset_index(drop=True),
        total_conversions=attribution.sum(),
        methodology='100% credit to last touchpoint before conversion'
    )


def first_touch_attribution(
    df: pd.DataFrame,
    customer_col: str,
    channel_col: str,
    conversion_col: str,
    date_col: str
) -> AttributionResult:
    """
    First-touch attribution model.
    Assigns 100% credit to the first touchpoint in customer journey.
    
    Args:
        df: DataFrame with touchpoint data
        customer_col: Customer ID column
        channel_col: Marketing channel column
        conversion_col: Conversion column
        date_col: Date column
        
    Returns:
        AttributionResult with channel attribution
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])
    
    # Filter to conversions
    conversions = df[df[conversion_col] > 0].copy()
    
    # Get first channel for each customer
    conversions['first_channel'] = conversions.groupby(customer_col)[channel_col].transform('first')
    
    # Attribute conversions
    attribution = conversions.groupby('first_channel')[conversion_col].sum()
    
    result_df = pd.DataFrame({
        'channel': attribution.index,
        'attributed_conversions': attribution.values,
        'attribution_percentage': (attribution.values / attribution.sum() * 100).round(2)
    })
    
    return AttributionResult(
        model_name='First Touch',
        channel_attribution=result_df.sort_values('attributed_conversions', ascending=False).reset_index(drop=True),
        total_conversions=attribution.sum(),
        methodology='100% credit to first touchpoint in customer journey'
    )


def linear_attribution(
    df: pd.DataFrame,
    customer_col: str,
    channel_col: str,
    conversion_col: str,
    date_col: str,
    lookback_days: int = 30
) -> AttributionResult:
    """
    Linear attribution model.
    Distributes credit equally across all touchpoints within lookback window.
    
    Args:
        df: DataFrame with touchpoint data
        customer_col: Customer ID column
        channel_col: Marketing channel column
        conversion_col: Conversion column
        date_col: Date column
        lookback_days: Days to look back from conversion
        
    Returns:
        AttributionResult with channel attribution
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])
    
    attributions = []
    
    for customer in df[customer_col].unique():
        customer_df = df[df[customer_col] == customer].copy()
        conversions = customer_df[customer_df[conversion_col] > 0]
        
        for idx, conv in conversions.iterrows():
            conv_date = conv[date_col]
            conv_value = conv[conversion_col]
            lookback_start = conv_date - timedelta(days=lookback_days)
            
            # Get all touches in lookback window
            touches = customer_df[
                (customer_df[date_col] >= lookback_start) & 
                (customer_df[date_col] <= conv_date)
            ]
            
            if len(touches) > 0:
                # Equal credit to each touch
                credit_per_touch = conv_value / len(touches)
                
                for _, touch in touches.iterrows():
                    attributions.append({
                        'channel': touch[channel_col],
                        'attributed_conversions': credit_per_touch
                    })
    
    if not attributions:
        return AttributionResult(
            model_name='Linear',
            channel_attribution=pd.DataFrame(columns=['channel', 'attributed_conversions', 'attribution_percentage']),
            total_conversions=0,
            methodology=f'Equal credit to all touchpoints within {lookback_days} days'
        )
    
    attribution_df = pd.DataFrame(attributions)
    attribution_summary = attribution_df.groupby('channel')['attributed_conversions'].sum()
    
    result_df = pd.DataFrame({
        'channel': attribution_summary.index,
        'attributed_conversions': attribution_summary.values,
        'attribution_percentage': (attribution_summary.values / attribution_summary.sum() * 100).round(2)
    })
    
    return AttributionResult(
        model_name='Linear',
        channel_attribution=result_df.sort_values('attributed_conversions', ascending=False).reset_index(drop=True),
        total_conversions=attribution_summary.sum(),
        methodology=f'Equal credit to all touchpoints within {lookback_days} days'
    )


def time_decay_attribution(
    df: pd.DataFrame,
    customer_col: str,
    channel_col: str,
    conversion_col: str,
    date_col: str,
    lookback_days: int = 30,
    half_life_days: int = 7
) -> AttributionResult:
    """
    Time-decay attribution model.
    More recent touchpoints get exponentially more credit.
    
    Args:
        df: DataFrame with touchpoint data
        customer_col: Customer ID column
        channel_col: Marketing channel column
        conversion_col: Conversion column
        date_col: Date column
        lookback_days: Days to look back from conversion
        half_life_days: Days for credit to halve
        
    Returns:
        AttributionResult with channel attribution
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])
    
    attributions = []
    
    for customer in df[customer_col].unique():
        customer_df = df[df[customer_col] == customer].copy()
        conversions = customer_df[customer_df[conversion_col] > 0]
        
        for idx, conv in conversions.iterrows():
            conv_date = conv[date_col]
            conv_value = conv[conversion_col]
            lookback_start = conv_date - timedelta(days=lookback_days)
            
            # Get all touches in lookback window
            touches = customer_df[
                (customer_df[date_col] >= lookback_start) & 
                (customer_df[date_col] <= conv_date)
            ].copy()
            
            if len(touches) > 0:
                # Calculate time decay weights
                touches['days_to_conversion'] = (conv_date - touches[date_col]).dt.days
                touches['weight'] = 2 ** (-touches['days_to_conversion'] / half_life_days)
                
                # Normalize weights
                total_weight = touches['weight'].sum()
                touches['credit'] = (touches['weight'] / total_weight) * conv_value
                
                for _, touch in touches.iterrows():
                    attributions.append({
                        'channel': touch[channel_col],
                        'attributed_conversions': touch['credit']
                    })
    
    if not attributions:
        return AttributionResult(
            model_name='Time Decay',
            channel_attribution=pd.DataFrame(columns=['channel', 'attributed_conversions', 'attribution_percentage']),
            total_conversions=0,
            methodology=f'Exponential decay with {half_life_days}-day half-life'
        )
    
    attribution_df = pd.DataFrame(attributions)
    attribution_summary = attribution_df.groupby('channel')['attributed_conversions'].sum()
    
    result_df = pd.DataFrame({
        'channel': attribution_summary.index,
        'attributed_conversions': attribution_summary.values,
        'attribution_percentage': (attribution_summary.values / attribution_summary.sum() * 100).round(2)
    })
    
    return AttributionResult(
        model_name='Time Decay',
        channel_attribution=result_df.sort_values('attributed_conversions', ascending=False).reset_index(drop=True),
        total_conversions=attribution_summary.sum(),
        methodology=f'Exponential decay with {half_life_days}-day half-life'
    )


def position_based_attribution(
    df: pd.DataFrame,
    customer_col: str,
    channel_col: str,
    conversion_col: str,
    date_col: str,
    lookback_days: int = 30,
    first_touch_weight: float = 0.4,
    last_touch_weight: float = 0.4
) -> AttributionResult:
    """
    Position-based (U-shaped) attribution model.
    Gives specified weight to first and last touch, distributes remainder equally.
    
    Args:
        df: DataFrame with touchpoint data
        customer_col: Customer ID column
        channel_col: Marketing channel column
        conversion_col: Conversion column
        date_col: Date column
        lookback_days: Days to look back from conversion
        first_touch_weight: Weight for first touch (default 0.4 = 40%)
        last_touch_weight: Weight for last touch (default 0.4 = 40%)
        
    Returns:
        AttributionResult with channel attribution
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])
    
    # Validate weights
    middle_weight = 1 - first_touch_weight - last_touch_weight
    if middle_weight < 0:
        raise ValueError("First and last touch weights must sum to <= 1")
    
    attributions = []
    
    for customer in df[customer_col].unique():
        customer_df = df[df[customer_col] == customer].copy()
        conversions = customer_df[customer_df[conversion_col] > 0]
        
        for idx, conv in conversions.iterrows():
            conv_date = conv[date_col]
            conv_value = conv[conversion_col]
            lookback_start = conv_date - timedelta(days=lookback_days)
            
            # Get all touches in lookback window
            touches = customer_df[
                (customer_df[date_col] >= lookback_start) & 
                (customer_df[date_col] <= conv_date)
            ].copy()
            
            if len(touches) == 0:
                continue
            elif len(touches) == 1:
                # Only one touch - gets 100%
                attributions.append({
                    'channel': touches.iloc[0][channel_col],
                    'attributed_conversions': conv_value
                })
            else:
                # First touch
                attributions.append({
                    'channel': touches.iloc[0][channel_col],
                    'attributed_conversions': conv_value * first_touch_weight
                })
                
                # Last touch
                attributions.append({
                    'channel': touches.iloc[-1][channel_col],
                    'attributed_conversions': conv_value * last_touch_weight
                })
                
                # Middle touches (if any)
                if len(touches) > 2:
                    middle_credit = (conv_value * middle_weight) / (len(touches) - 2)
                    for i in range(1, len(touches) - 1):
                        attributions.append({
                            'channel': touches.iloc[i][channel_col],
                            'attributed_conversions': middle_credit
                        })
    
    if not attributions:
        return AttributionResult(
            model_name='Position Based',
            channel_attribution=pd.DataFrame(columns=['channel', 'attributed_conversions', 'attribution_percentage']),
            total_conversions=0,
            methodology=f'First: {first_touch_weight*100}%, Last: {last_touch_weight*100}%, Middle: {middle_weight*100}%'
        )
    
    attribution_df = pd.DataFrame(attributions)
    attribution_summary = attribution_df.groupby('channel')['attributed_conversions'].sum()
    
    result_df = pd.DataFrame({
        'channel': attribution_summary.index,
        'attributed_conversions': attribution_summary.values,
        'attribution_percentage': (attribution_summary.values / attribution_summary.sum() * 100).round(2)
    })
    
    return AttributionResult(
        model_name='Position Based',
        channel_attribution=result_df.sort_values('attributed_conversions', ascending=False).reset_index(drop=True),
        total_conversions=attribution_summary.sum(),
        methodology=f'First: {first_touch_weight*100}%, Last: {last_touch_weight*100}%, Middle: {middle_weight*100}%'
    )


def compare_attribution_models(
    df: pd.DataFrame,
    customer_col: str,
    channel_col: str,
    conversion_col: str,
    date_col: str,
    lookback_days: int = 30
) -> pd.DataFrame:
    """
    Compare all attribution models side-by-side.
    
    Args:
        df: DataFrame with touchpoint data
        customer_col: Customer ID column
        channel_col: Marketing channel column
        conversion_col: Conversion column
        date_col: Date column
        lookback_days: Days to look back
        
    Returns:
        DataFrame comparing all models
    """
    models = {
        'Last Touch': last_touch_attribution(df, customer_col, channel_col, conversion_col, date_col),
        'First Touch': first_touch_attribution(df, customer_col, channel_col, conversion_col, date_col),
        'Linear': linear_attribution(df, customer_col, channel_col, conversion_col, date_col, lookback_days),
        'Time Decay': time_decay_attribution(df, customer_col, channel_col, conversion_col, date_col, lookback_days),
        'Position Based': position_based_attribution(df, customer_col, channel_col, conversion_col, date_col, lookback_days)
    }
    
    # Combine results
    all_channels = set()
    for model_result in models.values():
        all_channels.update(model_result.channel_attribution['channel'].tolist())
    
    comparison = pd.DataFrame({'channel': sorted(all_channels)})
    
    for model_name, model_result in models.items():
        channel_attr = model_result.channel_attribution.set_index('channel')['attributed_conversions']
        comparison[model_name] = comparison['channel'].map(channel_attr).fillna(0)
    
    return comparison


# Example usage
if __name__ == "__main__":
    # Generate sample touchpoint data
    np.random.seed(42)
    
    channels = ['Google Ads', 'Facebook', 'Email', 'Organic', 'Direct']
    
    touchpoints = []
    for customer_id in range(1, 101):
        # Random journey length
        journey_length = np.random.randint(1, 6)
        
        for i in range(journey_length):
            date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 90))
            touchpoints.append({
                'customer_id': f'CUST_{customer_id:03d}',
                'channel': np.random.choice(channels),
                'date': date,
                'conversion': 1 if i == journey_length - 1 and np.random.random() < 0.3 else 0
            })
    
    df = pd.DataFrame(touchpoints)
    
    # Run attribution models
    last_touch = last_touch_attribution(df, 'customer_id', 'channel', 'conversion', 'date')
    print("Last Touch Attribution:")
    print(last_touch.channel_attribution)
    
    # Compare all models
    comparison = compare_attribution_models(df, 'customer_id', 'channel', 'conversion', 'date')
    print("\nModel Comparison:")
    print(comparison)
