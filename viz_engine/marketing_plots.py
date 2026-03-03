"""
Marketing-specific visualizations for Local Analyst.
Specialized charts for funnel, attribution, customer segments.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional


# Color palettes
MARKETING_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_attribution_waterfall(
    attribution_df: pd.DataFrame,
    channel_col: str = 'channel',
    value_col: str = 'attributed_conversions',
    title: str = "Attribution Waterfall",
    interactive: bool = True
) -> go.Figure:
    """
    Waterfall chart showing attribution across channels.
    """
    df = attribution_df.sort_values(value_col, ascending=False)
    
    if interactive:
        fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            x=df[channel_col],
            y=df[value_col],
            text=df[value_col].round(1),
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Channel",
            yaxis_title="Attributed Conversions",
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(df[channel_col], df[value_col], color=MARKETING_COLORS[:len(df)], alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom')
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('Attributed Conversions')
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig


def plot_revenue_concentration(
    revenue_data: pd.DataFrame,
    cumulative_col: str = 'cumulative_pct',
    entity_col: str = 'entity',
    title: str = "Revenue Concentration (Pareto)",
    interactive: bool = True
) -> go.Figure:
    """
    Pareto chart showing revenue concentration.
    """
    if interactive:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=revenue_data.index,
                y=revenue_data['revenue'] if 'revenue' in revenue_data.columns else revenue_data.iloc[:, 0],
                name="Revenue",
                marker_color='steelblue'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=revenue_data.index,
                y=revenue_data[cumulative_col],
                name="Cumulative %",
                mode='lines+markers',
                marker=dict(color='red', size=6),
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )
        
        fig.add_hline(y=80, line_dash="dash", line_color="green", 
                     annotation_text="80%", secondary_y=True)
        
        fig.update_xaxes(title_text=entity_col.capitalize())
        fig.update_yaxes(title_text="Revenue", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        x = range(len(revenue_data))
        
        ax1.bar(x, revenue_data['revenue'] if 'revenue' in revenue_data.columns else revenue_data.iloc[:, 0],
               color='steelblue', alpha=0.7)
        ax1.set_xlabel(entity_col.capitalize())
        ax1.set_ylabel('Revenue', color='steelblue')
        
        ax2 = ax1.twinx()
        ax2.plot(x, revenue_data[cumulative_col], 'r-o', linewidth=2, markersize=4)
        ax2.axhline(y=80, color='green', linestyle='--')
        ax2.set_ylabel('Cumulative %', color='red')
        ax2.set_ylim([0, 100])
        
        plt.title(title)
        plt.tight_layout()
        
        return fig


# Export all functions
__all__ = [
    'plot_attribution_waterfall',
    'plot_revenue_concentration',
]
