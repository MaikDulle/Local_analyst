"""
Visualization engine for Local Analyst.
Professional charts using Plotly (interactive) and Matplotlib (static).
Fills the empty viz_engine directory.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple


# Color palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ==================== CAMPAIGN VISUALIZATIONS ====================

def plot_campaign_performance(
    df: pd.DataFrame,
    date_col: str,
    metric_cols: List[str],
    title: str = "Campaign Performance Over Time",
    interactive: bool = True
) -> go.Figure:
    """
    Plot campaign metrics over time.
    
    Args:
        df: DataFrame with campaign data
        date_col: Date column name
        metric_cols: List of metrics to plot
        title: Chart title
        interactive: If True, return Plotly figure; else Matplotlib
        
    Returns:
        Plotly Figure object if interactive=True
    """
    if interactive:
        fig = go.Figure()
        
        for i, col in enumerate(metric_cols):
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                name=col,
                mode='lines+markers',
                line=dict(width=2, color=COLORS[i % len(COLORS)]),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in metric_cols:
            ax.plot(df[date_col], df[col], marker='o', label=col, linewidth=2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig


def plot_yoy_comparison(
    df: pd.DataFrame,
    period_col: str,
    current_col: str,
    previous_col: str,
    metric_name: str = "Metric",
    interactive: bool = True
) -> go.Figure:
    """
    Year-over-year comparison chart.
    
    Args:
        df: DataFrame with YoY data
        period_col: Period column (e.g., month, week)
        current_col: Current year column
        previous_col: Previous year column
        metric_name: Name of the metric being compared
        interactive: Plotly or Matplotlib
        
    Returns:
        Figure object
    """
    if interactive:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[period_col],
            y=df[current_col],
            name='Current Year',
            mode='lines+markers',
            line=dict(width=3, color=COLORS[0]),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=df[period_col],
            y=df[previous_col],
            name='Previous Year',
            mode='lines+markers',
            line=dict(width=3, color=COLORS[1], dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"{metric_name} - Year-over-Year Comparison",
            xaxis_title="Period",
            yaxis_title=metric_name,
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(df[period_col], df[current_col], marker='o', label='Current Year', linewidth=3)
        ax.plot(df[period_col], df[previous_col], marker='s', label='Previous Year', linewidth=3, linestyle='--')
        
        ax.set_xlabel('Period')
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} - Year-over-Year Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig


# ==================== FUNNEL VISUALIZATIONS ====================

def plot_conversion_funnel(
    stages: Dict[str, float],
    title: str = "Conversion Funnel",
    interactive: bool = True
) -> go.Figure:
    """
    Conversion funnel chart.
    
    Args:
        stages: Dict mapping stage names to values
        title: Chart title
        interactive: Plotly or Matplotlib
        
    Returns:
        Figure object
    """
    stage_names = list(stages.keys())
    values = list(stages.values())
    
    if interactive:
        fig = go.Figure(go.Funnel(
            y=stage_names,
            x=values,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=COLORS[:len(stage_names)])
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate percentages
        percentages = [v/values[0]*100 for v in values]
        
        y_pos = np.arange(len(stage_names))
        bars = ax.barh(y_pos, values, color=COLORS[:len(stage_names)], alpha=0.7)
        
        # Add percentage labels
        for i, (value, pct) in enumerate(zip(values, percentages)):
            ax.text(value, i, f' {value:,.0f} ({pct:.1f}%)', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stage_names)
        ax.set_xlabel('Count')
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        
        return fig


# ==================== COHORT VISUALIZATIONS ====================

def plot_cohort_heatmap(
    cohort_data: pd.DataFrame,
    title: str = "Cohort Analysis Heatmap",
    value_format: str = 'percentage',
    interactive: bool = True
) -> go.Figure:
    """
    Cohort retention/revenue heatmap.
    
    Args:
        cohort_data: Cohort matrix (cohorts as rows, periods as columns)
        title: Chart title
        value_format: 'percentage' or 'value'
        interactive: Plotly or Matplotlib
        
    Returns:
        Figure object
    """
    if interactive:
        text_template = '%{text:.1f}%' if value_format == 'percentage' else '%{text:.0f}'
        
        fig = go.Figure(data=go.Heatmap(
            z=cohort_data.values,
            x=cohort_data.columns,
            y=cohort_data.index.astype(str),
            colorscale='RdYlGn',
            text=cohort_data.values,
            texttemplate=text_template,
            textfont={"size": 10},
            colorbar=dict(title="%" if value_format == 'percentage' else "Value")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Periods Since Cohort",
            yaxis_title="Cohort",
            template='plotly_white',
            height=max(400, len(cohort_data) * 30)
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, max(6, len(cohort_data) * 0.5)))
        
        fmt = '.1f' if value_format == 'percentage' else '.0f'
        sns.heatmap(
            cohort_data,
            annot=True,
            fmt=fmt,
            cmap='RdYlGn',
            ax=ax,
            cbar_kws={'label': '% Retention' if value_format == 'percentage' else 'Value'}
        )
        
        ax.set_title(title)
        ax.set_xlabel('Periods Since Cohort')
        ax.set_ylabel('Cohort')
        plt.tight_layout()
        
        return fig


# ==================== RFM VISUALIZATIONS ====================

def plot_rfm_segments(
    rfm_df: pd.DataFrame,
    segment_col: str = 'segment',
    title: str = "RFM Customer Segmentation",
    interactive: bool = True
) -> go.Figure:
    """
    RFM segment distribution pie chart.
    
    Args:
        rfm_df: DataFrame with RFM segments
        segment_col: Segment column name
        title: Chart title
        interactive: Plotly or Matplotlib
        
    Returns:
        Figure object
    """
    segment_counts = rfm_df[segment_col].value_counts()
    
    if interactive:
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=COLORS)
        )])
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = COLORS[:len(segment_counts)]
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
              colors=colors, startangle=90)
        ax.set_title(title)
        
        return fig


def plot_rfm_scatter(
    rfm_df: pd.DataFrame,
    x_col: str = 'recency',
    y_col: str = 'monetary',
    segment_col: str = 'segment',
    size_col: str = 'frequency',
    title: str = "RFM Scatter Plot",
    interactive: bool = True
) -> go.Figure:
    """
    RFM scatter plot by segment.
    
    Args:
        rfm_df: DataFrame with RFM data
        x_col: X-axis column
        y_col: Y-axis column
        segment_col: Color by this column
        size_col: Size bubbles by this column
        title: Chart title
        interactive: Plotly or Matplotlib
        
    Returns:
        Figure object
    """
    if interactive:
        fig = px.scatter(
            rfm_df,
            x=x_col,
            y=y_col,
            color=segment_col,
            size=size_col,
            hover_data=[segment_col, 'recency', 'frequency', 'monetary'],
            title=title,
            template='plotly_white',
            height=600
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        segments = rfm_df[segment_col].unique()
        for i, segment in enumerate(segments):
            segment_data = rfm_df[rfm_df[segment_col] == segment]
            ax.scatter(
                segment_data[x_col],
                segment_data[y_col],
                label=segment,
                alpha=0.6,
                s=segment_data[size_col]*10,
                color=COLORS[i % len(COLORS)]
            )
        
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel(y_col.capitalize())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig


# ==================== A/B TEST VISUALIZATIONS ====================

def plot_ab_test_comparison(
    variant_a_name: str,
    variant_b_name: str,
    variant_a_mean: float,
    variant_b_mean: float,
    variant_a_ci: Tuple[float, float],
    variant_b_ci: Tuple[float, float],
    lift_pct: float,
    is_significant: bool,
    p_value: float,
    title: str = "A/B Test Results",
    interactive: bool = True
) -> go.Figure:
    """
    A/B test comparison with confidence intervals.
    
    Args:
        variant_a_name: Name of control
        variant_b_name: Name of treatment
        variant_a_mean: Control mean
        variant_b_mean: Treatment mean
        variant_a_ci: Control confidence interval (lower, upper)
        variant_b_ci: Treatment confidence interval (lower, upper)
        lift_pct: Lift percentage
        is_significant: Whether test is significant
        p_value: P-value
        title: Chart title
        interactive: Plotly or Matplotlib
        
    Returns:
        Figure object
    """
    names = [variant_a_name, variant_b_name]
    means = [variant_a_mean, variant_b_mean]
    ci_lower = [variant_a_ci[0], variant_b_ci[0]]
    ci_upper = [variant_a_ci[1], variant_b_ci[1]]
    
    if interactive:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=names,
            y=means,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[u - m for u, m in zip(ci_upper, means)],
                arrayminus=[m - l for m, l in zip(means, ci_lower)]
            ),
            marker_color=COLORS[:2],
            text=[f'{m:.4f}' for m in means],
            textposition='outside'
        ))
        
        # Add significance indicator
        sig_text = f"Statistically Significant (p={p_value:.4f})" if is_significant else f"Not Significant (p={p_value:.4f})"
        
        fig.add_annotation(
            text=f"Lift: {lift_pct:.2f}% | {sig_text}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.1,
            showarrow=False,
            font=dict(size=12, color="green" if is_significant else "gray")
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Variant",
            yaxis_title="Mean Value",
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(names))
        errors = [[m - l for m, l in zip(means, ci_lower)],
                 [u - m for u, m in zip(ci_upper, means)]]
        
        bars = ax.bar(x_pos, means, yerr=errors, capsize=10,
                     color=COLORS[:2], alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names)
        ax.set_ylabel('Mean Value')
        ax.set_title(title)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, mean,
                   f'{mean:.4f}', ha='center', va='bottom')
        
        # Add significance info
        sig_text = "Significant" if is_significant else "Not Significant"
        ax.text(0.5, 0.95, f"Lift: {lift_pct:.2f}% | p={p_value:.4f} ({sig_text})",
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgreen' if is_significant else 'wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


# ==================== CORRELATION VISUALIZATIONS ====================

def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    interactive: bool = True
) -> go.Figure:
    """
    Correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix
        title: Chart title
        interactive: Plotly or Matplotlib
        
    Returns:
        Figure object
    """
    if interactive:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=max(500, len(corr_matrix) * 40),
            width=max(500, len(corr_matrix) * 40)
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title(title)
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Test campaign performance chart
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'clicks': np.random.poisson(1000, len(dates)),
        'conversions': np.random.poisson(50, len(dates))
    })
    
    fig = plot_campaign_performance(df, 'date', ['clicks', 'conversions'])
    # fig.show()  # Uncomment to display
    print("Visualization engine ready!")
