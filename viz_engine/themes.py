"""
Theme and color palette definitions for Local Analyst.
Professional color schemes for marketing visualizations.
"""

from typing import List, Dict


# Primary color palettes
COLORS = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    
    'marketing': ['#0066CC', '#FF6B35', '#00C9A7', '#FF006E', '#9B5DE5',
                  '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A', '#457B9D'],
    
    'sequential': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef',
                   '#eff3ff'],
    
    'diverging': ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6'],
    
    'categorical': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                    '#ffff33', '#a65628', '#f781bf'],
    
    'risk': {
        'low': '#2ca02c',
        'medium': '#ff7f0e', 
        'high': '#d62728'
    }
}


MARKETING_COLORS = COLORS['marketing']


def get_color_palette(palette_name: str = 'primary', n_colors: int = 10) -> List[str]:
    """
    Get color palette.
    
    Args:
        palette_name: Name of palette ('primary', 'marketing', 'sequential', etc.)
        n_colors: Number of colors needed
        
    Returns:
        List of hex color codes
    """
    if palette_name not in COLORS:
        palette_name = 'primary'
    
    palette = COLORS[palette_name]
    
    if isinstance(palette, dict):
        # For risk or other dict-based palettes
        return list(palette.values())
    
    # Repeat palette if needed
    if n_colors > len(palette):
        repeats = (n_colors // len(palette)) + 1
        palette = palette * repeats
    
    return palette[:n_colors]


# Chart styling defaults
CHART_CONFIG = {
    'plotly': {
        'template': 'plotly_white',
        'height': 500,
        'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
        'font': {'family': 'Arial, sans-serif', 'size': 12}
    },
    
    'matplotlib': {
        'figure.figsize': (12, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'legend.fontsize': 10
    }
}


def apply_matplotlib_theme():
    """Apply matplotlib theme settings."""
    import matplotlib.pyplot as plt
    
    for key, value in CHART_CONFIG['matplotlib'].items():
        plt.rcParams[key] = value


# Semantic color mappings for business metrics
METRIC_COLORS = {
    'positive': '#2ca02c',  # Green
    'negative': '#d62728',  # Red
    'neutral': '#1f77b4',   # Blue
    'warning': '#ff7f0e',   # Orange
}


def get_metric_color(value: float, threshold_positive: float = 0, threshold_negative: float = 0) -> str:
    """
    Get color based on metric value.
    
    Args:
        value: Metric value
        threshold_positive: Value above this is positive (green)
        threshold_negative: Value below this is negative (red)
        
    Returns:
        Hex color code
    """
    if value > threshold_positive:
        return METRIC_COLORS['positive']
    elif value < threshold_negative:
        return METRIC_COLORS['negative']
    else:
        return METRIC_COLORS['neutral']


# Export
__all__ = [
    'COLORS',
    'MARKETING_COLORS',
    'CHART_CONFIG',
    'METRIC_COLORS',
    'get_color_palette',
    'apply_matplotlib_theme',
    'get_metric_color'
]
