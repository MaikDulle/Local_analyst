"""
Visualization engine for Local Analyst.
Professional charts for marketing analytics with Plotly and Matplotlib.
"""

from .plots import (
    # Campaign visualizations
    plot_campaign_performance,
    plot_yoy_comparison,
    
    # Funnel visualization
    plot_conversion_funnel,
    
    # Cohort visualizations
    plot_cohort_heatmap,
    
    # RFM visualizations
    plot_rfm_segments,
    plot_rfm_scatter,
    
    # A/B test visualization
    plot_ab_test_comparison,
    
    # Correlation visualization
    plot_correlation_matrix,
)

from .marketing_plots import (
    plot_attribution_waterfall,
    plot_revenue_concentration,
)

from .export import (
    export_plotly_chart,
    export_matplotlib_chart,
    export_chart,
    export_multiple_charts,
    get_chart_as_bytes,
)

from .themes import (
    COLORS,
    MARKETING_COLORS,
    CHART_CONFIG,
    get_color_palette,
    apply_matplotlib_theme,
)


__all__ = [
    # Core plots
    'plot_campaign_performance',
    'plot_yoy_comparison',
    'plot_conversion_funnel',
    'plot_cohort_heatmap',
    'plot_rfm_segments',
    'plot_rfm_scatter',
    'plot_ab_test_comparison',
    'plot_correlation_matrix',
    
    # Marketing plots
    'plot_attribution_waterfall',
    'plot_revenue_concentration',
    
    # Export functions
    'export_plotly_chart',
    'export_matplotlib_chart',
    'export_chart',
    'export_multiple_charts',
    'get_chart_as_bytes',
    
    # Themes
    'COLORS',
    'MARKETING_COLORS',
    'CHART_CONFIG',
    'get_color_palette',
    'apply_matplotlib_theme',
]
