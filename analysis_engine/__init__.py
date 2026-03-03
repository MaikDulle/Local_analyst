"""
Analysis Engine - Deterministic analytics for ecom data.
No AI required - fast, reliable, statistically sound.

Usage:
    from analysis_engine import summarize_dataset, revenue_by_period, rfm_analysis
    
    # Get dataset summary
    summary = summarize_dataset(df)
    print(summary.insights)
    
    # Revenue analysis
    revenue = revenue_by_period(df, 'date', 'revenue', period='M')
    
    # Customer segmentation
    rfm = rfm_analysis(df, 'customer_id', 'date', 'revenue')
"""

# Summary statistics
from .summary import (
    summarize_dataset,
    summarize_numeric,
    summarize_categorical,
    quick_stats,
    compare_periods,
    top_performers,
    distribution_stats,
    DatasetSummary,
    NumericSummary,
    CategoricalSummary,
)

# Revenue analysis
from .revenue import (
    calculate_revenue_metrics,
    revenue_by_period,
    revenue_by_dimension,
    compare_periods_yoy,
    growth_metrics,
    pareto_analysis,
    revenue_forecast_simple,
    RevenueMetrics,
)

# Product analysis
from .product import (
    product_performance,
    top_products,
    bottom_products,
    product_comparison,
    price_analysis,
    product_trends,
    product_basket_analysis,
    category_performance,
    product_performance_score,
)

# Customer analysis
from .customer import (
    customer_summary,
    rfm_analysis,
    customer_value_tiers,
    customer_cohort_analysis,
    customer_lifetime_value,
    churn_risk_analysis,
    CustomerMetrics,
)

# Correlation analysis
from .correlations import (
    correlation_matrix,
    find_strong_correlations,
    correlation_with_target,
    cross_tabulation,
    group_comparison,
    detect_outliers,
    relationship_strength,
    multi_variable_analysis,
    CorrelationResult,
)

# mixed correlation
from .mixed_correlations import (
    analyze_mixed_correlations,
    find_all_relationships,
    MixedCorrelationResult,
)

# A/B Testing
from .ab_testing import (
    ab_test,
    ABTestResult,
    calculate_sample_size,
    sequential_test,
    multi_variant_test,
    conversion_rate_test,
)

# Cohort Analysis
from .cohort import (
    cohort_retention_analysis,
    CohortAnalysisResult,
    cohort_ltv_analysis,
    cohort_retention_rate_by_segment,
    identify_high_retention_cohorts,
    cohort_churn_analysis,
    cohort_reactivation_rate,
    compare_cohort_trends,
)

# Attribution
from .attribution import (
    last_touch_attribution,
    first_touch_attribution,
    linear_attribution,
    time_decay_attribution,
    position_based_attribution,
    compare_attribution_models,
    AttributionResult,
)

# Campaign Tracking
from .campaign import (
    campaign_performance_summary,
    year_over_year_comparison,
    wave_season_comparison,
    calculate_campaign_kpis,
    identify_top_campaigns,
    campaign_performance_by_device,
    detect_campaign_anomalies,
    CampaignMetrics,
)

# Conversion Funnel
from .funnel import (
    analyze_funnel,
    FunnelResult,
    analyze_funnel_by_cohort,
    identify_bottlenecks,
    compare_funnels,
)

from .anomaly_detection import (
    Anomaly,
    detect_value_anomalies,
    detect_pattern_anomalies,
    detect_sequence_anomalies,
    detect_all_anomalies,
    prioritize_anomalies,
)

# interpretation
from .interpretations import (
    Interpretation,
    interpret_ab_test,
    interpret_cohort_retention,
    interpret_revenue_trends,
    interpret_correlation,
    interpret_campaign_metrics,
)

# Utilities
from .utils import (
    clean_numeric_columns,
    auto_clean_dataframe,
    prepare_for_analysis,
    detect_column_roles,
)


__all__ = [
    # Summary
    'summarize_dataset',
    'summarize_numeric',
    'summarize_categorical',
    'quick_stats',
    'compare_periods',
    'top_performers',
    'distribution_stats',
    'DatasetSummary',
    'NumericSummary',
    'CategoricalSummary',
    
    # Revenue
    'calculate_revenue_metrics',
    'revenue_by_period',
    'revenue_by_dimension',
    'compare_periods_yoy',
    'growth_metrics',
    'pareto_analysis',
    'revenue_forecast_simple',
    'RevenueMetrics',
    
    # Product
    'product_performance',
    'top_products',
    'bottom_products',
    'product_comparison',
    'price_analysis',
    'product_trends',
    'product_basket_analysis',
    'category_performance',
    'product_performance_score',
    
    # Customer
    'customer_summary',
    'rfm_analysis',
    'customer_value_tiers',
    'customer_cohort_analysis',
    'customer_lifetime_value',
    'churn_risk_analysis',
    'CustomerMetrics',
    
    # Correlations
    'correlation_matrix',
    'find_strong_correlations',
    'correlation_with_target',
    'cross_tabulation',
    'group_comparison',
    'detect_outliers',
    'relationship_strength',
    'multi_variable_analysis',
    'CorrelationResult',
    'analyze_mixed_correlations',
    'find_all_relationships',
    'MixedCorrelationResult',

    #interpretation smart
    'Interpretation',
    'interpret_ab_test',
    'interpret_cohort_retention',
    'interpret_revenue_trends',
    'interpret_correlation',
    'interpret_campaign_metrics',

    #anomaly
    'Anomaly',
    'detect_value_anomalies',
    'detect_pattern_anomalies',
    'detect_sequence_anomalies',
    'detect_all_anomalies',
    'prioritize_anomalies',

    # Utilities
    'clean_numeric_columns',
    'auto_clean_dataframe',
    'prepare_for_analysis',
    'detect_column_roles',
]
