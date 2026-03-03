"""
A/B Testing analysis for Local Analyst.
Statistical significance testing with t-tests, confidence intervals, and effect sizes.
Integrates with existing analysis_engine pattern.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    variant_a_name: str
    variant_b_name: str
    variant_a_count: int
    variant_b_count: int
    variant_a_mean: float
    variant_b_mean: float
    variant_a_std: float
    variant_b_std: float
    lift_percentage: float
    p_value: float
    t_statistic: float
    confidence_level: float
    is_significant: bool
    cohens_d: float
    effect_size_interpretation: str
    confidence_interval_a: Tuple[float, float]
    confidence_interval_b: Tuple[float, float]
    recommendation: str


def ab_test(
    df: pd.DataFrame,
    variant_col: str,
    metric_col: str,
    variant_a: str,
    variant_b: str,
    confidence_level: float = 0.95
) -> ABTestResult:
    """
    Perform A/B test with statistical significance testing.
    
    Args:
        df: DataFrame with test data
        variant_col: Column containing variant labels
        metric_col: Column containing metric to test
        variant_a: Name of control variant
        variant_b: Name of treatment variant
        confidence_level: Confidence level for statistical test (default 0.95)
        
    Returns:
        ABTestResult with complete test statistics
    """
    # Filter to variants
    group_a = df[df[variant_col] == variant_a][metric_col].dropna()
    group_b = df[df[variant_col] == variant_b][metric_col].dropna()
    
    # Basic statistics
    a_count = len(group_a)
    b_count = len(group_b)
    a_mean = group_a.mean()
    b_mean = group_b.mean()
    a_std = group_a.std()
    b_std = group_b.std()
    
    # Lift calculation
    lift = ((b_mean - a_mean) / a_mean * 100) if a_mean != 0 else 0
    
    # T-test (Welch's t-test, doesn't assume equal variances)
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    # Statistical significance
    alpha = 1 - confidence_level
    is_significant = p_value < alpha
    
    # Confidence intervals
    ci_a = stats.t.interval(
        confidence_level,
        len(group_a) - 1,
        loc=a_mean,
        scale=stats.sem(group_a)
    )
    
    ci_b = stats.t.interval(
        confidence_level,
        len(group_b) - 1,
        loc=b_mean,
        scale=stats.sem(group_b)
    )
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((a_count - 1) * a_std**2 + (b_count - 1) * b_std**2) / 
        (a_count + b_count - 2)
    )
    cohens_d = (b_mean - a_mean) / pooled_std if pooled_std != 0 else 0
    
    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_interpretation = 'negligible'
    elif abs_d < 0.5:
        effect_interpretation = 'small'
    elif abs_d < 0.8:
        effect_interpretation = 'medium'
    else:
        effect_interpretation = 'large'
    
    # Generate recommendation
    if is_significant and lift > 0:
        recommendation = f"✅ Implement {variant_b}. Statistically significant improvement of {lift:.2f}%."
    elif is_significant and lift < 0:
        recommendation = f"❌ Do NOT implement {variant_b}. Statistically significant decline of {abs(lift):.2f}%."
    else:
        recommendation = f"⚠️ Insufficient evidence. Consider running test longer or increasing sample size."
    
    return ABTestResult(
        variant_a_name=variant_a,
        variant_b_name=variant_b,
        variant_a_count=a_count,
        variant_b_count=b_count,
        variant_a_mean=a_mean,
        variant_b_mean=b_mean,
        variant_a_std=a_std,
        variant_b_std=b_std,
        lift_percentage=lift,
        p_value=p_value,
        t_statistic=t_stat,
        confidence_level=confidence_level,
        is_significant=is_significant,
        cohens_d=cohens_d,
        effect_size_interpretation=effect_interpretation,
        confidence_interval_a=ci_a,
        confidence_interval_b=ci_b,
        recommendation=recommendation
    )


def calculate_sample_size(
    baseline_conversion: float,
    minimum_detectable_effect: float,
    confidence_level: float = 0.95,
    power: float = 0.80
) -> int:
    """
    Calculate required sample size for A/B test.
    
    Args:
        baseline_conversion: Current conversion rate (e.g., 0.05 for 5%)
        minimum_detectable_effect: Smallest effect to detect (e.g., 0.20 for 20% lift)
        confidence_level: Confidence level (default 0.95)
        power: Statistical power (default 0.80)
        
    Returns:
        Required sample size per variant
    """
    from scipy.stats import norm
    
    alpha = 1 - confidence_level
    beta = 1 - power
    
    # Z-scores
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    # Expected conversion rates
    p1 = baseline_conversion
    p2 = p1 * (1 + minimum_detectable_effect)
    
    # Pooled proportion
    p_pooled = (p1 + p2) / 2
    
    # Sample size formula
    n = (
        (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
    ) / (p2 - p1)**2
    
    return int(np.ceil(n))


def sequential_test(
    df: pd.DataFrame,
    variant_col: str,
    metric_col: str,
    variant_a: str,
    variant_b: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Sequential probability ratio test for early stopping.
    
    Allows you to check significance at multiple points during test.
    
    Args:
        df: DataFrame with cumulative test data
        variant_col: Variant column name
        metric_col: Metric column name
        variant_a: Control variant
        variant_b: Treatment variant
        alpha: Significance level
        
    Returns:
        Dict with sequential test results
    """
    result = ab_test(df, variant_col, metric_col, variant_a, variant_b)
    
    # Bonferroni correction for multiple testing
    # Assume maximum 10 peeks
    adjusted_alpha = alpha / 10
    
    return {
        'can_stop_for_winner': result.p_value < adjusted_alpha and result.lift_percentage > 0,
        'can_stop_for_loser': result.p_value < adjusted_alpha and result.lift_percentage < 0,
        'should_continue': result.p_value >= adjusted_alpha,
        'current_p_value': result.p_value,
        'adjusted_alpha': adjusted_alpha,
        'current_sample_size': result.variant_a_count + result.variant_b_count,
        'note': 'Using Bonferroni correction for 10 maximum peeks'
    }


def multi_variant_test(
    df: pd.DataFrame,
    variant_col: str,
    metric_col: str,
    control_variant: str
) -> pd.DataFrame:
    """
    Test multiple variants against a control.
    
    Args:
        df: DataFrame with test data
        variant_col: Variant column name
        metric_col: Metric column name
        control_variant: Name of control variant
        
    Returns:
        DataFrame with results for all variants vs control
    """
    variants = df[variant_col].unique()
    test_variants = [v for v in variants if v != control_variant]
    
    results = []
    
    for variant in test_variants:
        result = ab_test(df, variant_col, metric_col, control_variant, variant)
        
        results.append({
            'variant': variant,
            'sample_size': result.variant_b_count,
            'mean': result.variant_b_mean,
            'lift_pct': result.lift_percentage,
            'p_value': result.p_value,
            'is_significant': result.is_significant,
            'effect_size': result.cohens_d,
            'recommendation': result.recommendation
        })
    
    return pd.DataFrame(results).sort_values('lift_pct', ascending=False)


def conversion_rate_test(
    conversions_a: int,
    total_a: int,
    conversions_b: int,
    total_b: int,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Test difference in conversion rates using proportion test.
    
    Useful when you have conversion counts rather than raw data.
    
    Args:
        conversions_a: Number of conversions in variant A
        total_a: Total users in variant A
        conversions_b: Number of conversions in variant B
        total_b: Total users in variant B
        confidence_level: Confidence level
        
    Returns:
        Dict with test results
    """
    # Conversion rates
    rate_a = conversions_a / total_a if total_a > 0 else 0
    rate_b = conversions_b / total_b if total_b > 0 else 0
    
    # Pooled proportion
    pooled_prop = (conversions_a + conversions_b) / (total_a + total_b)
    
    # Standard error
    se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/total_a + 1/total_b))
    
    # Z-statistic
    z_stat = (rate_b - rate_a) / se if se > 0 else 0
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Lift
    lift = ((rate_b - rate_a) / rate_a * 100) if rate_a > 0 else 0
    
    # Significance
    alpha = 1 - confidence_level
    is_significant = p_value < alpha
    
    return {
        'conversion_rate_a': rate_a,
        'conversion_rate_b': rate_b,
        'lift_percentage': lift,
        'z_statistic': z_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'confidence_level': confidence_level,
        'conversions_a': conversions_a,
        'total_a': total_a,
        'conversions_b': conversions_b,
        'total_b': total_b
    }


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    n = 1000
    control = pd.DataFrame({
        'variant': ['A'] * n,
        'converted': np.random.binomial(1, 0.05, n)
    })
    
    treatment = pd.DataFrame({
        'variant': ['B'] * n,
        'converted': np.random.binomial(1, 0.06, n)
    })
    
    df = pd.concat([control, treatment], ignore_index=True)
    
    # Run test
    result = ab_test(df, 'variant', 'converted', 'A', 'B')
    
    print(f"Sample sizes: A={result.variant_a_count}, B={result.variant_b_count}")
    print(f"Conversion rates: A={result.variant_a_mean:.4f}, B={result.variant_b_mean:.4f}")
    print(f"Lift: {result.lift_percentage:.2f}%")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Significant: {result.is_significant}")
    print(f"Effect size: {result.cohens_d:.3f} ({result.effect_size_interpretation})")
    print(f"\nRecommendation: {result.recommendation}")
