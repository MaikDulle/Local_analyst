

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Interpretation:
    """Analysis interpretation with recommendations."""
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    insights: List[str]
    warnings: List[str]


# ==================== A/B TESTING ====================

def interpret_ab_test(
    variant_a_name: str,
    variant_b_name: str,
    lift_pct: float,
    p_value: float,
    is_significant: bool,
    effect_size: str,
    sample_size_a: int,
    sample_size_b: int,
    confidence_level: float = 0.95
) -> Interpretation:
    """Interpret A/B test results."""
    
    # Summary
    if is_significant and lift_pct > 0:
        summary = f"✅ {variant_b_name} performs {abs(lift_pct):.1f}% better than {variant_a_name} (statistically significant)"
    elif is_significant and lift_pct < 0:
        summary = f"⚠️ {variant_b_name} performs {abs(lift_pct):.1f}% worse than {variant_a_name} (statistically significant)"
    else:
        summary = f"❌ No significant difference between {variant_a_name} and {variant_b_name}"
    
    # Key findings
    findings = []
    findings.append(f"Sample sizes: {sample_size_a:,} vs {sample_size_b:,} users")
    findings.append(f"Statistical significance: p = {p_value:.4f} ({'✓ significant' if is_significant else '✗ not significant'} at {confidence_level:.0%} confidence)")
    findings.append(f"Effect size: {effect_size}")
    
    if abs(lift_pct) > 20:
        findings.append("🎯 Large practical impact - this is a meaningful difference")
    elif abs(lift_pct) > 5:
        findings.append("📊 Moderate practical impact")
    else:
        findings.append("⚠️ Small practical impact - may not be worth implementing")
    
    # Recommendations
    recs = []
    if is_significant and lift_pct > 10:
        recs.append(f"✓ Implement {variant_b_name} immediately - strong positive results")
        recs.append("✓ Monitor key metrics closely for 1-2 weeks post-implementation")
        recs.append("✓ Document this win and share learnings with team")
    elif is_significant and lift_pct > 0:
        recs.append(f"✓ Consider implementing {variant_b_name} - modest but significant improvement")
        recs.append("✓ Weigh implementation cost against expected benefit")
        recs.append("✓ Consider gradual rollout (10% → 50% → 100%)")
    elif is_significant and lift_pct < 0:
        recs.append(f"✗ DO NOT implement {variant_b_name} - negative impact confirmed")
        recs.append("✓ Investigate why performance declined")
        recs.append("✓ Test alternative variations")
    else:
        recs.append("⏸️ No clear winner - continue testing or try different variants")
        if sample_size_a < 1000 or sample_size_b < 1000:
            recs.append("✓ Consider running test longer to gather more data")
        recs.append("✓ Test more distinct variations for clearer signal")
    
    # Insights
    insights = []
    if effect_size in ['large', 'very large']:
        insights.append("💡 Effect size is large - users will notice this difference")
    elif effect_size in ['negligible', 'small']:
        insights.append("💡 Effect size is small - difference may be hard to perceive")
    
    if sample_size_a < 100 or sample_size_b < 100:
        insights.append("⚠️ Sample size is small - results may be unstable")
    elif sample_size_a > 10000 or sample_size_b > 10000:
        insights.append("✓ Large sample size provides high confidence in results")
    
    # Warnings
    warnings = []
    if not is_significant and abs(lift_pct) > 5:
        warnings.append("⚠️ Trend shows difference, but not statistically significant - may need more data")
    
    if sample_size_a != sample_size_b and abs(sample_size_a - sample_size_b) > min(sample_size_a, sample_size_b) * 0.2:
        warnings.append("⚠️ Unbalanced sample sizes - ensure random assignment was correct")
    
    return Interpretation(summary, findings, recs, insights, warnings)


# ==================== COHORT ANALYSIS ====================

def interpret_cohort_retention(
    avg_retention_by_period: Dict[int, float],
    total_cohorts: int,
    total_customers: int
) -> Interpretation:
    """Interpret cohort retention analysis."""
    
    periods = sorted(avg_retention_by_period.keys())
    retention_values = [avg_retention_by_period[p] for p in periods]
    
    if len(retention_values) < 2:
        return Interpretation(
            "Need more data to analyze retention trends",
            ["Not enough time periods to analyze"],
            ["Continue collecting data over more periods"],
            [],
            []
        )
    
    # Calculate retention metrics
    period_0_retention = retention_values[0]
    final_retention = retention_values[-1]
    retention_drop = period_0_retention - final_retention
    
    # Summary
    if retention_drop < 20:
        summary = f"✅ Strong retention: Only {retention_drop:.1f}% drop from period 0 to period {periods[-1]}"
    elif retention_drop < 50:
        summary = f"⚠️ Moderate retention: {retention_drop:.1f}% drop over {len(periods)} periods"
    else:
        summary = f"🔴 Poor retention: {retention_drop:.1f}% drop - urgent action needed"
    
    # Key findings
    findings = []
    findings.append(f"Analyzed {total_cohorts} cohorts with {total_customers:,} total customers")
    findings.append(f"Period 0 retention: {period_0_retention:.1f}%")
    findings.append(f"Period {periods[-1]} retention: {final_retention:.1f}%")
    
    # Check retention curve shape
    if len(retention_values) >= 3:
        early_drop = retention_values[0] - retention_values[1]
        late_drop = retention_values[-2] - retention_values[-1]
        
        if early_drop > late_drop * 2:
            findings.append("📉 Steep early drop-off - onboarding may be the issue")
        elif late_drop > early_drop * 2:
            findings.append("📉 Retention degrades over time - long-term engagement issue")
        else:
            findings.append("📊 Consistent drop-off rate across periods")
    
    # Recommendations
    recs = []
    if retention_drop > 60:
        recs.append("🚨 CRITICAL: Implement emergency retention program")
        recs.append("✓ Survey churned users immediately to understand why they left")
        recs.append("✓ Review onboarding flow and first-week experience")
    elif retention_drop > 40:
        recs.append("⚠️ Focus on retention improvement initiatives")
        recs.append("✓ Implement re-engagement campaigns for at-risk users")
        recs.append("✓ Analyze behavior of retained vs churned users")
    else:
        recs.append("✓ Retention is healthy - focus on maintaining quality")
        recs.append("✓ Test loyalty programs to further improve long-term retention")
        recs.append("✓ Benchmark against industry standards")
    
    # Specific onboarding rec if early drop is severe
    if len(retention_values) >= 2 and (retention_values[0] - retention_values[1]) > 30:
        recs.append("🎯 PRIORITY: Fix onboarding - losing 30%+ after first period")
    
    # Insights
    insights = []
    if final_retention > 50:
        insights.append("💡 Strong long-term retention indicates product-market fit")
    elif final_retention > 30:
        insights.append("💡 Moderate long-term retention - room for improvement")
    else:
        insights.append("💡 Low long-term retention - fundamental product/market issue")
    
    # Calculate retention half-life (period where retention drops to 50%)
    for i, val in enumerate(retention_values):
        if val < 50:
            insights.append(f"📊 Retention half-life: ~{periods[i]} periods")
            break
    
    # Warnings
    warnings = []
    if total_cohorts < 5:
        warnings.append("⚠️ Small number of cohorts - results may be unstable")
    
    if period_0_retention < 80:
        warnings.append("⚠️ Low initial retention - check data quality or definition")
    
    return Interpretation(summary, findings, recs, insights, warnings)


# ==================== REVENUE ANALYSIS ====================

def interpret_revenue_trends(
    revenue_data: pd.DataFrame,
    date_col: str,
    revenue_col: str
) -> Interpretation:
    """Interpret revenue trends."""
    
    # Calculate growth
    recent_revenue = revenue_data[revenue_col].tail(7).mean()
    previous_revenue = revenue_data[revenue_col].head(7).mean()
    
    growth_pct = ((recent_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
    
    # Summary
    if growth_pct > 20:
        summary = f"🚀 Strong growth: Revenue up {growth_pct:.1f}% (${recent_revenue:,.0f} recent vs ${previous_revenue:,.0f} earlier)"
    elif growth_pct > 0:
        summary = f"📈 Positive growth: Revenue up {growth_pct:.1f}%"
    elif growth_pct > -10:
        summary = f"📊 Stable: Revenue change {growth_pct:+.1f}%"
    else:
        summary = f"📉 Declining: Revenue down {abs(growth_pct):.1f}%"
    
    # Key findings
    findings = []
    findings.append(f"Recent period average: ${recent_revenue:,.0f}")
    findings.append(f"Earlier period average: ${previous_revenue:,.0f}")
    findings.append(f"Total revenue: ${revenue_data[revenue_col].sum():,.0f}")
    
    # Check volatility
    std_dev = revenue_data[revenue_col].std()
    mean_rev = revenue_data[revenue_col].mean()
    cv = (std_dev / mean_rev * 100) if mean_rev > 0 else 0
    
    if cv > 30:
        findings.append(f"⚠️ High volatility: {cv:.0f}% coefficient of variation")
    elif cv < 10:
        findings.append(f"✓ Stable revenue: {cv:.0f}% coefficient of variation")
    
    # Recommendations
    recs = []
    if growth_pct > 10:
        recs.append("✓ Identify and double down on growth drivers")
        recs.append("✓ Ensure infrastructure can scale with growth")
    elif growth_pct < -10:
        recs.append("🚨 Investigate causes of revenue decline immediately")
        recs.append("✓ Review customer feedback and churn data")
        recs.append("✓ Consider promotional campaigns to reverse trend")
    else:
        recs.append("✓ Focus on optimizing margins and efficiency")
        recs.append("✓ Test new growth initiatives")
    
    # Insights
    insights = []
    if cv > 40:
        insights.append("💡 Revenue is highly unpredictable - investigate causes")
    
    # Warnings
    warnings = []
    if len(revenue_data) < 30:
        warnings.append("⚠️ Limited data points - trends may not be reliable")
    
    return Interpretation(summary, findings, recs, insights, warnings)


# ==================== CORRELATION ANALYSIS ====================

def interpret_correlation(
    var1: str,
    var2: str,
    correlation: float,
    sample_size: int
) -> Interpretation:
    """Interpret correlation between two variables."""
    
    abs_corr = abs(correlation)
    
    # Strength classification
    if abs_corr > 0.7:
        strength = "Strong"
    elif abs_corr > 0.4:
        strength = "Moderate"
    elif abs_corr > 0.2:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    direction = "positive" if correlation > 0 else "negative"
    
    # Summary
    if abs_corr > 0.5:
        summary = f"🔗 {strength} {direction} correlation between {var1} and {var2} (r = {correlation:.3f})"
    else:
        summary = f"📊 {strength} relationship between {var1} and {var2} (r = {correlation:.3f})"
    
    # Key findings
    findings = []
    findings.append(f"Correlation coefficient: {correlation:.3f}")
    findings.append(f"Strength: {strength}")
    findings.append(f"Direction: {direction}")
    findings.append(f"Sample size: {sample_size:,}")
    
    # Explained variance
    r_squared = correlation ** 2
    findings.append(f"Shared variance: {r_squared:.1%} (R² = {r_squared:.3f})")
    
    # Recommendations
    recs = []
    if abs_corr > 0.7:
        recs.append(f"✓ {var1} and {var2} move together strongly")
        if correlation > 0:
            recs.append(f"✓ Increasing {var1} is associated with increasing {var2}")
        else:
            recs.append(f"✓ Increasing {var1} is associated with decreasing {var2}")
        recs.append("⚠️ Remember: Correlation ≠ Causation")
    elif abs_corr > 0.3:
        recs.append(f"✓ Some relationship exists - consider deeper analysis")
        recs.append("✓ Look for confounding variables")
    else:
        recs.append(f"✓ Little to no linear relationship")
        recs.append("✓ Variables may be independent or have non-linear relationship")
    
    # Insights
    insights = []
    if abs_corr > 0.5:
        insights.append(f"💡 {var2} could potentially be predicted from {var1} (and vice versa)")
    
    if correlation > 0.8:
        insights.append("💡 Very high correlation - check if these measure similar things")
    
    # Warnings
    warnings = []
    if sample_size < 30:
        warnings.append("⚠️ Small sample size - correlation may be unreliable")
    
    if abs_corr > 0.95:
        warnings.append("⚠️ Extremely high correlation - may indicate measurement redundancy")
    
    return Interpretation(summary, findings, recs, insights, warnings)


# ==================== CAMPAIGN PERFORMANCE ====================

def interpret_campaign_metrics(
    metrics_summary: Dict[str, Any]
) -> Interpretation:
    """Interpret campaign performance metrics."""
    
    # Extract metrics (this is flexible based on what's available)
    ctr = metrics_summary.get('ctr', 0)
    cvr = metrics_summary.get('cvr', 0)
    roas = metrics_summary.get('roas', 0)
    total_spent = metrics_summary.get('total_spent', 0)
    total_revenue = metrics_summary.get('total_revenue', 0)
    
    # Summary
    if roas > 4:
        summary = f"🚀 Excellent campaign performance: {roas:.1f}x ROAS"
    elif roas > 2:
        summary = f"✅ Good campaign performance: {roas:.1f}x ROAS"
    elif roas > 1:
        summary = f"⚠️ Profitable but low margin: {roas:.1f}x ROAS"
    else:
        summary = f"🔴 Unprofitable campaign: {roas:.1f}x ROAS"
    
    # Key findings
    findings = []
    if ctr > 0:
        if ctr > 3:
            findings.append(f"✓ Strong CTR: {ctr:.2f}% (above average)")
        elif ctr > 1:
            findings.append(f"📊 Average CTR: {ctr:.2f}%")
        else:
            findings.append(f"⚠️ Low CTR: {ctr:.2f}% (needs improvement)")
    
    if cvr > 0:
        if cvr > 3:
            findings.append(f"✓ High conversion: {cvr:.2f}%")
        elif cvr > 1:
            findings.append(f"📊 Average conversion: {cvr:.2f}%")
        else:
            findings.append(f"⚠️ Low conversion: {cvr:.2f}%")
    
    if total_spent > 0 and total_revenue > 0:
        findings.append(f"Total spent: ${total_spent:,.0f}")
        findings.append(f"Total revenue: ${total_revenue:,.0f}")
        findings.append(f"Net profit: ${(total_revenue - total_spent):,.0f}")
    
    # Recommendations
    recs = []
    if roas > 3:
        recs.append("✓ Scale this campaign - it's performing well")
        recs.append("✓ Allocate more budget to top-performing segments")
    elif roas > 1:
        recs.append("✓ Optimize to improve ROAS before scaling")
        recs.append("✓ Test new creative and targeting")
    else:
        recs.append("🚨 Pause or restructure this campaign")
        recs.append("✓ Review targeting and creative strategy")
        recs.append("✓ Consider different channels")
    
    if ctr > 0 and ctr < 1:
        recs.append("🎯 Improve ad creative to boost CTR")
    
    if cvr > 0 and cvr < 1:
        recs.append("🎯 Optimize landing page to improve conversion")
    
    # Insights
    insights = []
    if ctr > 3 and cvr < 1:
        insights.append("💡 Good ad engagement but poor conversion - landing page issue")
    elif ctr < 1 and cvr > 3:
        insights.append("💡 Poor ad engagement but good conversion - targeting issue")
    
    # Warnings
    warnings = []
    if roas < 1:
        warnings.append("⚠️ Campaign is losing money - immediate action required")
    
    return Interpretation(summary, findings, recs, insights, warnings)


# Export all functions
__all__ = [
    'Interpretation',
    'interpret_ab_test',
    'interpret_cohort_retention',
    'interpret_revenue_trends',
    'interpret_correlation',
    'interpret_campaign_metrics',
]

