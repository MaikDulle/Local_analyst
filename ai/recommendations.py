"""
Actionable recommendations generation for Local Analyst.
Provides specific next steps based on analysis results.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Recommendation:
    """A single actionable recommendation."""
    title: str
    description: str
    action_items: List[str]
    priority: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'optimization', 'investigation', 'expansion', 'reduction'
    estimated_impact: str  # 'high', 'medium', 'low'


def generate_ab_test_recommendations(
    lift_pct: float,
    p_value: float,
    is_significant: bool,
    sample_size: int,
    test_duration_days: Optional[int] = None
) -> List[Recommendation]:
    """
    Generate recommendations for A/B test.
    
    Args:
        lift_pct: Lift percentage
        p_value: P-value
        is_significant: Whether significant
        sample_size: Total sample size
        test_duration_days: Test duration
        
    Returns:
        List of Recommendations
    """
    recs = []
    
    # Implementation recommendation
    if is_significant and lift_pct > 0:
        if lift_pct > 10:
            recs.append(Recommendation(
                title="Immediate Implementation",
                description=f"Test shows strong positive lift of {lift_pct:.1f}%",
                action_items=[
                    "Implement winning variant immediately",
                    "Set up monitoring dashboard to track post-implementation performance",
                    "Document learnings for future tests"
                ],
                priority='critical',
                category='optimization',
                estimated_impact='high'
            ))
        else:
            recs.append(Recommendation(
                title="Gradual Rollout",
                description=f"Test shows modest lift of {lift_pct:.1f}%",
                action_items=[
                    "Implement with gradual rollout (10% → 50% → 100%)",
                    "Monitor key metrics during rollout",
                    "Prepare rollback plan if metrics degrade"
                ],
                priority='high',
                category='optimization',
                estimated_impact='medium'
            ))
    
    # Investigation recommendation
    elif not is_significant:
        if sample_size < 1000:
            recs.append(Recommendation(
                title="Increase Sample Size",
                description="Test inconclusive due to small sample",
                action_items=[
                    f"Continue test to reach at least 1,000 samples per variant (currently {sample_size})",
                    "Consider expanding test to more channels",
                    "Review test setup for any technical issues"
                ],
                priority='medium',
                category='investigation',
                estimated_impact='medium'
            ))
        else:
            recs.append(Recommendation(
                title="Test Alternative Variations",
                description="No significant difference detected",
                action_items=[
                    "Brainstorm more distinct variations",
                    "Consider testing different aspects of the experience",
                    "Review if chosen metric is sensitive enough"
                ],
                priority='medium',
                category='investigation',
                estimated_impact='low'
            ))
    
    # Negative result
    elif is_significant and lift_pct < 0:
        recs.append(Recommendation(
            title="Do Not Implement",
            description=f"Test shows negative impact of {abs(lift_pct):.1f}%",
            action_items=[
                "Abandon this variation",
                "Analyze why performance declined",
                "Share learnings with team to avoid similar issues"
            ],
            priority='critical',
            category='reduction',
            estimated_impact='high'
        ))
    
    return recs


def generate_cohort_recommendations(
    avg_retention_period_1: float,
    avg_retention_period_6: Optional[float],
    churn_rate: float
) -> List[Recommendation]:
    """
    Generate recommendations from cohort analysis.
    
    Args:
        avg_retention_period_1: Period 1 retention %
        avg_retention_period_6: Period 6 retention % (if available)
        churn_rate: Overall churn rate
        
    Returns:
        List of Recommendations
    """
    recs = []
    
    # Early retention
    if avg_retention_period_1 < 50:
        recs.append(Recommendation(
            title="Improve Onboarding Experience",
            description=f"Only {avg_retention_period_1:.0f}% retained after first period",
            action_items=[
                "Review and simplify onboarding flow",
                "Add in-app guidance for new users",
                "Implement welcome email series",
                "Survey churned users to understand pain points"
            ],
            priority='critical',
            category='optimization',
            estimated_impact='high'
        ))
    
    # Long-term retention
    if avg_retention_period_6 and avg_retention_period_6 < 20:
        recs.append(Recommendation(
            title="Implement Long-term Engagement Program",
            description=f"Retention drops to {avg_retention_period_6:.0f}% by period 6",
            action_items=[
                "Create loyalty or rewards program",
                "Develop content to keep users engaged long-term",
                "Implement win-back campaigns for dormant users",
                "Analyze what keeps successful cohorts engaged"
            ],
            priority='high',
            category='optimization',
            estimated_impact='high'
        ))
    
    # High churn
    if churn_rate > 50:
        recs.append(Recommendation(
            title="Address High Churn Rate",
            description=f"Churn rate of {churn_rate:.0f}% is concerning",
            action_items=[
                "Conduct churn analysis to identify patterns",
                "Interview churned customers",
                "Review competitor offerings",
                "Test retention incentives"
            ],
            priority='critical',
            category='investigation',
            estimated_impact='high'
        ))
    
    return recs


def generate_campaign_recommendations(
    current_roi: float,
    current_cac: Optional[float],
    yoy_change: Optional[float]
) -> List[Recommendation]:
    """
    Generate campaign recommendations.
    
    Args:
        current_roi: Current ROI
        current_cac: Customer acquisition cost
        yoy_change: Year-over-year change %
        
    Returns:
        List of Recommendations
    """
    recs = []
    
    # ROI optimization
    if current_roi < 2.0:
        recs.append(Recommendation(
            title="Improve Campaign ROI",
            description=f"Current ROI of {current_roi:.1f}x is below target",
            action_items=[
                "Analyze top-performing channels and reallocate budget",
                "Test new creative and messaging",
                "Review targeting to improve conversion rates",
                "Reduce spend on underperforming channels"
            ],
            priority='high',
            category='optimization',
            estimated_impact='high'
        ))
    
    # Scaling opportunity
    elif current_roi > 5.0:
        recs.append(Recommendation(
            title="Scale High-Performing Campaigns",
            description=f"Strong ROI of {current_roi:.1f}x indicates scaling opportunity",
            action_items=[
                "Increase budget on best-performing campaigns",
                "Expand to similar audiences",
                "Test new channels with similar characteristics",
                "Document success factors for replication"
            ],
            priority='high',
            category='expansion',
            estimated_impact='high'
        ))
    
    # YoY decline
    if yoy_change and yoy_change < -10:
        recs.append(Recommendation(
            title="Address Performance Decline",
            description=f"Performance down {abs(yoy_change):.0f}% year-over-year",
            action_items=[
                "Investigate cause of decline (market, creative fatigue, competition)",
                "Refresh creative assets",
                "Review targeting and bidding strategies",
                "Test new channels or tactics"
            ],
            priority='critical',
            category='investigation',
            estimated_impact='high'
        ))
    
    # CAC monitoring
    if current_cac:
        if current_cac > 100:  # Example threshold
            recs.append(Recommendation(
                title="Reduce Customer Acquisition Cost",
                description=f"CAC of ${current_cac:.2f} is high",
                action_items=[
                    "Focus on organic and referral channels",
                    "Improve landing page conversion rates",
                    "Optimize ad targeting to reduce waste",
                    "Test lower-funnel campaigns"
                ],
                priority='medium',
                category='optimization',
                estimated_impact='medium'
            ))
    
    return recs


def prioritize_recommendations(recommendations: List[Recommendation]) -> List[Recommendation]:
    """
    Sort recommendations by priority.
    
    Args:
        recommendations: List of recommendations
        
    Returns:
        Sorted list (highest priority first)
    """
    priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    
    return sorted(
        recommendations,
        key=lambda x: priority_order.get(x.priority, 0),
        reverse=True
    )


# Export
__all__ = [
    'Recommendation',
    'generate_ab_test_recommendations',
    'generate_cohort_recommendations',
    'generate_campaign_recommendations',
    'prioritize_recommendations'
]
