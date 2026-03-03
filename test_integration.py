"""
Test script to verify Local Analyst integration.
Run this after completing all integration steps.

Usage:
    python test_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("LOCAL ANALYST - INTEGRATION TEST SUITE")
print("=" * 70)

# Test 1: Import data_upload_engine
print("\n[1/8] Testing data_upload_engine imports...")
try:
    from data_upload_engine import load_file, get_supported_extensions
    from data_upload_engine.docx_loader import load_docx
    
    extensions = get_supported_extensions()
    assert '.docx' in extensions, "Word support not added!"
    assert '.doc' in extensions, "Word support not added!"
    
    print("✅ PASS: Word document support available")
    print(f"   Supported extensions: {', '.join(extensions)}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 2: Import analysis_engine modules
print("\n[2/8] Testing analysis_engine imports...")
try:
    from analysis_engine import (
        # Existing
        rfm_analysis,
        customer_summary,
        # New
        ab_test,
        cohort_retention_analysis,
        wave_season_comparison,
        last_touch_attribution,
    )
    print("✅ PASS: All analysis modules imported successfully")
    print("   - A/B Testing: ab_test")
    print("   - Cohort Analysis: cohort_retention_analysis")
    print("   - Campaign Tracking: wave_season_comparison")
    print("   - Attribution: last_touch_attribution")
except Exception as e:
    print(f"❌ FAIL: {e}")
    print("   Check analysis_engine/__init__.py for missing imports")
    sys.exit(1)

# Test 3: Import viz_engine
print("\n[3/8] Testing viz_engine imports...")
try:
    from viz_engine import (
        plot_campaign_performance,
        plot_cohort_heatmap,
        plot_ab_test_comparison,
        plot_rfm_scatter,
    )
    print("✅ PASS: Visualization engine loaded")
    print("   - Campaign viz")
    print("   - Cohort heatmaps")
    print("   - A/B test charts")
    print("   - RFM visualizations")
except Exception as e:
    print(f"❌ FAIL: {e}")
    print("   Check viz_engine/__init__.py and plots.py")
    sys.exit(1)

# Test 4: Test A/B testing with sample data
print("\n[4/8] Testing A/B testing module...")
try:
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'variant': ['A'] * 100 + ['B'] * 100,
        'conversion': np.concatenate([
            np.random.binomial(1, 0.05, 100),
            np.random.binomial(1, 0.06, 100)
        ])
    })
    
    result = ab_test(test_df, 'variant', 'conversion', 'A', 'B')
    
    assert result.variant_a_name == 'A'
    assert result.variant_b_name == 'B'
    assert result.p_value >= 0 and result.p_value <= 1
    assert hasattr(result, 'recommendation')
    
    print("✅ PASS: A/B testing calculations working")
    print(f"   Sample test: {result.variant_a_count} vs {result.variant_b_count} samples")
    print(f"   P-value: {result.p_value:.4f}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 5: Test Cohort analysis
print("\n[5/8] Testing cohort analysis module...")
try:
    # Generate sample transaction data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    transactions = []
    for customer_id in range(1, 51):
        first_purchase = np.random.choice(dates[:180])
        n_purchases = np.random.randint(1, 5)
        
        for i in range(n_purchases):
            purchase_date = first_purchase + pd.Timedelta(days=i*30)
            if purchase_date <= dates[-1]:
                transactions.append({
                    'customer_id': f'CUST_{customer_id:03d}',
                    'date': purchase_date,
                    'revenue': np.random.uniform(50, 500)
                })
    
    cohort_df = pd.DataFrame(transactions)
    
    result = cohort_retention_analysis(
        cohort_df,
        customer_col='customer_id',
        date_col='date',
        period='M',
        value_col='revenue'
    )
    
    assert result.metadata['total_cohorts'] > 0
    assert not result.retention_matrix.empty
    
    print("✅ PASS: Cohort analysis working")
    print(f"   Analyzed {result.metadata['total_cohorts']} cohorts")
    print(f"   Total customers: {result.metadata['total_customers']}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 6: Test Campaign tracking
print("\n[6/8] Testing campaign tracking module...")
try:
    campaign_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', '2024-12-31', freq='D'),
        'impressions': np.random.poisson(10000, 365),
        'clicks': np.random.poisson(500, 365),
        'conversions': np.random.poisson(25, 365)
    })
    
    yoy_result = year_over_year_comparison(
        pd.concat([
            campaign_df.assign(year=2024),
            campaign_df.assign(year=2023, date=campaign_df['date'] - pd.DateOffset(years=1))
        ]),
        'date',
        ['impressions', 'clicks'],
        2024,
        2023,
        'M'
    )
    
    assert not yoy_result.empty
    assert 'impressions_change_pct' in yoy_result.columns
    
    print("✅ PASS: Campaign tracking working")
    print(f"   YoY comparison: {len(yoy_result)} periods analyzed")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 7: Test Attribution
print("\n[7/8] Testing attribution module...")
try:
    touchpoints = []
    channels = ['Google', 'Facebook', 'Email', 'Direct']
    
    for customer_id in range(1, 21):
        journey_length = np.random.randint(1, 4)
        for i in range(journey_length):
            touchpoints.append({
                'customer_id': f'CUST_{customer_id:03d}',
                'channel': np.random.choice(channels),
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i*7),
                'conversion': 1 if i == journey_length - 1 else 0
            })
    
    attribution_df = pd.DataFrame(touchpoints)
    
    result = last_touch_attribution(
        attribution_df,
        'customer_id',
        'channel',
        'conversion',
        'date'
    )
    
    assert result.model_name == 'Last Touch'
    assert not result.channel_attribution.empty
    
    print("✅ PASS: Attribution modeling working")
    print(f"   {len(result.channel_attribution)} channels attributed")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Test 8: Test Visualization
print("\n[8/8] Testing visualization module...")
try:
    # Just test that functions are callable
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', '2024-01-31'),
        'clicks': np.random.randint(100, 200, 31)
    })
    
    fig = plot_campaign_performance(test_df, 'date', ['clicks'], interactive=True)
    
    assert fig is not None
    assert hasattr(fig, 'data')  # Plotly figure
    
    print("✅ PASS: Visualizations working")
    print("   Plotly charts available")
except Exception as e:
    print(f"❌ FAIL: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print("\n✅ Integration complete! Your Local Analyst is ready to use.")
print("\nTo run the application:")
print("   streamlit run app/main.py")
print("\n" + "=" * 70)
