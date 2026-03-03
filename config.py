"""
Configuration settings for Local Analyst.
Centralized configuration for thresholds, styling, business rules, and AI settings.
"""

from typing import Dict, Any


# ==================== VISUALIZATION SETTINGS ====================

VISUALIZATION_CONFIG = {
    'color_palettes': {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'marketing': ['#0066CC', '#FF6B35', '#00C9A7', '#FF006E', '#9B5DE5'],
        'sequential': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],
        'risk': {'low': '#2ca02c', 'medium': '#ff7f0e', 'high': '#d62728'}
    },
    
    'chart_defaults': {
        'plotly': {
            'template': 'plotly_white',
            'height': 500,
            'font_family': 'Arial, sans-serif',
            'font_size': 12
        },
        'matplotlib': {
            'figure_size': (12, 6),
            'dpi': 100,
            'grid_alpha': 0.3,
            'font_size': 10
        }
    },
    
    'export': {
        'default_format': 'png',
        'default_dpi': 300,
        'image_quality': 95
    }
}


# ==================== ANALYSIS SETTINGS ====================

ANALYSIS_CONFIG = {
    'ab_testing': {
        'default_confidence_level': 0.95,
        'min_sample_size_per_variant': 100,
        'max_p_value': 0.05,
        'effect_size_thresholds': {
            'negligible': 0.2,
            'small': 0.5,
            'medium': 0.8
        }
    },
    
    'cohort_analysis': {
        'default_period': 'M',  # Monthly
        'max_periods_to_analyze': 12,
        'retention_threshold_good': 40,  # % retained
        'retention_threshold_poor': 20
    },
    
    'rfm_segmentation': {
        'recency_bins': 5,
        'frequency_bins': 5,
        'monetary_bins': 5,
        'segments': {
            'Champions': {'R': [4, 5], 'F': [4, 5], 'M': [4, 5]},
            'Loyal': {'R': [3, 4, 5], 'F': [3, 4, 5], 'M': [3, 4, 5]},
            'At Risk': {'R': [1, 2], 'F': [3, 4, 5], 'M': [3, 4, 5]},
            'Cant Lose': {'R': [1, 2], 'F': [4, 5], 'M': [4, 5]},
            'Lost': {'R': [1], 'F': [1, 2], 'M': [1, 2]}
        }
    },
    
    'correlation': {
        'methods': ['pearson', 'spearman', 'kendall'],
        'default_method': 'pearson',
        'strong_correlation_threshold': 0.7,
        'moderate_correlation_threshold': 0.4
    },
    
    'attribution': {
        'default_lookback_days': 30,
        'time_decay_half_life': 7,
        'position_based_weights': {
            'first_touch': 0.4,
            'last_touch': 0.4,
            'middle': 0.2
        }
    },
    
    'campaign': {
        'anomaly_detection_std_threshold': 2.0,
        'yoy_comparison_period': 'M',  # Monthly comparison
        'trend_window_days': 7
    }
}


# ==================== DATA QUALITY SETTINGS ====================

DATA_QUALITY_CONFIG = {
    'missing_data': {
        'max_missing_percentage': 50,  # Warn if > 50% missing
        'imputation_methods': ['mean', 'median', 'mode', 'forward_fill']
    },
    
    'outliers': {
        'detection_method': 'iqr',  # 'iqr' or 'zscore'
        'iqr_multiplier': 1.5,
        'zscore_threshold': 3.0
    },
    
    'validation': {
        'min_rows_for_analysis': 30,
        'max_unique_categories': 50,
        'date_formats': ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
    }
}


# ==================== BUSINESS RULES ====================

BUSINESS_RULES_CONFIG = {
    'kpi_benchmarks': {
        'email': {
            'open_rate': {'poor': 15, 'average': 20, 'good': 25},
            'click_rate': {'poor': 2, 'average': 3, 'good': 5}
        },
        'ecommerce': {
            'conversion_rate': {'poor': 1, 'average': 2, 'good': 4},
            'cart_abandonment': {'good': 60, 'average': 70, 'poor': 80},
            'avg_order_value': {'currency': 'USD'}
        },
        'digital_ads': {
            'ctr': {'poor': 1, 'average': 2, 'good': 3},
            'cvr': {'poor': 1, 'average': 2, 'good': 4},
            'cpc': {'currency': 'USD', 'poor': 5, 'average': 2, 'good': 1},
            'roas': {'poor': 2, 'average': 4, 'good': 6}
        }
    },
    
    'thresholds': {
        'high_value_customer': 1000,  # Revenue threshold
        'churn_risk_days': 90,  # Days inactive
        'pareto_concentration': 80,  # 80/20 rule
        'minimum_roi': 2.0,  # Minimum acceptable ROI
    }
}


# ==================== AI SETTINGS ====================

AI_CONFIG = {
    'enabled': False,  # Enable AI interpretation (requires Ollama)
    
    'ollama': {
        'model': 'llama2',
        'base_url': 'http://localhost:11434',
        'timeout': 30
    },
    
    'prompts': {
        'ab_test_interpretation': """Analyze this A/B test result and provide:
1. One-sentence summary
2. Key findings (2-3 bullets)
3. Business recommendations (2-3 bullets)

Be concise and actionable.""",
        
        'cohort_interpretation': """Analyze this cohort retention data and provide:
1. One-sentence summary
2. Key trends (2-3 bullets)
3. Retention improvement recommendations (2-3 bullets)

Be concise and actionable."""
    },
    
    'fallback_to_rules': True  # Use rule-based if AI fails
}


# ==================== FILE UPLOAD SETTINGS ====================

FILE_UPLOAD_CONFIG = {
    'max_file_size_mb': 100,
    'cache_enabled': True,
    'cache_ttl_hours': 24,
    
    'supported_formats': {
        'csv': ['.csv', '.tsv', '.txt'],
        'excel': ['.xlsx', '.xls', '.xlsm', '.xlsb'],
        'json': ['.json'],
        'pdf': ['.pdf'],
        'powerpoint': ['.pptx', '.ppt'],
        'word': ['.docx', '.doc']
    },
    
    'csv_options': {
        'auto_detect_delimiter': True,
        'auto_detect_encoding': True,
        'default_encoding': 'utf-8',
        'possible_delimiters': [',', ';', '\t', '|']
    },
    
    'excel_options': {
        'read_all_sheets': False,
        'default_sheet': 0
    }
}


# ==================== UI SETTINGS ====================

UI_CONFIG = {
    'page_title': 'Local Analyst',
    'page_icon': '📊',
    'layout': 'wide',
    
    'tabs': [
        {'name': 'Overview', 'icon': '📋'},
        {'name': 'Revenue', 'icon': '💰'},
        {'name': 'Products', 'icon': '📦'},
        {'name': 'Customers', 'icon': '👥'},
        {'name': 'Correlations', 'icon': '🔗'},
        {'name': 'A/B Testing', 'icon': '🧪'},
        {'name': 'Cohort Analysis', 'icon': '📊'},
        {'name': 'Campaign Tracking', 'icon': '🎯'}
    ],
    
    'theme': {
        'primary_color': '#1f77b4',
        'background_color': '#ffffff',
        'text_color': '#262730'
    }
}


# ==================== EXPORT CONFIGURATION ====================

def get_config(section: str) -> Dict[str, Any]:
    """
    Get configuration for specific section.
    
    Args:
        section: Config section name
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'visualization': VISUALIZATION_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'data_quality': DATA_QUALITY_CONFIG,
        'business_rules': BUSINESS_RULES_CONFIG,
        'ai': AI_CONFIG,
        'file_upload': FILE_UPLOAD_CONFIG,
        'ui': UI_CONFIG
    }
    
    return configs.get(section, {})


def update_config(section: str, key: str, value: Any) -> None:
    """
    Update configuration value.
    
    Args:
        section: Config section
        key: Config key
        value: New value
    """
    if section == 'visualization':
        VISUALIZATION_CONFIG[key] = value
    elif section == 'analysis':
        ANALYSIS_CONFIG[key] = value
    elif section == 'data_quality':
        DATA_QUALITY_CONFIG[key] = value
    elif section == 'business_rules':
        BUSINESS_RULES_CONFIG[key] = value
    elif section == 'ai':
        AI_CONFIG[key] = value
    elif section == 'file_upload':
        FILE_UPLOAD_CONFIG[key] = value
    elif section == 'ui':
        UI_CONFIG[key] = value


# Export all configs
__all__ = [
    'VISUALIZATION_CONFIG',
    'ANALYSIS_CONFIG',
    'DATA_QUALITY_CONFIG',
    'BUSINESS_RULES_CONFIG',
    'AI_CONFIG',
    'FILE_UPLOAD_CONFIG',
    'UI_CONFIG',
    'get_config',
    'update_config'
]
