"""
Product performance analysis for ecom.
Covers top sellers, underperformers, pricing analysis, and product comparisons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ProductMetrics:
    """Metrics for a single product."""
    product_id: str
    revenue: float
    quantity: int
    avg_price: float
    transactions: int
    revenue_share: float
    quantity_share: float
    revenue_rank: int
    quantity_rank: int


def product_performance(
    df: pd.DataFrame,
    product_col: str,
    revenue_col: str,
    quantity_col: Optional[str] = None,
    top_n: Optional[int] = None,
    sort_by: str = 'revenue'
) -> pd.DataFrame:
    """
    Analyze product performance.
    
    Args:
        df: DataFrame with transaction data
        product_col: Column containing product identifier
        revenue_col: Column containing revenue
        quantity_col: Column containing quantity (optional)
        top_n: Limit to top N products
        sort_by: 'revenue', 'quantity', or 'transactions'
    
    Returns:
        DataFrame with product performance metrics
    """
    agg_dict = {revenue_col: 'sum'}
    
    if quantity_col and quantity_col in df.columns:
        agg_dict[quantity_col] = 'sum'
    
    grouped = df.groupby(product_col).agg(agg_dict)
    grouped.columns = ['revenue'] + (['quantity'] if quantity_col else [])
    
    # Transaction count
    grouped['transactions'] = df.groupby(product_col).size()
    
    # Average metrics
    if quantity_col and 'quantity' in grouped.columns:
        grouped['avg_price'] = (grouped['revenue'] / grouped['quantity']).round(2)
        grouped['avg_quantity_per_transaction'] = (grouped['quantity'] / grouped['transactions']).round(2)
    else:
        grouped['avg_price'] = (grouped['revenue'] / grouped['transactions']).round(2)
    
    grouped['avg_revenue_per_transaction'] = (grouped['revenue'] / grouped['transactions']).round(2)
    
    # Share calculations
    total_revenue = grouped['revenue'].sum()
    grouped['revenue_share_pct'] = (grouped['revenue'] / total_revenue * 100).round(2)
    
    if 'quantity' in grouped.columns:
        total_quantity = grouped['quantity'].sum()
        grouped['quantity_share_pct'] = (grouped['quantity'] / total_quantity * 100).round(2)
    
    # Rankings
    grouped['revenue_rank'] = grouped['revenue'].rank(ascending=False, method='min').astype(int)
    
    if 'quantity' in grouped.columns:
        grouped['quantity_rank'] = grouped['quantity'].rank(ascending=False, method='min').astype(int)
    
    # Sort
    if sort_by == 'quantity' and 'quantity' in grouped.columns:
        grouped = grouped.sort_values('quantity', ascending=False)
    elif sort_by == 'transactions':
        grouped = grouped.sort_values('transactions', ascending=False)
    else:
        grouped = grouped.sort_values('revenue', ascending=False)
    
    # Cumulative share for Pareto
    grouped['cumulative_revenue_share'] = grouped['revenue_share_pct'].cumsum().round(2)
    
    if top_n:
        grouped = grouped.head(top_n)
    
    return grouped.reset_index()


def top_products(
    df: pd.DataFrame,
    product_col: str,
    revenue_col: str,
    n: int = 10
) -> pd.DataFrame:
    """Get top N products by revenue."""
    return product_performance(df, product_col, revenue_col, top_n=n, sort_by='revenue')


def bottom_products(
    df: pd.DataFrame,
    product_col: str,
    revenue_col: str,
    n: int = 10,
    min_transactions: int = 1
) -> pd.DataFrame:
    """Get bottom N products by revenue (potential underperformers)."""
    perf = product_performance(df, product_col, revenue_col, sort_by='revenue')
    
    # Filter to products with minimum transactions
    perf = perf[perf['transactions'] >= min_transactions]
    
    # Get bottom N
    return perf.tail(n).sort_values('revenue', ascending=True)


def product_comparison(
    df: pd.DataFrame,
    product_col: str,
    revenue_col: str,
    products_to_compare: List[str],
    quantity_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare specific products side by side.
    
    Args:
        df: DataFrame
        product_col: Product column
        revenue_col: Revenue column
        products_to_compare: List of product IDs to compare
        quantity_col: Quantity column (optional)
    """
    df_filtered = df[df[product_col].isin(products_to_compare)]
    return product_performance(df_filtered, product_col, revenue_col, quantity_col)


def price_analysis(
    df: pd.DataFrame,
    product_col: str,
    price_col: str,
    quantity_col: Optional[str] = None,
    revenue_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze pricing across products.
    
    Returns DataFrame with price statistics per product.
    """
    agg_dict = {
        price_col: ['mean', 'min', 'max', 'std', 'count']
    }
    
    if quantity_col and quantity_col in df.columns:
        agg_dict[quantity_col] = 'sum'
    
    if revenue_col and revenue_col in df.columns:
        agg_dict[revenue_col] = 'sum'
    
    grouped = df.groupby(product_col).agg(agg_dict)
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    
    # Rename for clarity
    grouped = grouped.rename(columns={
        f'{price_col}_mean': 'avg_price',
        f'{price_col}_min': 'min_price',
        f'{price_col}_max': 'max_price',
        f'{price_col}_std': 'price_std',
        f'{price_col}_count': 'transactions'
    })
    
    # Price range
    grouped['price_range'] = grouped['max_price'] - grouped['min_price']
    grouped['price_cv'] = (grouped['price_std'] / grouped['avg_price']).round(3)  # Coefficient of variation
    
    # Flag products with high price variation
    grouped['high_price_variation'] = grouped['price_cv'] > 0.2
    
    return grouped.reset_index().sort_values('avg_price', ascending=False)


def product_trends(
    df: pd.DataFrame,
    date_col: str,
    product_col: str,
    revenue_col: str,
    period: str = 'M',
    top_n: int = 5
) -> pd.DataFrame:
    """
    Analyze product trends over time.
    
    Args:
        df: DataFrame
        date_col: Date column
        product_col: Product column
        revenue_col: Revenue column
        period: Time period ('D', 'W', 'M', 'Q')
        top_n: Number of top products to track
    
    Returns:
        Pivoted DataFrame with products as columns and periods as rows
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Get top N products by total revenue
    top_products = df.groupby(product_col)[revenue_col].sum().nlargest(top_n).index.tolist()
    
    # Filter to top products
    df_top = df[df[product_col].isin(top_products)]
    
    # Group by period and product
    df_top['period'] = df_top[date_col].dt.to_period(period)
    
    pivot = df_top.pivot_table(
        values=revenue_col,
        index='period',
        columns=product_col,
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate growth rates
    growth = pivot.pct_change() * 100
    growth.columns = [f'{col}_growth_pct' for col in growth.columns]
    
    # Combine
    result = pd.concat([pivot, growth], axis=1)
    result.index = result.index.astype(str)
    
    return result.reset_index().rename(columns={'period': date_col})


def product_basket_analysis(
    df: pd.DataFrame,
    order_col: str,
    product_col: str,
    min_support: float = 0.01
) -> pd.DataFrame:
    """
    Simple product co-occurrence analysis (which products are bought together).
    
    Args:
        df: DataFrame with order and product columns
        order_col: Column identifying unique orders
        product_col: Column identifying products
        min_support: Minimum support threshold (fraction of orders)
    
    Returns:
        DataFrame with product pairs and their co-occurrence frequency
    """
    # Get products per order
    orders = df.groupby(order_col)[product_col].apply(list)
    
    # Count pairs
    from collections import Counter
    pair_counts = Counter()
    
    for products in orders:
        products = list(set(products))  # Unique products per order
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                pair = tuple(sorted([products[i], products[j]]))
                pair_counts[pair] += 1
    
    total_orders = len(orders)
    min_count = int(total_orders * min_support)
    
    # Filter by min support
    pairs_df = pd.DataFrame([
        {'product_1': p[0], 'product_2': p[1], 'count': c, 'support': c / total_orders}
        for p, c in pair_counts.items()
        if c >= min_count
    ])
    
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=['product_1', 'product_2', 'count', 'support'])
    
    pairs_df['support_pct'] = (pairs_df['support'] * 100).round(2)
    
    return pairs_df.sort_values('count', ascending=False)


def category_performance(
    df: pd.DataFrame,
    category_col: str,
    revenue_col: str,
    product_col: Optional[str] = None,
    quantity_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze performance by product category.
    
    Args:
        df: DataFrame
        category_col: Category column
        revenue_col: Revenue column
        product_col: Product column (to count unique products per category)
        quantity_col: Quantity column (optional)
    """
    agg_dict = {revenue_col: 'sum'}
    
    if quantity_col and quantity_col in df.columns:
        agg_dict[quantity_col] = 'sum'
    
    grouped = df.groupby(category_col).agg(agg_dict)
    grouped.columns = ['revenue'] + (['quantity'] if quantity_col else [])
    
    # Count transactions and products
    grouped['transactions'] = df.groupby(category_col).size()
    
    if product_col and product_col in df.columns:
        grouped['unique_products'] = df.groupby(category_col)[product_col].nunique()
    
    # Averages
    grouped['avg_transaction_value'] = (grouped['revenue'] / grouped['transactions']).round(2)
    
    if 'unique_products' in grouped.columns:
        grouped['avg_revenue_per_product'] = (grouped['revenue'] / grouped['unique_products']).round(2)
    
    # Share
    total_revenue = grouped['revenue'].sum()
    grouped['revenue_share_pct'] = (grouped['revenue'] / total_revenue * 100).round(2)
    
    return grouped.sort_values('revenue', ascending=False).reset_index()


def product_performance_score(
    df: pd.DataFrame,
    product_col: str,
    revenue_col: str,
    quantity_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    return_rate_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate composite performance score for products.
    Combines revenue, quantity, rating, and return rate if available.
    
    Returns DataFrame with products and their performance scores (0-100).
    """
    perf = product_performance(df, product_col, revenue_col, quantity_col)
    
    # Normalize revenue share to 0-100
    perf['revenue_score'] = (perf['revenue_share_pct'] / perf['revenue_share_pct'].max() * 100).round(1)
    
    scores = ['revenue_score']
    
    # Add rating score if available
    if rating_col and rating_col in df.columns:
        ratings = df.groupby(product_col)[rating_col].mean()
        perf = perf.merge(ratings.rename('avg_rating'), left_on=product_col, right_index=True, how='left')
        perf['rating_score'] = (perf['avg_rating'] / 5 * 100).round(1)  # Assuming 5-point scale
        scores.append('rating_score')
    
    # Add return rate score (lower is better)
    if return_rate_col and return_rate_col in df.columns:
        returns = df.groupby(product_col)[return_rate_col].mean()
        perf = perf.merge(returns.rename('avg_return_rate'), left_on=product_col, right_index=True, how='left')
        # Invert: lower return rate = higher score
        perf['return_score'] = (100 - perf['avg_return_rate']).round(1)
        scores.append('return_score')
    
    # Calculate composite score (weighted average)
    weights = {'revenue_score': 0.5, 'rating_score': 0.3, 'return_score': 0.2}
    
    total_weight = sum(weights[s] for s in scores if s in weights)
    perf['performance_score'] = sum(
        perf[s] * weights.get(s, 0) / total_weight 
        for s in scores
    ).round(1)
    
    return perf.sort_values('performance_score', ascending=False)
