#!/usr/bin/env python
"""
Local Analyst CLI - Quick testing from command line.

Usage:
    python cli.py load <file>              Load and profile a file
    python cli.py analyze <file>           Run full analysis
    python cli.py summary <file>           Quick summary stats
    python cli.py correlations <file>      Find correlations
    python cli.py tables <file>            List tables (PDF/PPTX)
    python cli.py kpis <file>              Extract KPIs (PDF/PPTX)

Examples:
    python cli.py load data/samples/Q4_Report.pdf
    python cli.py analyze sales_data.csv
    python cli.py summary transactions.xlsx
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def cmd_load(args):
    """Load and profile a file."""
    from data_upload_engine import load_file, get_file_info
    
    print(f"\n{'='*60}")
    print(f"Loading: {args.file}")
    print('='*60)
    
    # File info first
    info = get_file_info(args.file)
    print(f"\nFile: {info.get('file_name')}")
    print(f"Size: {info.get('file_size_kb')} KB")
    print(f"Type: {info.get('file_type', 'unknown')}")
    
    # Full load
    result = load_file(args.file)
    
    if not result.success:
        print(f"\n✗ Error: {result.error}")
        return 1
    
    print(f"\n✓ Loaded successfully")
    print(f"  Is document: {result.is_document}")
    print(f"  DataFrame shape: {result.df.shape}")
    
    if result.profile:
        print(f"  Quality score: {result.profile.quality_score}")
        print(f"  Missing data: {result.profile.total_missing_pct}%")
        
        if result.profile.insights:
            print(f"\nInsights:")
            for insight in result.profile.insights:
                print(f"  {insight}")
    
    if result.ecom_mapping and any(result.ecom_mapping.values()):
        print(f"\nSuggested column mapping:")
        for role, col in result.ecom_mapping.items():
            if col:
                print(f"  {role}: {col}")
    
    # Show data preview
    if not result.df.empty:
        print(f"\nData preview:")
        print(result.df.head(10).to_string())
    
    return 0


def cmd_analyze(args):
    """Run full analysis on a file."""
    from data_upload_engine import load_file
    from analysis_engine import (
        prepare_for_analysis, 
        summarize_dataset,
        find_strong_correlations,
        detect_column_roles
    )
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {args.file}")
    print('='*60)
    
    result = load_file(args.file)
    
    if not result.success:
        print(f"\n✗ Error: {result.error}")
        return 1
    
    # Get DataFrame (from document tables or direct)
    if result.is_document and result.document_content:
        tables = result.document_content.tables
        if tables:
            print(f"\nFound {len(tables)} tables in document")
            for i, table in enumerate(tables):
                print(f"\n{'='*40}")
                print(f"Table {i+1}: {table.shape[0]} rows × {table.shape[1]} cols")
                print('='*40)
                
                df = prepare_for_analysis(table)
                _analyze_dataframe(df)
        else:
            print("\nNo tables found in document")
            print(f"\nText preview:\n{result.document_content.text[:500]}...")
    else:
        df = prepare_for_analysis(result.df)
        _analyze_dataframe(df)
    
    return 0


def _analyze_dataframe(df):
    """Analyze a single DataFrame."""
    from analysis_engine import (
        summarize_dataset,
        find_strong_correlations,
        detect_column_roles
    )
    
    # Show data
    print(f"\nData:")
    print(df.to_string())
    
    # Column roles
    roles = detect_column_roles(df)
    non_empty_roles = {k: v for k, v in roles.items() if v}
    if non_empty_roles:
        print(f"\nColumn roles:")
        for role, cols in non_empty_roles.items():
            print(f"  {role}: {cols}")
    
    # Summary
    summary = summarize_dataset(df)
    
    if summary.numeric_columns:
        print(f"\nNumeric columns: {summary.numeric_columns}")
        
        # Stats for each numeric column
        print(f"\nStatistics:")
        for col, stats in summary.numeric_summaries.items():
            print(f"  {col}: min={stats.min}, max={stats.max}, mean={stats.mean:.2f}, sum={stats.sum:.2f}")
    
    if summary.insights:
        print(f"\nInsights:")
        for insight in summary.insights:
            print(f"  {insight}")
    
    # Correlations
    correlations = find_strong_correlations(df, threshold=0.5)
    if correlations:
        print(f"\nStrong correlations:")
        for c in correlations[:5]:
            print(f"  {c.var1} <-> {c.var2}: {c.correlation:.2f} ({c.strength})")


def cmd_summary(args):
    """Quick summary stats."""
    from data_upload_engine import load_file
    from analysis_engine import prepare_for_analysis, quick_stats
    
    result = load_file(args.file)
    
    if not result.success:
        print(f"Error: {result.error}")
        return 1
    
    if result.is_document and result.document_content and result.document_content.tables:
        df = prepare_for_analysis(result.document_content.tables[0])
    else:
        df = prepare_for_analysis(result.df)
    
    print(f"\n{'='*40}")
    print(f"Quick Stats: {args.file}")
    print('='*40)
    
    stats = quick_stats(df)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")
    
    return 0


def cmd_correlations(args):
    """Find correlations in data."""
    from data_upload_engine import load_file
    from analysis_engine import prepare_for_analysis, correlation_matrix, find_strong_correlations
    
    result = load_file(args.file)
    
    if not result.success:
        print(f"Error: {result.error}")
        return 1
    
    if result.is_document and result.document_content and result.document_content.tables:
        df = prepare_for_analysis(result.document_content.tables[0])
    else:
        df = prepare_for_analysis(result.df)
    
    print(f"\n{'='*40}")
    print(f"Correlations: {args.file}")
    print('='*40)
    
    # Matrix
    corr = correlation_matrix(df)
    if not corr.empty:
        print(f"\nCorrelation matrix:")
        print(corr.round(2).to_string())
    
    # Strong correlations
    strong = find_strong_correlations(df, threshold=args.threshold)
    if strong:
        print(f"\nStrong correlations (threshold={args.threshold}):")
        for c in strong:
            print(f"  {c.var1} <-> {c.var2}: {c.correlation:.3f} ({c.strength} {c.direction})")
    else:
        print(f"\nNo correlations above threshold {args.threshold}")
    
    return 0


def cmd_tables(args):
    """List tables from PDF/PPTX."""
    from data_upload_engine import load_file
    
    result = load_file(args.file)
    
    if not result.success:
        print(f"Error: {result.error}")
        return 1
    
    if not result.is_document:
        print(f"Not a document file. Use 'load' command for CSV/Excel/JSON.")
        return 1
    
    tables = result.document_content.tables if result.document_content else []
    
    print(f"\n{'='*40}")
    print(f"Tables in: {args.file}")
    print('='*40)
    print(f"\nFound {len(tables)} tables")
    
    for i, table in enumerate(tables):
        page = table.attrs.get('source_page', table.attrs.get('source_slide', '?'))
        print(f"\n--- Table {i+1} (page/slide {page}) ---")
        print(f"Shape: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"Columns: {list(table.columns)}")
        
        if args.show_data:
            print(f"\nData:")
            print(table.to_string())
    
    return 0


def cmd_kpis(args):
    """Extract KPIs from PDF/PPTX."""
    from data_upload_engine import load_file
    
    result = load_file(args.file)
    
    if not result.success:
        print(f"Error: {result.error}")
        return 1
    
    metrics = result.metadata.get('extracted_metrics', {})
    
    print(f"\n{'='*40}")
    print(f"KPIs in: {args.file}")
    print('='*40)
    
    if metrics.get('kpis'):
        print(f"\nKPIs ({len(metrics['kpis'])}):")
        for kpi in metrics['kpis']:
            print(f"  {kpi['label']}: {kpi['full_value']}")
    
    if metrics.get('currencies'):
        print(f"\nCurrencies ({len(metrics['currencies'])}):")
        for c in metrics['currencies'][:10]:
            label = c.get('label', '?')
            print(f"  {label}: {c['value']}")
    
    if metrics.get('percentages'):
        print(f"\nPercentages ({len(metrics['percentages'])}):")
        for p in metrics['percentages'][:10]:
            label = p.get('label', '?')
            print(f"  {label}: {p['value']}")
    
    if metrics.get('ratios'):
        print(f"\nRatios:")
        for r in metrics['ratios']:
            print(f"  {r['label']}: {r['value']}")
    
    if metrics.get('projections'):
        print(f"\nProjections:")
        for p in metrics['projections']:
            print(f"  {p['type']}: {p['value']}")
    
    if not any(metrics.values()):
        print("\nNo KPIs extracted")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Local Analyst CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # load command
    p_load = subparsers.add_parser('load', help='Load and profile a file')
    p_load.add_argument('file', help='File to load')
    
    # analyze command
    p_analyze = subparsers.add_parser('analyze', help='Run full analysis')
    p_analyze.add_argument('file', help='File to analyze')
    
    # summary command
    p_summary = subparsers.add_parser('summary', help='Quick summary stats')
    p_summary.add_argument('file', help='File to summarize')
    
    # correlations command
    p_corr = subparsers.add_parser('correlations', help='Find correlations')
    p_corr.add_argument('file', help='File to analyze')
    p_corr.add_argument('-t', '--threshold', type=float, default=0.5, help='Correlation threshold')
    
    # tables command
    p_tables = subparsers.add_parser('tables', help='List tables from PDF/PPTX')
    p_tables.add_argument('file', help='Document file')
    p_tables.add_argument('-d', '--show-data', action='store_true', help='Show table data')
    
    # kpis command
    p_kpis = subparsers.add_parser('kpis', help='Extract KPIs from PDF/PPTX')
    p_kpis.add_argument('file', help='Document file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run command
    commands = {
        'load': cmd_load,
        'analyze': cmd_analyze,
        'summary': cmd_summary,
        'correlations': cmd_correlations,
        'tables': cmd_tables,
        'kpis': cmd_kpis,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
