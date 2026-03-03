"""
Manual testing with real files.
Place test files in data/samples/ and run:
    python -m tests.test_manual

Or test specific files:
    python -m tests.test_manual "path/to/file.pdf"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_upload_engine import load_file, get_pdf_summary, get_pptx_summary


def test_file(file_path: str):
    """Test any supported file."""
    path = Path(file_path)
    print(f"\n{'='*60}")
    print(f"Testing: {path.name}")
    print('='*60)
    
    result = load_file(file_path)
    
    if not result.success:
        print(f"✗ Failed: {result.error}")
        return
    
    print(f"✓ Loaded successfully")
    print(f"  Is document: {result.is_document}")
    print(f"  DataFrame shape: {result.df.shape}")
    
    if result.profile:
        print(f"  Quality score: {result.profile.quality_score}")
    
    # Show tables
    if not result.df.empty:
        print(f"\n--- Main DataFrame ---")
        print(result.df.head(10))
    
    # For documents, show all tables
    if result.is_document and result.document_content:
        tables = result.document_content.tables
        print(f"\n--- All Tables ({len(tables)}) ---")
        for i, table in enumerate(tables):
            page = table.attrs.get('source_page', table.attrs.get('source_slide', '?'))
            print(f"\nTable {i+1} (page/slide {page}): {table.shape}")
            print(table)
    
    # Show extracted metrics for PDFs/PPTXs
    if result.metadata.get('extracted_metrics'):
        metrics = result.metadata['extracted_metrics']
        print(f"\n--- Extracted Metrics ---")
        
        if metrics.get('kpis'):
            print(f"\nKPIs ({len(metrics['kpis'])}):")
            for kpi in metrics['kpis'][:15]:
                print(f"  {kpi['label']}: {kpi['full_value']}")
        
        if metrics.get('currencies'):
            print(f"\nCurrencies ({len(metrics['currencies'])}):")
            for c in metrics['currencies'][:10]:
                label = c.get('label', 'unknown')
                print(f"  {label}: {c['value']}")
        
        if metrics.get('percentages'):
            print(f"\nPercentages ({len(metrics['percentages'])}):")
            for p in metrics['percentages'][:10]:
                label = p.get('label', 'unknown')
                print(f"  {label}: {p['value']}")
        
        if metrics.get('ratios'):
            print(f"\nRatios:")
            for r in metrics['ratios']:
                print(f"  {r['label']}: {r['value']}")
        
        if metrics.get('projections'):
            print(f"\nProjections:")
            for p in metrics['projections']:
                print(f"  {p['type']}: {p['value']}")
    
    # Show text preview for documents
    if result.is_document and result.document_content:
        text = result.document_content.text
        print(f"\n--- Text Preview ({len(text)} chars total) ---")
        print(text[:800] + "..." if len(text) > 800 else text)


def main():
    # Test files from command line args or data/samples/
    if len(sys.argv) > 1:
        for file_path in sys.argv[1:]:
            test_file(file_path)
    else:
        # Look for files in data/samples/
        samples_dir = Path(__file__).parent.parent / "data" / "samples"
        if samples_dir.exists():
            files = list(samples_dir.iterdir())
            supported = [f for f in files if f.suffix.lower() in ['.csv', '.xlsx', '.xls', '.json', '.pdf', '.pptx']]
            
            if supported:
                print(f"Found {len(supported)} files in data/samples/")
                for file in supported:
                    test_file(str(file))
            else:
                print("No supported files found in data/samples/")
                print("Supported: .csv, .xlsx, .xls, .json, .pdf, .pptx")
        else:
            print("Usage:")
            print("  python -m tests.test_manual <file1> <file2> ...")
            print("  or: place files in data/samples/ and run without args")
            print("\nCreate data/samples/ folder:")
            print(f"  mkdir -p {samples_dir}")


if __name__ == '__main__':
    main()
