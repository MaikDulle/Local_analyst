"""
Excel file loader with sheet detection and selection.
Supports .xlsx, .xls, and .xlsm formats.
"""

import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path


def get_sheet_info(file_path: str) -> List[Dict[str, Any]]:
    """
    Get information about all sheets in an Excel file.
    
    Returns list of dicts with sheet name, row count estimate, and column count.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get sheet names
    xlsx = pd.ExcelFile(file_path)
    sheets_info = []
    
    for sheet_name in xlsx.sheet_names:
        # Read just first few rows to get column count
        preview = pd.read_excel(xlsx, sheet_name=sheet_name, nrows=5)
        
        # Estimate row count (read a bit more)
        sample = pd.read_excel(xlsx, sheet_name=sheet_name, nrows=1000)
        
        sheets_info.append({
            'name': sheet_name,
            'columns': len(preview.columns),
            'column_names': list(preview.columns),
            'estimated_rows': len(sample),
            'has_data': len(sample) > 0 and len(preview.columns) > 0
        })
    
    xlsx.close()
    return sheets_info


def load_excel(
    file_path: str,
    sheet_name: Optional[Union[str, int]] = None,
    header_row: int = 0,
    skip_rows: Optional[List[int]] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load Excel file with automatic sheet selection if not specified.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name or index (0-based). If None, uses first sheet with data.
        header_row: Row number to use as header (0-indexed)
        skip_rows: List of row numbers to skip
        **kwargs: Additional arguments passed to pd.read_excel
    
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    valid_extensions = ['.xlsx', '.xls', '.xlsm', '.xlsb']
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Expected Excel file, got: {path.suffix}")
    
    # Get sheet info
    sheets = get_sheet_info(file_path)
    available_sheets = [s['name'] for s in sheets]
    
    # Auto-select sheet if not specified
    selected_sheet = sheet_name
    if selected_sheet is None:
        # Find first sheet with actual data
        for sheet_info in sheets:
            if sheet_info['has_data']:
                selected_sheet = sheet_info['name']
                break
        
        if selected_sheet is None:
            selected_sheet = available_sheets[0] if available_sheets else 0
    
    # Load the sheet
    read_kwargs = {
        'sheet_name': selected_sheet,
        'header': header_row,
        **kwargs
    }
    
    if skip_rows:
        read_kwargs['skiprows'] = skip_rows
    
    df = pd.read_excel(file_path, **read_kwargs)
    
    # Clean column names
    df.columns = df.columns.astype(str).str.strip()
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # Try to parse date columns
    df = _try_parse_dates(df)
    
    # Get info about selected sheet
    selected_sheet_info = next((s for s in sheets if s['name'] == selected_sheet), None)
    
    metadata = {
        'source_file': str(path.name),
        'file_size_kb': round(path.stat().st_size / 1024, 2),
        'sheet_name': selected_sheet,
        'available_sheets': available_sheets,
        'header_row': header_row,
        'rows_loaded': len(df),
        'columns_loaded': len(df.columns),
    }
    
    return df, metadata


def load_all_sheets(file_path: str, **kwargs) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Load all sheets from an Excel file.
    
    Returns dict mapping sheet names to (DataFrame, metadata) tuples.
    """
    sheets = get_sheet_info(file_path)
    results = {}
    
    for sheet_info in sheets:
        if sheet_info['has_data']:
            df, meta = load_excel(file_path, sheet_name=sheet_info['name'], **kwargs)
            results[sheet_info['name']] = (df, meta)
    
    return results


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to parse columns that look like dates.
    """
    date_keywords = ['date', 'datum', 'time', 'zeit', 'created', 'updated', 'timestamp',
                     'day', 'month', 'year', 'tag', 'monat', 'jahr']
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Skip if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # Only try if column name suggests date
        if not any(kw in col_lower for kw in date_keywords):
            continue
        
        # Try to parse
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() / len(df) > 0.8:
                df[col] = parsed
        except:
            pass
    
    return df
