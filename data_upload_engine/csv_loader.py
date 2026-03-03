"""
CSV file loader with automatic encoding and delimiter detection.
"""

import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


def detect_encoding(file_path: str) -> str:
    """
    Detect file encoding by trying common encodings.
    Returns the first encoding that works.
    """
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(10000)  # Read first 10KB
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Fallback
    return 'utf-8'


def detect_delimiter(file_path: str, encoding: str) -> str:
    """
    Detect CSV delimiter by analyzing first few lines.
    """
    delimiters = [',', ';', '\t', '|']
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            sample = f.read(5000)
    except:
        return ','
    
    # Count occurrences of each delimiter
    counts = {d: sample.count(d) for d in delimiters}
    
    # Pick delimiter with highest count (but must appear consistently)
    lines = sample.split('\n')[:10]
    
    for delimiter in delimiters:
        if counts[delimiter] == 0:
            continue
        
        # Check if delimiter count is consistent across lines
        line_counts = [line.count(delimiter) for line in lines if line.strip()]
        if len(set(line_counts)) == 1 and line_counts[0] > 0:
            return delimiter
    
    # Fallback: return delimiter with most occurrences
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ','


def detect_decimal_separator(file_path: str, encoding: str, delimiter: str) -> str:
    """
    Detect decimal separator (comma vs dot) for German vs English number formats.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            sample = f.read(10000)
    except:
        return '.'
    
    # If delimiter is semicolon, decimal is likely comma (German format)
    if delimiter == ';':
        # Look for patterns like "1,5" or "100,00"
        import re
        comma_decimals = len(re.findall(r'\d+,\d{1,2}(?!\d)', sample))
        dot_decimals = len(re.findall(r'\d+\.\d{1,2}(?!\d)', sample))
        
        if comma_decimals > dot_decimals:
            return ','
    
    return '.'


def load_csv(
    file_path: str,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    decimal: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load CSV file with automatic detection of encoding, delimiter, and decimal separator.
    
    Args:
        file_path: Path to CSV file
        encoding: Override encoding detection
        delimiter: Override delimiter detection
        decimal: Override decimal separator detection
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not path.suffix.lower() == '.csv':
        raise ValueError(f"Expected CSV file, got: {path.suffix}")
    
    # Auto-detect settings if not provided
    detected_encoding = encoding or detect_encoding(file_path)
    detected_delimiter = delimiter or detect_delimiter(file_path, detected_encoding)
    detected_decimal = decimal or detect_decimal_separator(file_path, detected_encoding, detected_delimiter)
    
    # Handle German thousands separator if decimal is comma
    thousands = '.' if detected_decimal == ',' else ','
    
    # Load the CSV
    try:
        df = pd.read_csv(
            file_path,
            encoding=detected_encoding,
            delimiter=detected_delimiter,
            decimal=detected_decimal,
            thousands=thousands if detected_decimal == ',' else None,
            **kwargs
        )
    except Exception as e:
        # Retry with more permissive settings
        df = pd.read_csv(
            file_path,
            encoding=detected_encoding,
            delimiter=detected_delimiter,
            on_bad_lines='skip',
            **kwargs
        )
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Try to parse date columns
    df = _try_parse_dates(df)
    
    metadata = {
        'source_file': str(path.name),
        'file_size_kb': round(path.stat().st_size / 1024, 2),
        'encoding': detected_encoding,
        'delimiter': detected_delimiter,
        'decimal_separator': detected_decimal,
        'rows_loaded': len(df),
        'columns_loaded': len(df.columns),
    }
    
    return df, metadata


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to parse columns that look like dates.
    """
    date_keywords = ['date', 'datum', 'time', 'zeit', 'created', 'updated', 'timestamp', 
                     'day', 'month', 'year', 'tag', 'monat', 'jahr']
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Skip if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # Only try if column name suggests date
        if not any(kw in col_lower for kw in date_keywords):
            continue
        
        # Try to parse
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            # Only convert if most values parsed successfully
            if parsed.notna().sum() / len(df) > 0.8:
                df[col] = parsed
        except:
            pass
    
    return df
