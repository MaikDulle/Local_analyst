"""
JSON file loader with automatic structure detection.
Handles nested JSON, JSON arrays, and JSON Lines format.
"""

import pandas as pd
import json
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path


def detect_json_structure(file_path: str) -> str:
    """
    Detect the structure of a JSON file.
    
    Returns:
        'array' - JSON array of objects [{"a": 1}, {"a": 2}]
        'object_with_data' - Object with a data key {"data": [...], "meta": ...}
        'nested' - Deeply nested structure
        'jsonlines' - JSON Lines format (one object per line)
        'single_object' - Single flat object
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # First, try to parse as standard JSON
    try:
        data = json.loads(content)
        
        if isinstance(data, list):
            return 'array'
        
        if isinstance(data, dict):
            # Check for common data wrapper keys
            data_keys = ['data', 'results', 'items', 'records', 'rows', 'entries']
            for key in data_keys:
                if key in data and isinstance(data[key], list):
                    return 'object_with_data'
            
            # Check nesting depth
            if _get_max_depth(data) > 2:
                return 'nested'
            
            return 'single_object'
        
        return 'single_object'
        
    except json.JSONDecodeError:
        # Not valid JSON as a whole, check for JSON Lines format
        lines = content.split('\n')
        valid_json_lines = 0
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                valid_json_lines += 1
            except json.JSONDecodeError:
                pass
        
        # If most non-empty lines are valid JSON, it's JSON Lines
        non_empty_lines = sum(1 for l in lines[:10] if l.strip())
        if non_empty_lines > 0 and valid_json_lines / non_empty_lines > 0.8:
            return 'jsonlines'
        
        raise ValueError(f"Invalid JSON file: could not parse as JSON or JSON Lines")


def _get_max_depth(obj: Any, current_depth: int = 0) -> int:
    """Get maximum nesting depth of a JSON structure."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(_get_max_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_get_max_depth(item, current_depth + 1) for item in obj[:10])  # Sample first 10
    return current_depth


def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (str, int, float, bool)):
            # Keep simple lists as-is (will become a column with list values)
            items.append((new_key, v))
        elif isinstance(v, list):
            # Skip complex nested lists for now
            items.append((new_key, str(v)[:100] + '...' if len(str(v)) > 100 else str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def load_json(
    file_path: str,
    data_key: Optional[str] = None,
    flatten: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load JSON file into DataFrame with automatic structure handling.
    
    Args:
        file_path: Path to JSON file
        data_key: Key containing the data array (auto-detected if None)
        flatten: Whether to flatten nested structures
        **kwargs: Additional arguments passed to pd.json_normalize
    
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() != '.json':
        raise ValueError(f"Expected JSON file, got: {path.suffix}")
    
    structure = detect_json_structure(file_path)
    
    # Load based on detected structure
    if structure == 'jsonlines':
        df = pd.read_json(file_path, lines=True, **kwargs)
        
    elif structure == 'array':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if flatten:
            data = [_flatten_dict(item) if isinstance(item, dict) else item for item in data]
        df = pd.DataFrame(data)
        
    elif structure == 'object_with_data':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find the data key
        if data_key and data_key in data:
            records = data[data_key]
        else:
            # Auto-detect
            data_keys = ['data', 'results', 'items', 'records', 'rows', 'entries']
            records = None
            for key in data_keys:
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    data_key = key
                    break
            
            if records is None:
                # Use first list found
                for key, value in data.items():
                    if isinstance(value, list):
                        records = value
                        data_key = key
                        break
        
        if records is None:
            raise ValueError("Could not find data array in JSON object")
        
        if flatten:
            records = [_flatten_dict(item) if isinstance(item, dict) else item for item in records]
        df = pd.DataFrame(records)
        
    elif structure == 'nested':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Try json_normalize for nested structures
        if isinstance(data, list):
            df = pd.json_normalize(data, **kwargs)
        elif isinstance(data, dict):
            # Try to find array to normalize
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    df = pd.json_normalize(value, **kwargs)
                    data_key = key
                    break
            else:
                # Flatten single object
                flat = _flatten_dict(data)
                df = pd.DataFrame([flat])
        else:
            df = pd.DataFrame([data])
            
    else:  # single_object
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if flatten and isinstance(data, dict):
            data = _flatten_dict(data)
        df = pd.DataFrame([data])
    
    # Clean column names
    df.columns = df.columns.astype(str).str.strip()
    
    # Try to parse date columns
    df = _try_parse_dates(df)
    
    metadata = {
        'source_file': str(path.name),
        'file_size_kb': round(path.stat().st_size / 1024, 2),
        'json_structure': structure,
        'data_key': data_key,
        'flattened': flatten,
        'rows_loaded': len(df),
        'columns_loaded': len(df.columns),
    }
    
    return df, metadata


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to parse columns that look like dates."""
    date_keywords = ['date', 'datum', 'time', 'zeit', 'created', 'updated', 'timestamp',
                     '_at', '_on', 'day', 'month', 'year']
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        if not any(kw in col_lower for kw in date_keywords):
            continue
        
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() / len(df) > 0.8:
                df[col] = parsed
        except:
            pass
    
    return df
