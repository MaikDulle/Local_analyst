"""
Unified data loader interface.
Routes to appropriate loader based on file type, handles caching, and provides profiling.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass

from .csv_loader import load_csv
from .excel_loader import load_excel, get_sheet_info
from .json_loader import load_json, detect_json_structure
from .pdf_loader import load_pdf, pdf_to_dataframe, get_pdf_summary, PDFContent
from .pptx_loader import load_pptx, pptx_to_dataframe, get_pptx_summary, PPTXContent
from .docx_loader import load_docx, docx_to_dataframe, get_docx_summary, DocxContent
from .validators import profile_dataframe, DataProfile, suggest_ecom_mapping
from .cache import get_cache


@dataclass
class LoadResult:
    """Result of loading a data file."""
    df: pd.DataFrame
    metadata: Dict[str, Any]
    profile: DataProfile
    ecom_mapping: Dict[str, Optional[str]]
    success: bool
    error: Optional[str] = None
    # For documents (PDF, PPTX, DOCX)
    document_content: Optional[Any] = None  # PDFContent, PPTXContent, or DocxContent
    is_document: bool = False


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.csv': 'csv',
    '.xlsx': 'excel',
    '.xls': 'excel',
    '.xlsm': 'excel',
    '.xlsb': 'excel',
    '.json': 'json',
    '.pdf': 'pdf',
    '.pptx': 'pptx',
    '.ppt': 'pptx',
    '.docx': 'docx',
    '.doc': 'docx',
    '.jpg': 'image',
    '.jpeg': 'image',
    '.png': 'image',
    '.bmp': 'image',
    '.tiff': 'image',
    '.tif': 'image',
    '.webp': 'image',
}


# Document types that don't produce tabular data directly
DOCUMENT_TYPES = {'pdf', 'pptx', 'docx', 'image'}


def get_supported_extensions() -> List[str]:
    """Return list of supported file extensions."""
    return list(SUPPORTED_EXTENSIONS.keys())


def is_supported_file(file_path: str) -> bool:
    """Check if file type is supported."""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def load_file(
    file_path: str,
    use_cache: bool = True,
    profile_data: bool = True,
    **kwargs
) -> LoadResult:
    """
    Load data file with automatic type detection.
    
    This is the main entry point for loading data.
    
    Args:
        file_path: Path to data file
        use_cache: Whether to use caching (default True)
        profile_data: Whether to generate data profile (default True)
        **kwargs: Additional arguments passed to specific loader
            - CSV: encoding, delimiter, decimal
            - Excel: sheet_name, header_row, skip_rows
            - JSON: data_key, flatten
    
    Returns:
        LoadResult with DataFrame, metadata, profile, and ecom mapping suggestions
    """
    path = Path(file_path)
    
    # Validate file exists
    if not path.exists():
        return LoadResult(
            df=pd.DataFrame(),
            metadata={},
            profile=None,
            ecom_mapping={},
            success=False,
            error=f"File not found: {file_path}"
        )
    
    # Check extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return LoadResult(
            df=pd.DataFrame(),
            metadata={},
            profile=None,
            ecom_mapping={},
            success=False,
            error=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )
    
    # Check cache first
    cache = get_cache()
    if use_cache:
        cached = cache.get_cached_data(file_path, **kwargs)
        if cached is not None:
            df, metadata = cached
            metadata['from_cache'] = True
            
            profile = profile_dataframe(df) if profile_data else None
            ecom_mapping = suggest_ecom_mapping(df) if profile_data else {}
            
            return LoadResult(
                df=df,
                metadata=metadata,
                profile=profile,
                ecom_mapping=ecom_mapping,
                success=True
            )
    
    # Route to appropriate loader
    file_type = SUPPORTED_EXTENSIONS[ext]
    
    try:
        if file_type == 'csv':
            df, metadata = load_csv(file_path, **kwargs)
            document_content = None
            is_document = False
        elif file_type == 'excel':
            df, metadata = load_excel(file_path, **kwargs)
            document_content = None
            is_document = False
        elif file_type == 'json':
            df, metadata = load_json(file_path, **kwargs)
            document_content = None
            is_document = False
        elif file_type == 'pdf':
            # PDF is a document type - extract tables if possible
            document_content, metadata = load_pdf(file_path, **kwargs)
            df, table_metadata = pdf_to_dataframe(file_path, **kwargs)
            metadata.update(table_metadata)
            is_document = True
        elif file_type == 'pptx':
            # PPTX is a document type - extract tables if possible
            document_content, metadata = load_pptx(file_path)
            df, table_metadata = pptx_to_dataframe(file_path, **kwargs)
            metadata.update(table_metadata)
            is_document = True
            # Also extract data from embedded images/charts
            try:
                from .image_loader import extract_all_image_data_from_pptx
                img_tables, img_summary = extract_all_image_data_from_pptx(file_path)
                if img_tables:
                    document_content.all_tables.extend(img_tables)
                    metadata['image_extraction'] = img_summary
            except Exception:
                pass  # OCR extraction is best-effort
        elif file_type == 'docx':
            # DOCX is a document type - extract tables if possible
            document_content, metadata = load_docx(file_path)
            df, table_metadata = docx_to_dataframe(file_path, **kwargs)
            metadata.update(table_metadata)
            is_document = True
        elif file_type == 'image':
            # Image file - OCR extraction
            from .image_loader import load_image, image_to_dataframe, ImageContent
            document_content_raw, metadata = load_image(file_path)
            df, table_metadata = image_to_dataframe(file_path)
            metadata.update(table_metadata)
            # Wrap in a lightweight object with .tables for document_content
            class _ImageDoc:
                def __init__(self, ic):
                    self.tables = ic.tables if ic.tables else ([df] if not df.empty else [])
                    self.raw_text = ic.raw_text
                    self.kpis = ic.kpis
                    self.numbers = ic.numbers
            document_content = _ImageDoc(document_content_raw)
            is_document = True
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        metadata['from_cache'] = False
        
        # Cache the result (only for tabular data)
        if use_cache and not is_document:
            cache.cache_data(file_path, df, metadata, **kwargs)
        
        # Generate profile (only if we have data)
        if not df.empty and profile_data:
            profile = profile_dataframe(df)
            ecom_mapping = suggest_ecom_mapping(df)
        else:
            profile = None
            ecom_mapping = {}
        
        return LoadResult(
            df=df,
            metadata=metadata,
            profile=profile,
            ecom_mapping=ecom_mapping,
            success=True,
            document_content=document_content if is_document else None,
            is_document=is_document
        )
        
    except Exception as e:
        return LoadResult(
            df=pd.DataFrame(),
            metadata={},
            profile=None,
            ecom_mapping={},
            success=False,
            error=str(e),
            document_content=None,
            is_document=False
        )


def preview_file(file_path: str, n_rows: int = 10) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Quick preview of a file without full loading.
    
    Args:
        file_path: Path to data file
        n_rows: Number of rows to preview
    
    Returns:
        Tuple of (preview DataFrame or None, status message)
    """
    path = Path(file_path)
    
    if not path.exists():
        return None, f"File not found: {file_path}"
    
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None, f"Unsupported file type: {ext}"
    
    file_type = SUPPORTED_EXTENSIONS[ext]
    
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path, nrows=n_rows)
        elif file_type == 'excel':
            df = pd.read_excel(file_path, nrows=n_rows)
        elif file_type == 'json':
            # For JSON, load full then truncate (structure detection needed)
            result = load_file(file_path, use_cache=True, profile_data=False)
            if result.success:
                df = result.df.head(n_rows)
            else:
                return None, result.error
        elif file_type == 'pdf':
            # For PDF, show summary instead of table preview
            summary = get_pdf_summary(file_path)
            info = f"PDF: {summary['pages']} pages, {summary['text_length']} chars, {len(summary['tables'])} tables found"
            if summary['tables']:
                df, _ = pdf_to_dataframe(file_path)
                df = df.head(n_rows)
                return df, info
            else:
                return None, info + "\n\nText preview:\n" + summary['text_preview'][:500]
        elif file_type == 'pptx':
            # For PPTX, show summary
            summary = get_pptx_summary(file_path)
            info = f"PPTX: {summary['slides']} slides, {len(summary['tables'])} tables found"
            if summary['tables']:
                df, _ = pptx_to_dataframe(file_path)
                df = df.head(n_rows)
                return df, info
            else:
                return None, info + "\n\nText preview:\n" + summary['text_preview'][:500]
        elif file_type == 'docx':
            # For DOCX, show summary
            summary = get_docx_summary(file_path)
            info = f"DOCX: {summary['paragraphs']} paragraphs, {len(summary.get('tables', []))} tables found"
            if summary.get('has_tables', False):
                df, _ = docx_to_dataframe(file_path)
                df = df.head(n_rows)
                return df, info
            else:
                return None, info + "\n\nText preview:\n" + summary.get('text_preview', '')[:500]
        elif file_type == 'image':
            from .image_loader import image_to_dataframe
            df, meta = image_to_dataframe(file_path)
            extraction = meta.get('extraction_type', 'unknown')
            info = f"Image: {meta.get('dimensions', '?')} — extraction: {extraction}"
            if not df.empty:
                return df.head(n_rows), info
            else:
                return None, info + " (no data could be extracted)"
        
        return df, f"Preview: {len(df)} rows, {len(df.columns)} columns"
        
    except Exception as e:
        return None, f"Error previewing file: {e}"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file without loading it.
    
    Returns:
        Dict with file info (size, type, sheets for Excel, structure for JSON, etc.)
    """
    path = Path(file_path)
    
    if not path.exists():
        return {'error': 'File not found'}
    
    ext = path.suffix.lower()
    info = {
        'file_name': path.name,
        'file_size_kb': round(path.stat().st_size / 1024, 2),
        'file_size_mb': round(path.stat().st_size / (1024 * 1024), 2),
        'extension': ext,
        'supported': ext in SUPPORTED_EXTENSIONS
    }
    
    if ext not in SUPPORTED_EXTENSIONS:
        return info
    
    file_type = SUPPORTED_EXTENSIONS[ext]
    info['file_type'] = file_type
    info['is_document'] = file_type in DOCUMENT_TYPES
    
    try:
        if file_type == 'excel':
            sheets = get_sheet_info(file_path)
            info['sheets'] = sheets
            info['sheet_count'] = len(sheets)
            
        elif file_type == 'json':
            structure = detect_json_structure(file_path)
            info['json_structure'] = structure
            
        elif file_type == 'pdf':
            summary = get_pdf_summary(file_path)
            info['pages'] = summary['pages']
            info['tables_found'] = len(summary['tables'])
            info['text_length'] = summary['text_length']
            info['is_likely_scanned'] = summary['is_likely_scanned']
            info['tables'] = summary['tables']
            
        elif file_type == 'pptx':
            summary = get_pptx_summary(file_path)
            info['slides'] = summary['slides']
            info['tables_found'] = len(summary['tables'])
            info['text_length'] = summary['text_length']
            info['has_speaker_notes'] = summary['has_speaker_notes']
            info['tables'] = summary['tables']
            
        elif file_type == 'docx':
            summary = get_docx_summary(file_path)
            info['paragraphs'] = summary['paragraphs']
            info['tables_found'] = summary['tables']
            info['text_length'] = summary['total_text_length']
            info['has_tables'] = summary['has_tables']
            info['tables'] = summary.get('table_shapes', [])

        elif file_type == 'image':
            from .image_loader import load_image
            content, meta = load_image(file_path)
            info['dimensions'] = meta.get('dimensions', '')
            info['text_extracted'] = meta.get('text_extracted', 0)
            info['tables_found'] = meta.get('tables_found', 0)
            info['kpis_found'] = meta.get('kpis_found', 0)
            info['numbers_found'] = meta.get('numbers_found', 0)
            
    except Exception as e:
        info['info_error'] = str(e)
    
    return info


def load_multiple_files(
    file_paths: List[str],
    combine: bool = False,
    **kwargs
) -> Union[Dict[str, LoadResult], LoadResult]:
    """
    Load multiple files.
    
    Args:
        file_paths: List of file paths to load
        combine: If True, concatenate all DataFrames into one
        **kwargs: Arguments passed to load_file
    
    Returns:
        If combine=False: Dict mapping file paths to LoadResults
        If combine=True: Single LoadResult with combined DataFrame
    """
    results = {}
    
    for file_path in file_paths:
        results[file_path] = load_file(file_path, **kwargs)
    
    if not combine:
        return results
    
    # Combine successful loads
    dfs = []
    all_metadata = []
    
    for path, result in results.items():
        if result.success:
            df = result.df.copy()
            df['_source_file'] = Path(path).name
            dfs.append(df)
            all_metadata.append(result.metadata)
    
    if not dfs:
        return LoadResult(
            df=pd.DataFrame(),
            metadata={},
            profile=None,
            ecom_mapping={},
            success=False,
            error="No files loaded successfully"
        )
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    combined_metadata = {
        'files_loaded': len(dfs),
        'total_rows': len(combined_df),
        'source_files': [m.get('source_file') for m in all_metadata]
    }
    
    return LoadResult(
        df=combined_df,
        metadata=combined_metadata,
        profile=profile_dataframe(combined_df),
        ecom_mapping=suggest_ecom_mapping(combined_df),
        success=True
    )
