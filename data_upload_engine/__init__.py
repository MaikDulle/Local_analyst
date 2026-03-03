"""
Data Upload Engine - Robust file loading with automatic type detection and profiling.

Usage:
    from data_upload_engine import load_file, preview_file, get_file_info
    
    # Load a file (CSV, Excel, JSON, PDF, or PPTX)
    result = load_file("data.csv")
    
    if result.success:
        df = result.df                    # The loaded DataFrame
        profile = result.profile          # Data quality profile
        mapping = result.ecom_mapping     # Suggested column mappings for ecom
        
        # For documents (PDF/PPTX)
        if result.is_document:
            content = result.document_content  # Full text and structure
    else:
        print(result.error)
    
    # Quick preview without full load
    preview_df, status = preview_file("data.xlsx", n_rows=5)
    
    # Get file info without loading
    info = get_file_info("report.pdf")
"""

from .loader import (
    load_file,
    preview_file,
    get_file_info,
    load_multiple_files,
    get_supported_extensions,
    is_supported_file,
    LoadResult,
    DOCUMENT_TYPES,
)

from .validators import (
    profile_dataframe,
    profile_column,
    validate_for_analysis,
    suggest_ecom_mapping,
    infer_column_type,
    ColumnType,
    ColumnProfile,
    DataProfile,
)

from .cache import (
    DataCache,
    get_cache,
)

# Expose individual loaders for advanced use
from .csv_loader import load_csv
from .excel_loader import load_excel, get_sheet_info
from .json_loader import load_json, detect_json_structure
from .pdf_loader import (
    load_pdf, 
    pdf_to_dataframe, 
    get_pdf_summary,
    extract_metrics_from_text,
    PDFContent,
)
from .pptx_loader import (
    load_pptx,
    pptx_to_dataframe,
    get_pptx_summary,
    extract_metrics_from_pptx,
    PPTXContent,
    SlideContent,
)

from .image_loader import (
    load_image,
    image_to_dataframe,
    extract_images_from_pptx,
    extract_all_image_data_from_pptx,
    ImageContent,
)


__all__ = [
    # Main interface
    'load_file',
    'preview_file', 
    'get_file_info',
    'load_multiple_files',
    'get_supported_extensions',
    'is_supported_file',
    'LoadResult',
    'DOCUMENT_TYPES',
    
    # Validation & profiling
    'profile_dataframe',
    'profile_column',
    'validate_for_analysis',
    'suggest_ecom_mapping',
    'infer_column_type',
    'ColumnType',
    'ColumnProfile', 
    'DataProfile',
    
    # Caching
    'DataCache',
    'get_cache',
    
    # Individual loaders
    'load_csv',
    'load_excel',
    'get_sheet_info',
    'load_json',
    'detect_json_structure',
    
    # PDF loader
    'load_pdf',
    'pdf_to_dataframe',
    'get_pdf_summary',
    'extract_metrics_from_text',
    'PDFContent',
    
    # PPTX loader
    'load_pptx',
    'pptx_to_dataframe',
    'get_pptx_summary',
    'extract_metrics_from_pptx',
    'PPTXContent',
    'SlideContent',
]
