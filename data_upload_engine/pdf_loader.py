"""
PDF file loader with text and table extraction.
Handles text-based PDFs and attempts table extraction.

Dependencies:
    pip install pdfplumber PyMuPDF
    
For scanned PDFs (OCR), additionally:
    pip install pytesseract pdf2image
    (requires tesseract-ocr system package)
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import re


@dataclass
class PDFContent:
    """Structured content extracted from a PDF."""
    text: str
    tables: List[pd.DataFrame]
    pages: int
    metadata: Dict[str, Any]
    extraction_method: str
    warnings: List[str]


def extract_with_pdfplumber(file_path: str) -> PDFContent:
    """
    Extract content using pdfplumber (best for tables).
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber required: pip install pdfplumber")
    
    text_parts = []
    tables = []
    warnings = []
    
    with pdfplumber.open(file_path) as pdf:
        metadata = pdf.metadata or {}
        pages = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")
            
            # Extract tables
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 1:
                    try:
                        # First row as header
                        df = pd.DataFrame(table[1:], columns=table[0])
                        # Clean up
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        if not df.empty and len(df.columns) > 0:
                            # Quality filter: skip junk tables from chart axes
                            named_cols = sum(1 for c in df.columns if str(c).strip() and str(c) != 'nan')
                            non_empty_cells = df.astype(str).apply(lambda s: s.str.strip().ne('')).sum().sum()
                            total_cells = df.shape[0] * df.shape[1]
                            fill_rate = non_empty_cells / total_cells if total_cells > 0 else 0
                            if len(df) >= 1 and named_cols >= 1 and fill_rate > 0.3:
                                df.attrs['source_page'] = i + 1
                                tables.append(df)
                    except Exception as e:
                        warnings.append(f"Table extraction failed on page {i+1}: {e}")
    
    return PDFContent(
        text="\n\n".join(text_parts),
        tables=tables,
        pages=pages,
        metadata=dict(metadata),
        extraction_method="pdfplumber",
        warnings=warnings
    )


def extract_with_pymupdf(file_path: str) -> PDFContent:
    """
    Extract content using PyMuPDF/fitz (faster, good for text).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF required: pip install PyMuPDF")
    
    text_parts = []
    warnings = []
    
    doc = fitz.open(file_path)
    metadata = doc.metadata or {}
    pages = len(doc)
    
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_parts.append(f"--- Page {i+1} ---\n{text}")
    
    doc.close()
    
    # PyMuPDF doesn't extract tables well, so we return empty list
    # and suggest using pdfplumber if tables are needed
    
    return PDFContent(
        text="\n\n".join(text_parts),
        tables=[],  # PyMuPDF doesn't do tables
        pages=pages,
        metadata=dict(metadata),
        extraction_method="pymupdf",
        warnings=warnings + ["PyMuPDF used - table extraction not available. Install pdfplumber for tables."]
    )


def extract_text_fallback(file_path: str) -> PDFContent:
    """
    Basic fallback using PyPDF2 if other libraries unavailable.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("No PDF library available. Install: pip install pdfplumber PyMuPDF")
    
    reader = PdfReader(file_path)
    text_parts = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_parts.append(f"--- Page {i+1} ---\n{text}")
    
    metadata = dict(reader.metadata) if reader.metadata else {}
    
    return PDFContent(
        text="\n\n".join(text_parts),
        tables=[],
        pages=len(reader.pages),
        metadata=metadata,
        extraction_method="pypdf_fallback",
        warnings=["Basic extraction only - install pdfplumber for better results"]
    )


def detect_if_scanned(content: PDFContent) -> bool:
    """
    Detect if PDF is likely scanned (image-based) with little extractable text.
    """
    if not content.text.strip():
        return True
    
    # Very low text-to-page ratio suggests scanned
    text_length = len(content.text.replace(" ", "").replace("\n", ""))
    chars_per_page = text_length / max(content.pages, 1)
    
    # Typical text PDF has 1500+ chars per page, scanned might have <100
    return chars_per_page < 100


def extract_metrics_from_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract common business metrics from PDF text using regex patterns.
    Returns dict of metric types with found values.
    """
    metrics = {
        'currencies': [],
        'percentages': [],
        'large_numbers': [],
        'kpis': [],
        'ratios': [],
        'projections': []
    }
    
    # Currency patterns (EUR, USD, €, $) - improved to capture full context
    currency_patterns = [
        r'(€|EUR)\s*([\d.,]+(?:\s*(?:Mio|Mrd|K|M|B|million|billion))?)',
        r'(\$)\s*([\d,]+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:K|M|B|million|billion))?)',
        r'([\d.,]+)\s*(€|EUR|USD|\$)',
    ]
    
    for pattern in currency_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get surrounding context to find the label
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 20)
            context = text[start:end].replace('\n', ' ')
            
            # Try to find label before the value
            label_match = re.search(r'([A-Za-z][A-Za-z\s\(\)]+?)[\s:]+$', context[:match.start()-start])
            label = label_match.group(1).strip() if label_match else None
            
            metrics['currencies'].append({
                'value': match.group(0).strip(),
                'label': label,
                'context': context.strip()
            })
    
    # Percentage patterns - with context
    pct_pattern = r'([\d.,]+)\s*(%|Prozent|percent)'
    for match in re.finditer(pct_pattern, text, re.IGNORECASE):
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 20)
        context = text[start:end].replace('\n', ' ')
        
        label_match = re.search(r'([A-Za-z][A-Za-z\s\(\)]+?)[\s:]+$', context[:match.start()-start])
        label = label_match.group(1).strip() if label_match else None
        
        metrics['percentages'].append({
            'value': match.group(0).strip(),
            'numeric': match.group(1),
            'label': label,
            'context': context.strip()
        })
    
    # KPI patterns: "Label: Value" or "Label: $Value" or "Label: Value%"
    # This catches things like "MRR Growth Rate: 8.5%" or "ARR: $10,200,000"
    kpi_patterns = [
        # Label: $currency
        r'([A-Za-z][A-Za-z\s\(\)\-/]+?):\s*\$?([\d,]+(?:,\d{3})*(?:\.\d+)?)\s*(%|million|billion|M|B|K)?',
        # Label: percentage
        r'([A-Za-z][A-Za-z\s\(\)\-/]+?):\s*([\d.,]+)\s*(%)',
        # Label: ratio (e.g., "2.8:1")
        r'([A-Za-z][A-Za-z\s\(\)\-/]+?):\s*([\d.]+:\d+)',
        # Label: days/months
        r'([A-Za-z][A-Za-z\s\(\)\-/]+?):\s*([\d.]+)\s*(days?|months?|years?)',
    ]
    
    for pattern in kpi_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            label = match.group(1).strip()
            value = match.group(2).strip()
            unit = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else None
            
            # Skip if label is too short or too long (noise)
            if len(label) < 3 or len(label) > 50:
                continue
            
            # Skip table headers or noise
            if label.lower() in ['region', 'product', 'channel', 'department', 'metric', 'segment']:
                continue
                
            full_value = f"{value}{unit}" if unit else value
            
            metrics['kpis'].append({
                'label': label,
                'value': value,
                'unit': unit,
                'full_value': full_value
            })
    
    # Ratio patterns (e.g., "7.55:1", "2.8:1")
    ratio_pattern = r'([A-Za-z][A-Za-z\s]+?)(?:ratio)?[\s:]+(\d+\.?\d*:\d+)'
    for match in re.finditer(ratio_pattern, text, re.IGNORECASE):
        metrics['ratios'].append({
            'label': match.group(1).strip(),
            'value': match.group(2)
        })
    
    # Projection patterns (forward-looking statements)
    projection_patterns = [
        r'(projected|expected|target|forecast)[^\n.]*?([\$€][\d,]+(?:\.\d+)?(?:\s*(?:M|B|million|billion))?)',
        r'(Q[1-4]\s*20\d{2})[^\n.]*?([\$€][\d,]+(?:\.\d+)?(?:\s*(?:M|B|million|billion))?)',
    ]
    
    for pattern in projection_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 50)
            context = text[start:end].replace('\n', ' ')
            
            metrics['projections'].append({
                'type': match.group(1),
                'value': match.group(2),
                'context': context.strip()
            })
    
    # Deduplicate KPIs
    seen_kpis = set()
    unique_kpis = []
    for kpi in metrics['kpis']:
        key = f"{kpi['label']}:{kpi['value']}"
        if key not in seen_kpis:
            seen_kpis.add(key)
            unique_kpis.append(kpi)
    metrics['kpis'] = unique_kpis
    
    return metrics


def tables_to_dataframe(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Combine multiple extracted tables into a single DataFrame if they have compatible structure.
    Returns None if tables are too different to combine.
    """
    if not tables:
        return None
    
    if len(tables) == 1:
        return tables[0]
    
    # Check if tables have same columns
    first_cols = set(tables[0].columns)
    compatible = all(set(t.columns) == first_cols for t in tables)
    
    if compatible:
        combined = pd.concat(tables, ignore_index=True)
        return combined
    
    # Return largest table if not compatible
    return max(tables, key=len)


def _ocr_pdf_chart_pages(file_path: str, content: PDFContent) -> List[pd.DataFrame]:
    """
    Render PDF pages as images and OCR them to extract data from charts/plots.
    
    Tries: PyMuPDF (fitz) → pdf2image (poppler) → skip.
    Only processes pages that had no table extracted by pdfplumber.
    """
    # Find which pages already have tables
    pages_with_tables = set()
    for t in content.tables:
        pg = getattr(t, 'attrs', {}).get('source_page')
        if pg:
            pages_with_tables.add(pg)

    pages_to_ocr = [p for p in range(1, content.pages + 1) if p not in pages_with_tables]
    if not pages_to_ocr:
        return []

    # Try to import image_loader OCR functions
    try:
        from .image_loader import _run_ocr, _preprocess_image, _parse_table_from_text, \
            _parse_numbers_from_text, _extract_kpis, _detect_ocr_backend
    except ImportError:
        return []

    if _detect_ocr_backend() is None:
        return []

    import numpy as np

    # Render pages to images
    page_images = {}

    # Method 1: PyMuPDF
    try:
        import fitz
        doc = fitz.open(file_path)
        for pg in pages_to_ocr:
            pix = doc[pg - 1].get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:  # RGBA
                img = img[:, :, :3]
            page_images[pg] = img
        doc.close()
    except (ImportError, Exception):
        pass

    # Method 2: pdf2image
    if not page_images:
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path, dpi=200, first_page=min(pages_to_ocr),
                                       last_page=max(pages_to_ocr))
            # Map back to page numbers
            all_pages_range = list(range(min(pages_to_ocr), max(pages_to_ocr) + 1))
            for pil_img, pg in zip(images, all_pages_range):
                if pg in pages_to_ocr:
                    page_images[pg] = np.array(pil_img)
        except (ImportError, Exception):
            pass

    if not page_images:
        return []

    # OCR each rendered page
    extracted_tables = []
    for pg, img_array in page_images.items():
        try:
            preprocessed = _preprocess_image(img_array)
            text, _ = _run_ocr(preprocessed)
            if not text or len(text.strip()) < 20:
                continue

            # Try to get tables from OCR text
            tables = _parse_table_from_text(text)
            for t in tables:
                t.attrs = {'source_page': pg, 'extraction': 'ocr'}
                extracted_tables.append(t)

            # If no tables, try KPIs
            if not tables:
                kpis = _extract_kpis(text)
                if len(kpis) >= 2:
                    kpi_df = pd.DataFrame(kpis)
                    kpi_df.attrs = {'source_page': pg, 'extraction': 'ocr_kpi'}
                    extracted_tables.append(kpi_df)
        except Exception:
            continue

    return extracted_tables


def load_pdf(
    file_path: str,
    extract_tables: bool = True,
    extract_metrics: bool = True,
    prefer_method: str = "auto"
) -> Tuple[PDFContent, Dict[str, Any]]:
    """
    Load and extract content from a PDF file.
    
    Args:
        file_path: Path to PDF file
        extract_tables: Whether to attempt table extraction
        extract_metrics: Whether to extract business metrics from text
        prefer_method: 'pdfplumber', 'pymupdf', or 'auto'
    
    Returns:
        Tuple of (PDFContent, metadata dict)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"Expected PDF file, got: {path.suffix}")
    
    # Choose extraction method
    content = None
    
    if prefer_method == "auto" or prefer_method == "pdfplumber":
        try:
            content = extract_with_pdfplumber(file_path)
        except ImportError:
            if prefer_method == "pdfplumber":
                raise
            # Fall through to try other methods
        except Exception as e:
            if prefer_method == "pdfplumber":
                raise
    
    if content is None and (prefer_method == "auto" or prefer_method == "pymupdf"):
        try:
            content = extract_with_pymupdf(file_path)
        except ImportError:
            if prefer_method == "pymupdf":
                raise
        except Exception as e:
            if prefer_method == "pymupdf":
                raise
    
    if content is None:
        content = extract_text_fallback(file_path)
    
    # Check if scanned
    is_scanned = detect_if_scanned(content)
    if is_scanned:
        content.warnings.append(
            "PDF appears to be scanned/image-based. Text extraction may be incomplete. "
            "Consider using OCR tools for better results."
        )
    
    # Extract metrics if requested
    extracted_metrics = {}
    if extract_metrics and content.text:
        extracted_metrics = extract_metrics_from_text(content.text)
    
    # OCR rendered pages to capture data from charts/plots
    try:
        ocr_tables = _ocr_pdf_chart_pages(file_path, content)
        if ocr_tables:
            content.tables.extend(ocr_tables)
            content.warnings.append(
                f"Extracted {len(ocr_tables)} additional table(s) from chart images via OCR."
            )
    except Exception:
        pass  # OCR is best-effort

    metadata = {
        'source_file': path.name,
        'file_size_kb': round(path.stat().st_size / 1024, 2),
        'pages': content.pages,
        'extraction_method': content.extraction_method,
        'tables_found': len(content.tables),
        'text_length': len(content.text),
        'is_likely_scanned': is_scanned,
        'pdf_metadata': content.metadata,
        'extracted_metrics': extracted_metrics,
        'warnings': content.warnings,
    }
    
    return content, metadata


def pdf_to_dataframe(
    file_path: str,
    table_index: Optional[int] = None,
    combine_tables: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load PDF and return extracted tables as DataFrame.
    
    This is the main entry point for treating PDFs as data sources.
    
    Args:
        file_path: Path to PDF file
        table_index: Specific table to return (0-indexed), or None for all/combined
        combine_tables: If True and multiple tables found, try to combine them
    
    Returns:
        Tuple of (DataFrame, metadata)
        DataFrame may be empty if no tables found
    """
    content, metadata = load_pdf(file_path, extract_tables=True)
    
    if not content.tables:
        # No tables found - return empty DataFrame with text summary
        metadata['note'] = "No tables found in PDF. Text content available in metadata."
        metadata['text_preview'] = content.text[:2000] if content.text else ""
        return pd.DataFrame(), metadata
    
    if table_index is not None:
        if table_index < len(content.tables):
            df = content.tables[table_index]
        else:
            raise IndexError(f"Table index {table_index} out of range. Found {len(content.tables)} tables.")
    elif combine_tables:
        df = tables_to_dataframe(content.tables)
        if df is None:
            df = content.tables[0]  # Fallback to first table
    else:
        df = content.tables[0]
    
    metadata['table_index'] = table_index
    metadata['rows_loaded'] = len(df)
    metadata['columns_loaded'] = len(df.columns)
    
    return df, metadata


def get_pdf_summary(file_path: str) -> Dict[str, Any]:
    """
    Quick summary of PDF contents without full extraction.
    Useful for previewing what's in a PDF.
    """
    content, metadata = load_pdf(file_path, extract_tables=True, extract_metrics=True)
    
    # Get text preview
    text_preview = content.text[:1000] + "..." if len(content.text) > 1000 else content.text
    
    # Table summaries
    table_summaries = []
    for i, table in enumerate(content.tables):
        table_summaries.append({
            'index': i,
            'rows': len(table),
            'columns': len(table.columns),
            'column_names': list(table.columns),
            'source_page': table.attrs.get('source_page', 'unknown')
        })
    
    return {
        'file_name': Path(file_path).name,
        'pages': content.pages,
        'text_preview': text_preview,
        'text_length': len(content.text),
        'tables': table_summaries,
        'metrics_found': metadata.get('extracted_metrics', {}),
        'is_likely_scanned': metadata.get('is_likely_scanned', False),
        'warnings': content.warnings
    }
