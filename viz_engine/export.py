"""
Export functionality for Local Analyst visualizations.
Save charts as images, PDFs, and interactive HTML.
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional
import io


def export_plotly_chart(
    fig: go.Figure,
    filename: str,
    format: str = 'html',
    width: Optional[int] = None,
    height: Optional[int] = None
) -> str:
    """
    Export Plotly chart to file.
    
    Args:
        fig: Plotly Figure object
        filename: Output filename (without extension)
        format: Output format ('html', 'png', 'pdf', 'svg', 'jpeg')
        width: Width in pixels (for image formats)
        height: Height in pixels (for image formats)
        
    Returns:
        Path to saved file
    """
    filepath = Path(filename).with_suffix(f'.{format}')
    
    if format == 'html':
        fig.write_html(str(filepath))
    elif format in ['png', 'jpeg', 'pdf', 'svg']:
        # Requires kaleido
        try:
            fig.write_image(str(filepath), width=width, height=height, format=format)
        except Exception as e:
            print(f"Error: {e}")
            print("Install kaleido for image export: pip install kaleido")
            return None
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(filepath)


def export_matplotlib_chart(
    fig: plt.Figure,
    filename: str,
    format: str = 'png',
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> str:
    """
    Export Matplotlib chart to file.
    
    Args:
        fig: Matplotlib Figure object
        filename: Output filename (without extension)
        format: Output format ('png', 'pdf', 'svg', 'eps', 'jpg')
        dpi: Resolution for raster formats
        bbox_inches: Bounding box ('tight' removes whitespace)
        
    Returns:
        Path to saved file
    """
    filepath = Path(filename).with_suffix(f'.{format}')
    
    fig.savefig(
        str(filepath),
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches
    )
    
    return str(filepath)


def export_chart(
    fig: Union[go.Figure, plt.Figure],
    filename: str,
    format: str = 'png',
    **kwargs
) -> str:
    """
    Universal export function - auto-detects chart type.
    
    Args:
        fig: Figure object (Plotly or Matplotlib)
        filename: Output filename
        format: Output format
        **kwargs: Additional arguments for specific exporters
        
    Returns:
        Path to saved file
    """
    if isinstance(fig, go.Figure):
        return export_plotly_chart(fig, filename, format, **kwargs)
    elif isinstance(fig, plt.Figure):
        return export_matplotlib_chart(fig, filename, format, **kwargs)
    else:
        raise TypeError(f"Unsupported figure type: {type(fig)}")


def export_multiple_charts(
    figures: dict,
    output_dir: str = './exports',
    format: str = 'png'
) -> list:
    """
    Export multiple charts at once.
    
    Args:
        figures: Dict mapping filenames to Figure objects
        output_dir: Output directory
        format: Output format
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    saved_files = []
    
    for name, fig in figures.items():
        filepath = output_path / name
        saved_path = export_chart(fig, str(filepath), format)
        if saved_path:
            saved_files.append(saved_path)
    
    return saved_files


def get_chart_as_bytes(
    fig: Union[go.Figure, plt.Figure],
    format: str = 'png'
) -> bytes:
    """
    Get chart as bytes (for in-memory operations).
    
    Args:
        fig: Figure object
        format: Output format
        
    Returns:
        Bytes object
    """
    buffer = io.BytesIO()
    
    if isinstance(fig, go.Figure):
        if format == 'html':
            html_str = fig.to_html()
            buffer.write(html_str.encode())
        else:
            fig.write_image(buffer, format=format)
    elif isinstance(fig, plt.Figure):
        fig.savefig(buffer, format=format, bbox_inches='tight')
    
    buffer.seek(0)
    return buffer.getvalue()


# Export
__all__ = [
    'export_plotly_chart',
    'export_matplotlib_chart',
    'export_chart',
    'export_multiple_charts',
    'get_chart_as_bytes'
]
