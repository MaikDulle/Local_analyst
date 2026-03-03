"""
Local Analyst - Streamlit UI - ENHANCED VERSION
Fixes: caching, performance, reliability, UX, data export, dedup
Run with: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import hashlib
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_upload_engine import load_file, get_supported_extensions
from analysis_engine import (
    prepare_for_analysis,
    summarize_dataset,
    quick_stats,
    detect_column_roles,
    # Revenue
    revenue_by_period,
    revenue_by_dimension,
    pareto_analysis,
    growth_metrics,
    # Product
    product_performance,
    top_products,
    category_performance,
    # Customer
    rfm_analysis,
    customer_summary,
    customer_value_tiers,
    churn_risk_analysis,
    # Correlations
    correlation_matrix,
    find_strong_correlations,
    detect_outliers,
    # A/B Testing
    ab_test,
    # Cohort Analysis
    cohort_retention_analysis,
    # Campaign Tracking
    year_over_year_comparison,
    wave_season_comparison,
    campaign_performance_summary,
    # Interpretations
    interpret_ab_test,
    interpret_cohort_retention,
    interpret_revenue_trends,
    interpret_correlation,
    interpret_campaign_metrics,
    # Mixed correlations
    find_all_relationships,
    # Anomaly Detection
    detect_all_anomalies,
    prioritize_anomalies,
    # Funnel
    analyze_funnel,
    analyze_funnel_by_cohort,
    identify_bottlenecks,
    compare_funnels,
    # Attribution
    last_touch_attribution,
    first_touch_attribution,
    linear_attribution,
    time_decay_attribution,
    position_based_attribution,
    compare_attribution_models,
)

# Visualization imports
from viz_engine import (
    plot_cohort_heatmap,
    plot_rfm_scatter,
    plot_ab_test_comparison,
    plot_conversion_funnel,
    plot_attribution_waterfall,
)

# Page config
st.set_page_config(
    page_title="Local Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS
st.markdown("""
<style>
    .insight-box {
        background-color: #e8f4f8;
        padding: 0.75rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────
#  CACHING HELPERS
# ──────────────────────────────────────────────────

def _file_hash(file_bytes: bytes) -> str:
    """Fast hash for upload dedup."""
    return hashlib.md5(file_bytes).hexdigest()


def df_hash(df: pd.DataFrame) -> str:
    """Cheap hash for cache keys — uses shape + column names + head sample."""
    sig = f"{df.shape}|{'|'.join(df.columns)}|{df.head(3).to_json()}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


@st.cache_data(show_spinner=False)
def cached_summarize(_df_hash: str, _df: pd.DataFrame):
    """Cache dataset summary keyed by content hash."""
    return summarize_dataset(_df)


@st.cache_data(show_spinner=False)
def cached_quick_stats(_df_hash: str, _df: pd.DataFrame) -> pd.DataFrame:
    """Cache quick stats and return as a proper DataFrame (fixes dict→dataframe bug)."""
    _ = quick_stats(_df)  # run original for any side effects
    numeric_cols = _df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    stats = _df[numeric_cols].describe().T
    stats.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    stats = stats.round(2)
    return stats


@st.cache_data(show_spinner=False)
def cached_data_quality(_df_hash: str, _df: pd.DataFrame):
    """Cache data quality check."""
    return check_data_quality(_df)


@st.cache_data(show_spinner="Analyzing relationships…")
def cached_relationships(_df_hash: str, _df: pd.DataFrame, threshold: float):
    return find_all_relationships(_df, threshold=threshold)


@st.cache_data(show_spinner="Detecting anomalies…")
def cached_anomalies(_df_hash: str, _df: pd.DataFrame, date_col, sensitivity):
    return detect_all_anomalies(_df, date_col=date_col, sensitivity=sensitivity)


# ──────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────

def init_session_state():
    defaults = {
        'df': None,
        'raw_df': None,
        'load_result': None,
        'column_mapping': {},
        'selected_table': 0,
        'doc_tables': [],
        'file_hash': None,
        # AI settings
        'ai_backend': 'rule',       # 'rule' | 'local' | 'ollama'
        'ai_model_path': '',        # path to GGUF file for local backend
        'ai_ollama_model': 'llama3',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_analysis():
    """Reset all analysis results and cache."""
    keys_to_keep = {'df', 'raw_df', 'load_result', 'column_mapping',
                    'selected_table', 'file_hash', 'doc_tables'}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.cache_data.clear()
    st.success("✅ Analysis cache cleared!")
    st.rerun()


# ──────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.title("📊 Local Analyst")

    if st.sidebar.button("🔄 Reset Analysis", help="Clear all cached results"):
        reset_analysis()

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Upload Data")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'json', 'pdf', 'pptx', 'docx',
              'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
        help="Supported: CSV/TSV/TXT, Excel, JSON, PDF, PowerPoint, Word, Images"
    )

    if uploaded_file:
        process_upload(uploaded_file)

    # Persistent table selector for multi-table documents
    render_table_selector()

    if st.session_state.df is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Column Mapping")
        render_column_mapping()

        # Data filtering
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. Filter Data")
        render_data_filter()

        # Data export
        st.sidebar.markdown("---")
        st.sidebar.subheader("4. Export Data")
        _render_export()

    # AI Settings — always visible at the bottom
    st.sidebar.markdown("---")
    _render_ai_settings()


def _render_export():
    """Allow downloading the current (cleaned) DataFrame."""
    df = st.session_state.df
    if df is None:
        return

    fmt = st.sidebar.radio("Format", ["CSV", "Excel"], horizontal=True, key="_export_fmt")
    if fmt == "CSV":
        buf = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("⬇️ Download CSV", buf, "analyst_export.csv",
                                   "text/csv", use_container_width=True)
    else:
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        st.sidebar.download_button("⬇️ Download Excel", buf.getvalue(),
                                   "analyst_export.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)


_DOWNLOAD_MODELS = [
    {
        "label": "Qwen 2.5 · 1.5B · Q4_K_M  (~1.0 GB) ✓ Recommended",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
    {
        "label": "Llama 3.2 · 1B · Q4_K_M  (~0.8 GB) Fastest",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },
    {
        "label": "Phi-3 Mini · 4K · Q4  (~2.2 GB) Best quality",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    },
]


def _download_model_in_app(model: dict, models_dir: Path):
    """Stream-download a GGUF model with a Streamlit progress bar."""
    import requests

    dest = models_dir / model["filename"]
    resume_byte = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={resume_byte}-"} if resume_byte else {}

    status = st.empty()
    bar = st.progress(0)
    status.info(f"Connecting to HuggingFace…")

    try:
        with requests.get(model["url"], headers=headers, stream=True, timeout=30) as r:
            if r.status_code == 416:
                status.success("Already downloaded.")
                bar.progress(1.0)
                return True
            r.raise_for_status()

            total = int(r.headers.get("Content-Length", 0)) + resume_byte
            downloaded = resume_byte
            mode = "ab" if resume_byte else "wb"
            chunk = 1024 * 512  # 512 KB

            models_dir.mkdir(parents=True, exist_ok=True)
            with open(dest, mode) as f:
                for data in r.iter_content(chunk_size=chunk):
                    if data:
                        f.write(data)
                        downloaded += len(data)
                        if total:
                            pct = downloaded / total
                            mb = downloaded / (1024 ** 2)
                            total_mb = total / (1024 ** 2)
                            bar.progress(min(pct, 1.0))
                            status.info(f"Downloading… {mb:.0f} / {total_mb:.0f} MB ({pct*100:.1f}%)")

        bar.progress(1.0)
        status.success(f"Downloaded: {dest.name}")
        return True

    except Exception as e:
        status.error(f"Download failed: {e}")
        return False


def _render_ai_settings():
    """AI interpretation settings in the sidebar."""
    from ai.local_llm import find_local_models, default_model_path, is_available

    with st.sidebar.expander("🤖 AI Settings", expanded=False):
        backend = st.radio(
            "Interpretation mode",
            options=["rule", "local", "ollama"],
            format_func=lambda x: {
                "rule":   "Rule-based (always available)",
                "local":  "Local LLM  —  no server needed",
                "ollama": "Ollama  —  server required",
            }[x],
            index=["rule", "local", "ollama"].index(
                st.session_state.get('ai_backend', 'rule')
            ),
            key="_ai_backend_sel",
        )
        st.session_state['ai_backend'] = backend

        if backend == "local":
            # ── llama-cpp-python check ──
            if not is_available():
                st.warning(
                    "Install the inference engine first (pre-built, no compiler needed):\n\n"
                    "```\npip install llama-cpp-python "
                    "--only-binary=llama-cpp-python "
                    "--extra-index-url "
                    "https://abetlen.github.io/llama-cpp-python/whl/cpu\n```"
                )

            # ── auto-detect models ──
            models_dir = Path(__file__).parent.parent / "models"
            found = find_local_models(models_dir)

            if found:
                model_labels = [p.name for p in found]
                # Try to keep previously selected model
                prev = st.session_state.get('ai_model_path', '')
                prev_name = Path(prev).name if prev else ''
                default_idx = next(
                    (i for i, p in enumerate(found) if p.name == prev_name), 0
                )
                selected_name = st.selectbox(
                    "Model", model_labels, index=default_idx, key="_ai_model_sel"
                )
                chosen_path = str(models_dir / selected_name)
                st.session_state['ai_model_path'] = chosen_path
                st.success(f"Ready — {Path(chosen_path).stat().st_size // (1024**2)} MB")
            else:
                st.info("No model found in `models/` folder.")
                st.session_state['ai_model_path'] = ''

            # ── download section ──
            st.markdown("**Download a model**")
            dl_labels = [m["label"] for m in _DOWNLOAD_MODELS]
            dl_choice = st.selectbox("Choose model", dl_labels, key="_ai_dl_choice")
            dl_model = _DOWNLOAD_MODELS[dl_labels.index(dl_choice)]
            dest_path = models_dir / dl_model["filename"]

            if dest_path.exists():
                st.caption(f"✓ Already in models/ ({dest_path.stat().st_size // (1024**2)} MB)")

            if st.button("⬇️ Download", key="_ai_dl_btn", use_container_width=True):
                _download_model_in_app(dl_model, models_dir)
                st.rerun()

            st.caption("Or run `python download_model.py` in a terminal.")

        elif backend == "ollama":
            model_name = st.text_input(
                "Ollama model",
                value=st.session_state.get('ai_ollama_model', 'llama3'),
                key="_ai_ollama_model_inp",
            )
            st.session_state['ai_ollama_model'] = model_name
            st.caption("Run `ollama serve` and pull the model first.")


def _get_ai_kwargs() -> dict:
    """Return kwargs to pass to AI interpreter functions based on current settings."""
    backend = st.session_state.get('ai_backend', 'rule')
    if backend == 'rule':
        return {'use_ai': False}
    if backend == 'local':
        return {
            'use_ai': True,
            'model_path': st.session_state.get('ai_model_path', '') or None,
        }
    # ollama
    return {
        'use_ai': True,
        'model_path': None,
        'ollama_model': st.session_state.get('ai_ollama_model', 'llama3'),
    }


def process_upload(uploaded_file):
    """Process uploaded file — skip if already loaded (same content)."""
    import tempfile, os

    file_bytes = uploaded_file.getvalue()
    fhash = _file_hash(file_bytes)

    # Skip re-processing if same file
    if st.session_state.file_hash == fhash and st.session_state.df is not None:
        return

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    with st.spinner("Loading file…"):
        result = load_file(temp_path)

    if not result.success:
        st.sidebar.error(f"Error: {result.error}")
        return

    st.session_state.load_result = result
    st.session_state.file_hash = fhash

    # Handle documents with multiple tables
    if result.is_document and result.document_content:
        tables = getattr(result.document_content, 'tables', None) or getattr(result.document_content, 'all_tables', [])
        st.session_state.doc_tables = tables if tables else []

        # Show extraction summary for documents
        ext = Path(uploaded_file.name).suffix.lower()
        img_extraction = result.metadata.get('image_extraction', {})
        if img_extraction:
            charts = img_extraction.get('charts_with_data', 0)
            imgs = img_extraction.get('images_with_text', 0)
            total = img_extraction.get('images_found', 0)
            if charts or imgs:
                st.sidebar.info(
                    f"Image extraction: {total} image(s) found — "
                    f"{charts} native chart(s), {imgs} via OCR."
                )
        # Show OCR warning for PDFs that look scanned
        if result.metadata.get('is_likely_scanned'):
            st.sidebar.warning(
                "This PDF appears to be image-based (scanned). "
                "OCR was used to extract data. Results may need cleanup."
            )

        if not tables:
            if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'):
                st.sidebar.warning("No data extracted from image.")
                ocr_backend = result.metadata.get('ocr_backend', 'unknown')
                if ocr_backend == 'none':
                    st.sidebar.info(
                        "**No OCR backend installed.** Install one:\n\n"
                        "```\npip install easyocr\n```\n"
                        "*(pure Python — recommended)*"
                    )
                else:
                    st.sidebar.info(
                        "No structured data found in image. "
                        "The chart may not have legible text labels."
                    )
            elif ext in ('.pdf',):
                st.sidebar.warning(
                    "No tables found in this PDF. "
                    "If it contains charts as images, make sure `easyocr` is installed:\n\n"
                    "```\npip install easyocr\n```"
                )
            elif ext in ('.pptx', '.ppt'):
                st.sidebar.warning(
                    "No tables found in this presentation. "
                    "For slides with only chart images, install `easyocr` to extract data via OCR:\n\n"
                    "```\npip install easyocr\n```"
                )
            else:
                st.sidebar.warning("No tables found in document.")
            st.session_state.df = None
            return
        # Default to first table — selector rendered separately
        raw_df = tables[st.session_state.selected_table]
    else:
        st.session_state.doc_tables = []
        raw_df = result.df

    st.session_state.raw_df = raw_df
    st.session_state.df = prepare_for_analysis(raw_df)

    # Auto-detect column mapping
    if result.ecom_mapping:
        st.session_state.column_mapping = result.ecom_mapping
    else:
        roles = detect_column_roles(st.session_state.df)
        st.session_state.column_mapping = {
            'date_column': roles['date_columns'][0] if roles['date_columns'] else None,
            'revenue_column': roles['revenue_columns'][0] if roles['revenue_columns'] else None,
            'quantity_column': roles['quantity_columns'][0] if roles['quantity_columns'] else None,
            'customer_id': roles['id_columns'][0] if roles['id_columns'] else None,
            'product_id': None,
            'category_column': roles['category_columns'][0] if roles['category_columns'] else None,
        }

    st.sidebar.success(f"Loaded: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} cols")


def render_table_selector():
    """Render table selector for multi-table documents. Persists across reruns."""
    tables = st.session_state.get('doc_tables', [])
    if len(tables) < 2:
        return

    st.sidebar.success(f"Found {len(tables)} tables")
    table_names = [f"Table {i+1}: {t.shape[0]}×{t.shape[1]}" for i, t in enumerate(tables)]

    # Clamp index to valid range
    current_idx = min(st.session_state.selected_table, len(table_names) - 1)

    selected = st.sidebar.selectbox("Select table", table_names,
                                    index=current_idx, key="_doc_table_sel")
    new_idx = table_names.index(selected)

    if new_idx != st.session_state.selected_table:
        st.session_state.selected_table = new_idx
        raw_df = tables[new_idx]
        st.session_state.raw_df = raw_df
        st.session_state.df = prepare_for_analysis(raw_df)
        st.rerun()


def render_column_mapping():
    df = st.session_state.df
    columns = ['(none)'] + list(df.columns)
    mapping = st.session_state.column_mapping

    def _select(label, key):
        idx = columns.index(mapping.get(key)) if mapping.get(key) in columns else 0
        val = st.sidebar.selectbox(label, columns, index=idx, key=f"_map_{key}")
        mapping[key] = val if val != '(none)' else None

    _select("📅 Date column", 'date_column')
    _select("💰 Revenue column", 'revenue_column')
    _select("📦 Quantity column", 'quantity_column')
    _select("👤 Customer ID", 'customer_id')
    _select("🏷️ Category/Product", 'category_column')

    st.session_state.column_mapping = mapping


def render_data_filter():
    """Allow users to filter rows before analysis."""
    df = st.session_state.raw_df
    if df is None:
        return

    with st.sidebar.expander("🔍 Row Filters", expanded=False):
        active_filters = {}

        # Date range filter
        date_col = st.session_state.column_mapping.get('date_column')
        if date_col and date_col in df.columns:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                if len(dates) > 0:
                    min_d, max_d = dates.min().date(), dates.max().date()
                    date_range = st.date_input(
                        f"📅 {date_col}", value=(min_d, max_d),
                        min_value=min_d, max_value=max_d, key="_filt_date"
                    )
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        active_filters['date'] = (date_col, date_range)
            except Exception:
                pass

        # Numeric range filters (up to 3 columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:3]
        for col in numeric_cols:
            col_min, col_max = float(df[col].min()), float(df[col].max())
            if col_min == col_max:
                continue
            vals = st.slider(
                f"📊 {col}", col_min, col_max, (col_min, col_max),
                key=f"_filt_num_{col}"
            )
            if vals[0] > col_min or vals[1] < col_max:
                active_filters[f"num_{col}"] = (col, vals)

        # Categorical filter (up to 2 columns with <20 unique values)
        cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                    if 2 <= df[c].nunique() <= 20][:2]
        for col in cat_cols:
            options = sorted(df[col].dropna().unique().tolist())
            selected = st.multiselect(f"🏷️ {col}", options, default=options, key=f"_filt_cat_{col}")
            if len(selected) < len(options):
                active_filters[f"cat_{col}"] = (col, selected)

        # Apply filters
        if active_filters:
            filtered = df.copy()
            for key, val in active_filters.items():
                if key == 'date':
                    col_name, date_range = val
                    dt = pd.to_datetime(filtered[col_name], errors='coerce')
                    filtered = filtered[
                        (dt >= pd.Timestamp(date_range[0])) &
                        (dt <= pd.Timestamp(date_range[1]))
                    ]
                elif key.startswith('num_'):
                    col_name, (lo, hi) = val
                    filtered = filtered[(filtered[col_name] >= lo) & (filtered[col_name] <= hi)]
                elif key.startswith('cat_'):
                    col_name, selected = val
                    filtered = filtered[filtered[col_name].isin(selected)]

            new_df = prepare_for_analysis(filtered)
            if len(new_df) != len(st.session_state.df):
                st.session_state.df = new_df
                st.sidebar.caption(f"🔽 Filtered: {len(new_df):,} of {len(df):,} rows")
        else:
            # Restore full dataset if no filters active
            if len(st.session_state.df) != len(df):
                st.session_state.df = prepare_for_analysis(df)


# ──────────────────────────────────────────────────
#  DATA QUALITY
# ──────────────────────────────────────────────────

def check_data_quality(df):
    warnings = []
    issues = []

    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 50:
        issues.append("critical")
        warnings.append(f"🔴 **CRITICAL**: {missing_pct:.1f}% of data is missing")
    elif missing_pct > 30:
        issues.append("poor")
        warnings.append(f"🟠 **High missing data**: {missing_pct:.1f}% missing")
    elif missing_pct > 10:
        issues.append("fair")
        warnings.append(f"🟡 **Moderate missing data**: {missing_pct:.1f}% missing")

    dup_count = df.duplicated().sum()
    dup_pct = (dup_count / len(df)) * 100 if len(df) > 0 else 0
    if dup_pct > 50:
        issues.append("critical")
        warnings.append(f"🔴 **CRITICAL**: {dup_pct:.1f}% duplicate rows ({dup_count:,})")
    elif dup_pct > 20:
        issues.append("poor")
        warnings.append(f"🟠 **High duplicates**: {dup_pct:.1f}% ({dup_count:,})")
    elif dup_pct > 5:
        issues.append("fair")
        warnings.append(f"🟡 **Some duplicates**: {dup_pct:.1f}% ({dup_count:,})")

    if len(df) < 10:
        issues.append("critical")
        warnings.append(f"🔴 **CRITICAL**: Only {len(df)} rows — too small for meaningful analysis")
    elif len(df) < 50:
        issues.append("poor")
        warnings.append(f"🟠 **Small dataset**: Only {len(df)} rows")
    elif len(df) < 100:
        issues.append("fair")
        warnings.append(f"🟡 **Limited data**: {len(df)} rows")

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0 and len(df.columns) > 0:
        issues.append("poor")
        warnings.append("🟠 **No numeric columns**: Cannot perform quantitative analysis")

    single_value_cols = [c for c in df.columns if df[c].nunique() == 1]
    if single_value_cols:
        warnings.append(f"ℹ️ **{len(single_value_cols)} constant column(s)**: {', '.join(single_value_cols[:3])}")

    if "critical" in issues:
        quality_level = "critical"
    elif "poor" in issues:
        quality_level = "poor"
    elif "fair" in issues:
        quality_level = "fair"
    else:
        quality_level = "good"

    return quality_level, warnings


def render_data_quality_banner(df):
    h = df_hash(df)
    quality_level, warnings = cached_data_quality(h, df)

    if quality_level == "good":
        return

    if quality_level == "critical":
        st.error("🚨 **CRITICAL DATA QUALITY ISSUES** — results will be unreliable!")
    elif quality_level == "poor":
        st.warning("⚠️ **DATA QUALITY ISSUES** — results may not be accurate")
    else:
        st.info("💡 **DATA QUALITY NOTICE**")

    with st.expander("📋 View Details", expanded=(quality_level == "critical")):
        for w in warnings:
            st.markdown(f"- {w}")
    st.markdown("---")


# ──────────────────────────────────────────────────
#  MAIN CONTENT
# ──────────────────────────────────────────────────

def render_main_content():
    if st.session_state.df is None:
        render_welcome()
        return

    df = st.session_state.df
    mapping = st.session_state.column_mapping

    render_data_quality_banner(df)

    tab_names = [
        "📋 Overview",
        "💰 Revenue",
        "📦 Products",
        "👥 Customers",
        "🔗 Correlations",
        "🧪 A/B Testing",
        "📊 Cohort Analysis",
        "🎯 Campaign Tracking",
        "🔀 Funnel",
        "📡 Attribution",
        "🚨 Anomaly Detection",
        "🔎 Full Scan",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_overview_tab(df)
    with tabs[1]:
        render_revenue_tab(df, mapping)
    with tabs[2]:
        render_products_tab(df, mapping)
    with tabs[3]:
        render_customers_tab(df, mapping)
    with tabs[4]:
        render_correlations_tab(df)
    with tabs[5]:
        render_ab_testing_tab(df, mapping)
    with tabs[6]:
        render_cohort_tab(df, mapping)
    with tabs[7]:
        render_campaign_tab(df, mapping)
    with tabs[8]:
        render_funnel_tab(df, mapping)
    with tabs[9]:
        render_attribution_tab(df, mapping)
    with tabs[10]:
        render_anomaly_tab(df, mapping)
    with tabs[11]:
        render_full_scan_tab(df, mapping)


def render_welcome():
    st.title("📊 Local Analyst")
    st.markdown("""
    ### Your local data analyst — no cloud, no API keys

    Upload a file to get started:
    - **CSV, Excel, JSON** — Tabular data
    - **PDF, PowerPoint, Word** — Extract tables and KPIs from reports
    - **JPEG, PNG, BMP, TIFF** — OCR extraction from chart screenshots and dashboard images

    #### Analysis Features
    - 📋 **Overview** — Data quality, statistics, auto-insights
    - 💰 **Revenue** — Trends, growth, Pareto analysis
    - 📦 **Products** — Top performers, category performance
    - 👥 **Customers** — RFM segmentation, churn risk
    - 🔗 **Correlations** — Numeric, categorical, and mixed relationships
    - 🧪 **A/B Testing** — Statistical significance with effect sizes
    - 📊 **Cohort Analysis** — Retention and LTV tracking
    - 🎯 **Campaign Tracking** — YoY comparison, Wave Season analysis
    - 🔀 **Funnel** — Conversion funnels with bottleneck detection
    - 📡 **Attribution** — Multi-touch attribution model comparison
    - 🚨 **Anomaly Detection** — Value, pattern, and sequence anomalies
    - 🔎 **Full Scan** — One-click scan for trends, anomalies, and odd patterns

    ---
    *All analysis runs locally. Your data never leaves your machine.*
    """)


# ──────────────────────────────────────────────────
#  OVERVIEW TAB
# ──────────────────────────────────────────────────

def render_overview_tab(df):
    st.subheader("📋 Dataset Overview")

    h = df_hash(df)
    summary = cached_summarize(h, df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary.row_count:,}")
    col2.metric("Columns", summary.column_count)
    col3.metric("Missing %", f"{summary.total_missing_pct:.1f}%")
    col4.metric("Duplicates", f"{summary.duplicate_rows:,}")

    st.markdown("#### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Column Types")
        type_df = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(df[c].dtype) for c in df.columns],
            'Non-Null': [df[c].notna().sum() for c in df.columns],
            'Unique': [df[c].nunique() for c in df.columns]
        })
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Auto-Generated Insights")
        if summary.insights:
            for insight in summary.insights:
                st.info(insight)
        else:
            st.success("✓ No issues detected")

    if summary.numeric_columns:
        st.markdown("---")
        st.markdown("#### Numeric Column Statistics")
        stats_df = cached_quick_stats(h, df)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)

    # AI interpretation of dataset overview
    st.markdown("---")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    stats_lines = []
    for col in numeric_cols[:5]:
        stats_lines.append(
            f"{col}: mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}"
        )
    _ai_interpret(
        system="You are a concise data analyst. Describe datasets factually in plain English. Be brief.",
        user=(
            f"Dataset: {summary.row_count:,} rows, {summary.column_count} columns. "
            f"Missing: {summary.total_missing_pct:.1f}%. Duplicates: {summary.duplicate_rows:,}. "
            f"Columns: {', '.join(df.columns.tolist()[:10])}. "
            f"Numeric stats: {'; '.join(stats_lines) if stats_lines else 'none'}. "
            f"Issues noted: {'; '.join(summary.insights[:3]) if summary.insights else 'none'}. "
            f"Summarize what this dataset likely contains and flag any data quality concerns. "
            f"3 bullet points max."
        ),
        key="_ai_overview_btn",
    )


# ──────────────────────────────────────────────────
#  REVENUE TAB
# ──────────────────────────────────────────────────

def render_revenue_tab(df, mapping):
    st.subheader("💰 Revenue Analysis")

    rev_col = mapping.get('revenue_column')
    date_col = mapping.get('date_column')
    cat_col = mapping.get('category_column')

    if not rev_col:
        st.warning("⚠️ Please select a revenue column in the sidebar")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Revenue by Category", use_container_width=True):
            if cat_col:
                result = revenue_by_dimension(df, cat_col, rev_col)
                st.session_state.rev_result = ('revenue_by_category', result)
            else:
                st.warning("Select a category column first")

    with col2:
        if st.button("📈 Pareto Analysis", use_container_width=True):
            if cat_col:
                result = pareto_analysis(df, cat_col, rev_col)
                st.session_state.rev_result = ('pareto', result)
            else:
                st.warning("Select a category column first")

    with col3:
        if st.button("📉 Revenue Trends", use_container_width=True):
            if date_col:
                result = revenue_by_period(df, date_col, rev_col, period='M')
                st.session_state.rev_result = ('revenue_trend', result)
            else:
                st.warning("Select a date column first")

    with col4:
        if st.button("📈 Growth Metrics", use_container_width=True):
            if date_col:
                result = growth_metrics(df, date_col, rev_col)
                st.session_state.rev_result = ('growth', result)
            else:
                st.warning("Select a date column first")

    if 'rev_result' not in st.session_state:
        return

    analysis_type, result = st.session_state.rev_result
    st.markdown("---")

    if analysis_type == 'revenue_by_category':
        st.markdown("#### Revenue by Category")
        st.dataframe(result, use_container_width=True, hide_index=True)
        if len(result) > 0 and cat_col in result.columns:
            try:
                st.bar_chart(result.set_index(cat_col)['revenue'])
            except Exception:
                pass

    elif analysis_type == 'pareto':
        st.markdown("#### Pareto Analysis (80/20)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vital Few", f"{result['vital_few_count']} items")
        c2.metric("% of Items", f"{result['vital_few_pct']:.1f}%")
        c3.metric("% of Revenue", f"{result['vital_few_revenue_pct']:.1f}%")
        st.markdown(f"**Top performers:** {', '.join(map(str, result['vital_few_items'][:5]))}")
        if result['is_concentrated']:
            st.info("💡 Revenue is highly concentrated — focus on these key items")

    elif analysis_type == 'revenue_trend':
        st.markdown("#### Revenue Trend")
        st.dataframe(result, use_container_width=True, hide_index=True)
        if len(result) > 1:
            try:
                st.line_chart(result.set_index(date_col)['revenue'])
            except Exception:
                pass
            with st.expander("💡 Revenue Trend Interpretation", expanded=True):
                try:
                    interp = interpret_revenue_trends(result, date_col, 'revenue')
                    st.markdown(f"### {interp.summary}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**📊 Key Findings:**")
                        for f in interp.key_findings:
                            st.markdown(f"- {f}")
                    with c2:
                        st.markdown("**✅ Recommendations:**")
                        for r in interp.recommendations:
                            st.markdown(f"- {r}")
                    for w in interp.warnings:
                        st.warning(w)
                except Exception as e:
                    st.warning(f"Could not generate interpretation: {e}")

    elif analysis_type == 'growth':
        st.markdown("#### Growth Metrics")
        if isinstance(result, dict):
            metrics = {k: v for k, v in result.items() if v is not None}
            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Revenue", f"{metrics.get('latest_revenue', 0):,.0f}")
            c2.metric("Period Growth", f"{metrics.get('period_growth', 0):+.1f}%")
            c3.metric("Trend", metrics.get('trend', 'unknown').capitalize())
            st.json(metrics)


# ──────────────────────────────────────────────────
#  PRODUCTS TAB  (isolated session key: product_result)
# ──────────────────────────────────────────────────

def render_products_tab(df, mapping):
    st.subheader("📦 Product Analysis")

    rev_col = mapping.get('revenue_column')
    cat_col = mapping.get('category_column')
    qty_col = mapping.get('quantity_column')

    if not cat_col:
        st.warning("⚠️ Please select a category/product column in the sidebar")
        return

    num_products = df[cat_col].nunique()
    if num_products < 2:
        st.error(f"⚠️ Need at least 2 unique products. Found: {num_products}")
        return

    c1, c2 = st.columns(2)

    with c1:
        max_top_n = min(10, num_products)
        top_n = st.slider("Number of top products", 1, max_top_n, min(5, max_top_n))
        if st.button("🏆 Top Products", use_container_width=True):
            if rev_col:
                try:
                    result = product_performance(df, cat_col, rev_col, qty_col, top_n=top_n)
                    st.session_state.product_result = ('top_products', result)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Select a revenue column first")

    with c2:
        if st.button("📊 Category Performance", use_container_width=True):
            if rev_col:
                try:
                    result = category_performance(df, cat_col, rev_col, quantity_col=qty_col)
                    st.session_state.product_result = ('category_perf', result)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Select a revenue column first")

    if 'product_result' not in st.session_state:
        return

    analysis_type, result = st.session_state.product_result
    st.markdown("---")

    if analysis_type == 'top_products':
        st.markdown("#### Top Products by Revenue")
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result, use_container_width=True, hide_index=True)
            if 'revenue' in result.columns and cat_col in result.columns:
                try:
                    st.bar_chart(result.set_index(cat_col)['revenue'])
                except Exception:
                    pass
        else:
            st.warning("No product data available")

    elif analysis_type == 'category_perf':
        st.markdown("#### Category Performance")
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.warning("No category data available")


# ──────────────────────────────────────────────────
#  CUSTOMERS TAB  (isolated session key: cust_result)
# ──────────────────────────────────────────────────

def render_customers_tab(df, mapping):
    st.subheader("👥 Customer Analysis")

    cust_col = mapping.get('customer_id')
    rev_col = mapping.get('revenue_column')
    date_col = mapping.get('date_column')

    if not cust_col:
        st.warning("⚠️ Please select a customer ID column in the sidebar")
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("📊 Customer Summary", use_container_width=True):
            if rev_col:
                result = customer_summary(df, cust_col, rev_col, date_col)
                st.session_state.cust_result = ('summary', result)
            else:
                st.warning("Select a revenue column first")

    with c2:
        if st.button("🎯 RFM Segmentation", use_container_width=True):
            if rev_col and date_col:
                result = rfm_analysis(df, cust_col, date_col, rev_col)
                st.session_state.cust_result = ('rfm', result)
            else:
                st.warning("Need date and revenue columns")

    with c3:
        if st.button("💎 Value Tiers", use_container_width=True):
            if rev_col:
                result = customer_value_tiers(df, cust_col, rev_col)
                st.session_state.cust_result = ('tiers', result)
            else:
                st.warning("Select a revenue column first")

    if 'cust_result' not in st.session_state:
        return

    analysis_type, result = st.session_state.cust_result
    st.markdown("---")

    if analysis_type == 'summary':
        st.markdown("#### Customer Summary")
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)
        st.markdown(f"**Total customers:** {len(result):,}")
        if 'total_revenue' in result.columns:
            st.markdown(f"**Total revenue:** {result['total_revenue'].sum():,.2f}")

    elif analysis_type == 'rfm':
        st.markdown("#### RFM Segmentation")
        segment_counts = result['segment'].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Segment Distribution**")
            st.dataframe(segment_counts, use_container_width=True)
        with c2:
            st.bar_chart(segment_counts)
        st.markdown("**Customer Details**")
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)

    elif analysis_type == 'tiers':
        st.markdown("#### Customer Value Tiers")
        if isinstance(result, dict) and 'tier_summary' in result:
            st.dataframe(result['tier_summary'], use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────
#  CORRELATIONS TAB
# ──────────────────────────────────────────────────

def _render_ai_result(text: str):
    """Render an AI interpretation block with a consistent disclaimer."""
    st.markdown("---")
    st.markdown("**🤖 AI Interpretation**")
    st.info(text)
    st.caption(
        "⚠️ AI interpretation is generated by a local language model and may be incorrect, "
        "incomplete, or misleading. Always verify findings against the actual data above."
    )


def _ai_interpret(system: str, user: str, key: str, spinner_text: str = "Thinking…"):
    """Render an 'Get AI Interpretation' button and show result inline."""
    if st.session_state.get('ai_backend', 'rule') == 'rule':
        return
    if st.button("🤖 Get AI Interpretation", key=key):
        try:
            from ai.local_llm import get_llm
            model_path = st.session_state.get('ai_model_path', '')
            if not model_path:
                st.warning("No model selected. Set a model path in AI Settings.")
                return
            llm = get_llm(model_path)
            with st.spinner(spinner_text):
                result = llm.chat(system=system, user=user)
            _render_ai_result(result)
        except Exception as e:
            st.error(f"AI failed: {e}")


def render_correlations_tab(df):
    st.subheader("🔗 Correlation & Association Analysis")
    st.info("💡 Analyzes relationships between ALL types of variables — numeric, categorical, and mixed.")

    c1, c2 = st.columns(2)
    with c2:
        threshold = st.slider("Minimum Strength (for highlights)", 0.1, 0.9, 0.3, 0.1)
    with c1:
        if st.button("📊 Analyze All Relationships", use_container_width=True):
            h = df_hash(df)
            result = cached_relationships(h, df, threshold)
            st.session_state.all_relationships = result

    if 'all_relationships' not in st.session_state:
        return

    result = st.session_state.all_relationships
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Columns", result['summary']['total_columns'])
    c2.metric("Numeric Columns", result['summary']['numeric_columns'])
    c3.metric("Categorical Columns", result['summary']['categorical_columns'])

    # ── Numeric correlations — always shown ──
    numeric_matrix = result.get('numeric_matrix')
    numeric_all = result.get('numeric_all', [])

    if numeric_matrix is not None and not numeric_matrix.empty and len(numeric_matrix.columns) >= 2:
        st.markdown("### 🔢 Numeric Correlations")

        # Heatmap — always rendered regardless of threshold
        fig_corr = px.imshow(
            numeric_matrix.round(3),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson Correlation Matrix",
            aspect="auto",
        )
        fig_corr.update_layout(height=max(300, len(numeric_matrix.columns) * 60))
        st.plotly_chart(fig_corr, use_container_width=True)

        # All pairs table sorted by abs value
        if numeric_all:
            pairs_df = pd.DataFrame(numeric_all).sort_values('abs_correlation', ascending=False)
            for _, row in pairs_df.iterrows():
                direction = "↗️" if row['type'] == 'positive' else "↘️"
                strength = "Strong" if row['abs_correlation'] > 0.7 else "Moderate" if row['abs_correlation'] > 0.4 else "Weak"
                flag = " ✓" if row['abs_correlation'] >= threshold else ""
                st.markdown(f"{direction} **{row['var1']}** ↔ **{row['var2']}**: {row['correlation']:.3f} ({strength}){flag}")

            if not result['numeric']:
                st.caption(f"No pair exceeds the {threshold:.1f} threshold — all correlations shown above.")

        with st.expander("💡 Rule-based Interpretation", expanded=True):
            try:
                # Use strongest pair whether or not it passes threshold
                all_pairs = result['numeric'] or numeric_all
                if all_pairs:
                    top_pairs = sorted(all_pairs, key=lambda x: x['abs_correlation'], reverse=True)
                    top = top_pairs[0]
                    interp = interpret_correlation(top['var1'], top['var2'], top['correlation'], len(df))
                    st.markdown(f"**{interp.summary}**")
                    c1, c2 = st.columns(2)
                    with c1:
                        for f in interp.key_findings:
                            st.markdown(f"- {f}")
                    with c2:
                        for r in interp.recommendations:
                            st.markdown(f"- {r}")
                else:
                    st.info("No numeric pairs to interpret.")
            except Exception as e:
                st.warning(f"Could not generate interpretation: {e}")

        # AI interpretation
        if st.session_state.get('ai_backend', 'rule') != 'rule':
            if st.button("🤖 Get AI Interpretation", key="_ai_corr_btn"):
                try:
                    from ai.local_llm import get_llm
                    model_path = st.session_state.get('ai_model_path', '')
                    llm = get_llm(model_path)
                    pairs_summary = "; ".join(
                        f"{r['var1']} vs {r['var2']}: r={r['correlation']:.3f}"
                        for r in sorted(numeric_all, key=lambda x: x['abs_correlation'], reverse=True)[:6]
                    )
                    prompt = (
                        f"Dataset has {len(df):,} rows. "
                        f"Numeric correlations: {pairs_summary}. "
                        f"List only notable findings and business implications in 3 bullet points. "
                        f"If correlations are all near zero, say so clearly."
                    )
                    with st.spinner("Thinking…"):
                        ai_text = llm.chat(system="You are a concise data analyst. Be factual and brief.", user=prompt)
                    _render_ai_result(ai_text)
                except Exception as e:
                    st.error(f"AI failed: {e}")
    else:
        st.info("No numeric columns found — nothing to correlate numerically.")

    if result['categorical']:
        st.markdown("### 🏷️ Categorical Associations")
        for assoc in result['categorical'][:10]:
            st.markdown(f"🔗 **{assoc['variable_1']}** ↔ **{assoc['variable_2']}**: {assoc['cramers_v']:.3f} ({assoc['strength']})")

    if result['mixed']:
        st.markdown("### 🔀 Mixed Type Relationships")
        for rel in result['mixed'][:10]:
            st.markdown(f"📊 **{rel['numeric_var']}** differs by **{rel['categorical_var']}**: η² = {rel['eta_squared']:.3f} ({rel['strength']})")

    if not result['categorical'] and not result['mixed'] and (numeric_matrix is None or numeric_matrix.empty):
        st.warning("Not enough data to compute relationships. Need at least 2 columns of the same or compatible types.")


# ──────────────────────────────────────────────────
#  A/B TESTING TAB
# ──────────────────────────────────────────────────

def render_ab_testing_tab(df, mapping):
    st.subheader("🧪 A/B Testing")

    variant_col = st.selectbox("Variant Column", df.columns, key="_ab_variant")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for testing")
        return

    metric_col = st.selectbox("Metric to Test", numeric_cols, key="_ab_metric")
    variants = df[variant_col].unique()

    if len(variants) < 2:
        st.warning("⚠️ Need at least 2 variants for testing")
        return

    c1, c2 = st.columns(2)
    with c1:
        variant_a = st.selectbox("Control (A)", variants, key="_ab_a")
    with c2:
        available_b = [v for v in variants if v != variant_a]
        if not available_b:
            st.warning("Select different variants")
            return
        variant_b = st.selectbox("Treatment (B)", available_b, key="_ab_b")

    confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="_ab_conf")

    if st.button("🔍 Run A/B Test", use_container_width=True):
        with st.spinner("Running statistical test…"):
            result = ab_test(df, variant_col, metric_col, variant_a, variant_b, confidence)
        st.session_state.ab_result = result

    if 'ab_result' not in st.session_state:
        return

    result = st.session_state.ab_result
    st.markdown("---")
    st.markdown("### 📊 Test Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Control ({result.variant_a_name})", f"{result.variant_a_mean:.4f}")
    c2.metric(f"Treatment ({result.variant_b_name})", f"{result.variant_b_mean:.4f}",
              delta=f"{result.lift_percentage:.2f}%")
    c3.metric("P-value", f"{result.p_value:.4f}")
    c4.metric("Significant?", "✅ Yes" if result.is_significant else "❌ No")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample A", f"{result.variant_a_count:,}")
    c2.metric("Sample B", f"{result.variant_b_count:,}")
    c3.metric("Effect Size", f"{result.cohens_d:.3f} ({result.effect_size_interpretation})")

    try:
        fig = plot_ab_test_comparison(
            result.variant_a_name, result.variant_b_name,
            result.variant_a_mean, result.variant_b_mean,
            result.confidence_interval_a, result.confidence_interval_b,
            result.lift_percentage, result.is_significant, result.p_value,
            interactive=True
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Visualization error: {e}")

    with st.expander("💡 Detailed Interpretation", expanded=True):
        try:
            interp = interpret_ab_test(
                result.variant_a_name, result.variant_b_name,
                result.lift_percentage, result.p_value,
                result.is_significant, result.effect_size_interpretation,
                result.variant_a_count, result.variant_b_count, confidence
            )
            st.markdown(f"### {interp.summary}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📊 Key Findings:**")
                for f in interp.key_findings:
                    st.markdown(f"- {f}")
            with c2:
                st.markdown("**✅ Recommendations:**")
                for r in interp.recommendations:
                    st.markdown(f"- {r}")
            for w in interp.warnings:
                st.warning(w)
        except Exception as e:
            st.warning(f"Could not generate interpretation: {e}")

        # AI enhancement (optional)
        if st.session_state.get('ai_backend', 'rule') != 'rule':
            if st.button("🤖 Get AI Interpretation", key="_ai_ab_btn"):
                try:
                    from ai.interpreter import interpret_ab_test as ai_interpret_ab
                    ai_kwargs = _get_ai_kwargs()
                    ai_result = ai_interpret_ab(
                        result.variant_a_name, result.variant_b_name,
                        result.lift_percentage, result.p_value,
                        result.is_significant, result.effect_size_interpretation,
                        (result.variant_a_count, result.variant_b_count),
                        **ai_kwargs,
                    )
                    st.markdown("---")
                    st.markdown(f"**🤖 AI Analysis** *(confidence: {ai_result.confidence})*")
                    st.info(ai_result.summary)
                    for finding in ai_result.key_findings:
                        st.markdown(f"- {finding}")
                    if ai_result.recommendations:
                        st.markdown("**Recommendations:**")
                        for rec in ai_result.recommendations:
                            st.markdown(f"- {rec}")
                except Exception as e:
                    st.error(f"AI interpretation failed: {e}")


# ──────────────────────────────────────────────────
#  COHORT TAB
# ──────────────────────────────────────────────────

def render_cohort_tab(df, mapping):
    st.subheader("📊 Cohort Analysis")

    cust_col = mapping.get('customer_id')
    date_col = mapping.get('date_column')
    rev_col = mapping.get('revenue_column')

    if not cust_col or not date_col:
        st.warning("⚠️ Need customer ID and date columns. Please configure in sidebar.")
        return

    period = st.selectbox("Cohort Period",
                          options=['D', 'W', 'M', 'Q'],
                          format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly'}[x],
                          index=2, key="_cohort_period")

    if st.button("🔍 Analyze Cohorts", use_container_width=True):
        with st.spinner("Calculating cohort metrics…"):
            result = cohort_retention_analysis(df, cust_col, date_col, period, rev_col)
        st.session_state.cohort_result = result

    if 'cohort_result' not in st.session_state:
        return

    result = st.session_state.cohort_result
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cohorts", result.metadata['total_cohorts'])
    c2.metric("Avg Cohort Size", f"{result.metadata['avg_cohort_size']:.0f}")
    c3.metric("Total Customers", f"{result.metadata['total_customers']:,}")

    st.markdown("### 📊 Retention Heatmap (%)")
    try:
        fig = plot_cohort_heatmap(result.retention_matrix, title="Cohort Retention Rates",
                                 value_format='percentage', interactive=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Visualization error: {e}")
        st.dataframe(result.retention_matrix.round(1), use_container_width=True)

    st.markdown("### 📉 Average Retention by Period")
    st.line_chart(result.avg_retention_by_period)

    if not result.revenue_matrix.empty:
        st.markdown("### 💰 Revenue by Cohort")
        st.dataframe(result.revenue_matrix.round(2), use_container_width=True)

    with st.expander("💡 Retention Interpretation", expanded=True):
        try:
            # FIX: Convert Series to dict for the interpreter (was causing TypeError)
            retention_dict = result.avg_retention_by_period.to_dict()
            interp = interpret_cohort_retention(
                retention_dict,
                result.metadata['total_cohorts'],
                result.metadata['total_customers']
            )
            st.markdown(f"### {interp.summary}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📊 Key Findings:**")
                for f in interp.key_findings:
                    st.markdown(f"- {f}")
            with c2:
                st.markdown("**✅ Recommendations:**")
                for r in interp.recommendations:
                    st.markdown(f"- {r}")
            for w in interp.warnings:
                st.warning(w)
        except Exception as e:
            st.warning(f"Could not generate interpretation: {e}")

        # AI enhancement (optional)
        if st.session_state.get('ai_backend', 'rule') != 'rule':
            if st.button("🤖 Get AI Interpretation", key="_ai_cohort_btn"):
                try:
                    from ai.interpreter import interpret_cohort_retention as ai_interpret_cohort
                    retention_dict = result.avg_retention_by_period.to_dict()
                    ai_kwargs = _get_ai_kwargs()
                    ai_result = ai_interpret_cohort(
                        retention_dict,
                        result.metadata['total_cohorts'],
                        **ai_kwargs,
                    )
                    st.markdown("---")
                    st.markdown(f"**🤖 AI Analysis** *(confidence: {ai_result.confidence})*")
                    st.info(ai_result.summary)
                    for finding in ai_result.key_findings:
                        st.markdown(f"- {finding}")
                    if ai_result.recommendations:
                        st.markdown("**Recommendations:**")
                        for rec in ai_result.recommendations:
                            st.markdown(f"- {rec}")
                except Exception as e:
                    st.error(f"AI interpretation failed: {e}")


# ──────────────────────────────────────────────────
#  CAMPAIGN TAB
# ──────────────────────────────────────────────────

def render_campaign_tab(df, mapping):
    st.subheader("🎯 Campaign Performance Tracking")

    date_col = mapping.get('date_column')
    if not date_col:
        st.warning("⚠️ Need date column. Please configure in sidebar.")
        return

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    metric_cols = st.multiselect(
        "Select Metrics to Track", numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
        key="_camp_metrics"
    )

    if not metric_cols:
        st.info("📊 Select at least one metric to analyze")
        return

    analysis_type = st.radio("Analysis Type",
                             ["Performance Over Time", "Year-over-Year Comparison"],
                             horizontal=True, key="_camp_type")

    if analysis_type == "Performance Over Time":
        if st.button("📊 Analyze Performance", use_container_width=True):
            with st.spinner("Generating charts…"):
                try:
                    df_viz = df.copy()
                    df_viz[date_col] = pd.to_datetime(df_viz[date_col], errors='coerce')
                    df_viz = df_viz.dropna(subset=[date_col]).sort_values(date_col)
                    df_agg = df_viz.groupby(date_col)[metric_cols].sum().reset_index()

                    fig = px.line(df_agg, x=date_col, y=metric_cols,
                                 title="Campaign Performance Over Time",
                                 template='plotly_white')
                    fig.update_layout(height=500, hovermode='x unified')
                    fig.update_traces(line=dict(width=2.5))
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 📊 Summary Statistics")
                    try:
                        summary = campaign_performance_summary(df_agg, date_col, metric_cols)
                        st.dataframe(summary, use_container_width=True)
                    except Exception as e:
                        # Fallback: basic describe
                        st.dataframe(df_agg[metric_cols].describe().round(2), use_container_width=True)

                    st.markdown("---")
                    with st.expander("💡 Trend Analysis", expanded=True):
                        for metric in metric_cols:
                            first_half = df_agg[metric].head(len(df_agg)//2).mean()
                            second_half = df_agg[metric].tail(len(df_agg)//2).mean()
                            if first_half == 0:
                                st.info(f"{metric}: Insufficient data for trend")
                                continue
                            change = ((second_half - first_half) / first_half * 100)
                            if change > 10:
                                st.success(f"{metric}: 📈 Increasing (+{change:.1f}%)")
                            elif change < -10:
                                st.warning(f"{metric}: 📉 Decreasing ({change:.1f}%)")
                            else:
                                st.info(f"{metric}: 📊 Stable ({change:+.1f}%)")

                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    elif analysis_type == "Year-over-Year Comparison":
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=[date_col])
        years = sorted(df_temp[date_col].dt.year.unique(), reverse=True)

        if len(years) < 2:
            st.warning("⚠️ Need data from at least 2 years for YoY comparison")
            return

        c1, c2 = st.columns(2)
        with c1:
            current_year = st.selectbox("Current Year", years, index=0, key="_yoy_cur")
        with c2:
            available_previous = [y for y in years if y != current_year]
            previous_year = st.selectbox("Compare to Year", available_previous, key="_yoy_prev")

        if st.button("📊 Compare Years", use_container_width=True):
            with st.spinner("Calculating year-over-year changes…"):
                try:
                    comparison = year_over_year_comparison(
                        df_temp, date_col, metric_cols,
                        current_year, previous_year, period='M'
                    )

                    st.markdown("### 📈 Year-over-Year Performance")
                    change_cols = [c for c in comparison.columns if '_change_pct' in c]
                    if change_cols:
                        cols = st.columns(len(change_cols))
                        for i, col in enumerate(change_cols):
                            metric_name = col.replace('_change_pct', '').replace('_', ' ').title()
                            avg_change = comparison[col].mean()
                            cols[i].metric(f"{metric_name}", f"{avg_change:+.1f}%")

                    st.dataframe(comparison, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in YoY comparison: {e}")


# ──────────────────────────────────────────────────
#  FUNNEL TAB
# ──────────────────────────────────────────────────

def render_funnel_tab(df, mapping):
    st.subheader("🔀 Conversion Funnel Analysis")

    st.markdown("""
    **How it works:** Select columns that represent funnel stages (binary: 1 = completed, 0 = not completed).
    The analysis calculates conversion and drop-off rates between each stage.
    """)

    # Detect candidate columns: binary or boolean columns
    binary_cols = []
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, True, False, '0', '1', 'yes', 'no', 'Yes', 'No'}):
            binary_cols.append(col)

    # Also offer numeric columns that could represent counts at each stage
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    mode = st.radio("Funnel Mode", ["Binary Stage Columns", "Manual Stage Values"],
                    horizontal=True, key="_funnel_mode",
                    help="Binary: columns with 0/1 values per row. Manual: enter aggregate counts per stage.")

    if mode == "Binary Stage Columns":
        if not binary_cols:
            st.info("No binary (0/1) columns detected. Use 'Manual Stage Values' mode, "
                    "or ensure your data has columns where 1 = completed the stage.")
            return

        stage_cols = st.multiselect(
            "Select stage columns (in funnel order, top→bottom)",
            binary_cols, default=binary_cols[:min(5, len(binary_cols))],
            key="_funnel_stages"
        )

        if len(stage_cols) < 2:
            st.warning("Select at least 2 stage columns to build a funnel.")
            return

        stage_names = []
        with st.expander("Rename stages (optional)"):
            for col in stage_cols:
                name = st.text_input(f"Name for '{col}'", value=col, key=f"_fn_{col}")
                stage_names.append(name)

        # Optional: compare by segment
        cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                    if 2 <= df[c].nunique() <= 10]
        segment_col = st.selectbox("Compare by segment (optional)",
                                   ["(none)"] + cat_cols, key="_funnel_seg")

        if st.button("🔍 Analyze Funnel", use_container_width=True):
            with st.spinner("Calculating funnel metrics…"):
                # Ensure binary
                df_funnel = df[stage_cols].copy()
                for col in stage_cols:
                    df_funnel[col] = pd.to_numeric(df_funnel[col], errors='coerce').fillna(0).astype(int).clip(0, 1)

                result = analyze_funnel(df_funnel, stage_cols, stage_names)
                st.session_state.funnel_result = result

                # Segment comparison
                if segment_col != "(none)":
                    df_seg = df[stage_cols + [segment_col]].copy()
                    for col in stage_cols:
                        df_seg[col] = pd.to_numeric(df_seg[col], errors='coerce').fillna(0).astype(int).clip(0, 1)
                    seg_results = analyze_funnel_by_cohort(df_seg, stage_cols, segment_col, stage_names)
                    st.session_state.funnel_segments = seg_results
                else:
                    st.session_state.funnel_segments = None

    else:  # Manual Stage Values
        st.markdown("Enter stage names and their aggregate values (e.g. visitor counts at each step):")
        num_stages = st.number_input("Number of stages", 2, 10, 4, key="_fn_num")
        stage_data = {}
        for i in range(int(num_stages)):
            c1, c2 = st.columns([2, 1])
            with c1:
                name = st.text_input(f"Stage {i+1} name", value=f"Stage {i+1}", key=f"_fn_name_{i}")
            with c2:
                val = st.number_input(f"Count", min_value=0, value=max(1000 - i * 200, 10), key=f"_fn_val_{i}")
            stage_data[name] = val

        if st.button("🔍 Visualize Funnel", use_container_width=True):
            st.session_state.funnel_manual = stage_data
            st.session_state.funnel_result = None
            st.session_state.funnel_segments = None

    # ── Render results ──

    if 'funnel_manual' in st.session_state and st.session_state.get('funnel_manual'):
        stage_data = st.session_state.funnel_manual
        st.markdown("---")

        # Metrics
        stages = list(stage_data.keys())
        values = list(stage_data.values())
        overall = (values[-1] / values[0] * 100) if values[0] > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Entering", f"{values[0]:,}")
        c2.metric("Final Stage", f"{values[-1]:,}")
        c3.metric("Overall Conversion", f"{overall:.1f}%")

        try:
            fig = plot_conversion_funnel(stage_data, title="Conversion Funnel")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Visualization error: {e}")

        # Drop-off table
        rows = []
        for i in range(len(stages)):
            prev = values[i - 1] if i > 0 else values[0]
            conv = (values[i] / prev * 100) if prev > 0 else 0
            drop = 100 - conv if i > 0 else 0
            rows.append({'Stage': stages[i], 'Count': values[i],
                         'Conv %': f"{conv:.1f}%", 'Drop-off %': f"{drop:.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if 'funnel_result' in st.session_state and st.session_state.funnel_result:
        result = st.session_state.funnel_result
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Users", f"{result.metadata['total_users']:,}")
        c2.metric("Final Conversions", f"{result.metadata['final_conversions']:,}")
        c3.metric("Overall Conversion", f"{result.overall_conversion:.1f}%")

        # Funnel chart
        stage_dict = dict(zip(result.stages, result.counts))
        try:
            fig = plot_conversion_funnel(stage_dict, title="Conversion Funnel")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Visualization error: {e}")

        # Detail table
        detail = pd.DataFrame({
            'Stage': result.stages,
            'Users': result.counts,
            'Conversion %': [f"{r:.1f}%" for r in result.conversion_rates],
            'Drop-off %': [f"{r:.1f}%" for r in result.drop_off_rates],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

        # Bottlenecks
        bottlenecks = identify_bottlenecks(result, threshold=30.0)
        if bottlenecks:
            st.markdown("### 🚧 Bottlenecks (>30% drop-off)")
            for bn in bottlenecks:
                st.warning(f"**{bn['stage']}**: {bn['drop_off_rate']:.1f}% drop-off "
                           f"({bn['users_lost']:,} users lost)")
        else:
            st.success("No major bottlenecks detected (all stages <30% drop-off)")

        # Segment comparison
        seg_results = st.session_state.get('funnel_segments')
        if seg_results:
            st.markdown("### 📊 Funnel by Segment")
            seg_rows = []
            for seg_name, seg_result in seg_results.items():
                seg_rows.append({
                    'Segment': seg_name,
                    'Users': seg_result.metadata['total_users'],
                    'Conversions': seg_result.metadata['final_conversions'],
                    'Overall %': f"{seg_result.overall_conversion:.1f}%"
                })
            st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

            # Side-by-side chart
            seg_data = {}
            for seg_name, seg_result in seg_results.items():
                seg_data[seg_name] = dict(zip(seg_result.stages, seg_result.conversion_rates))
            seg_df = pd.DataFrame(seg_data)
            seg_df.index.name = 'Stage'
            st.bar_chart(seg_df)


# ──────────────────────────────────────────────────
#  ATTRIBUTION TAB
# ──────────────────────────────────────────────────

def render_attribution_tab(df, mapping):
    st.subheader("📡 Marketing Attribution")

    st.markdown("""
    **How it works:** Assign conversion credit to marketing channels using different models.
    Your data needs: customer ID, channel/source, conversion flag/value, and a date column.
    """)

    cust_col = mapping.get('customer_id')
    date_col = mapping.get('date_column')

    # Let user pick channel and conversion columns
    all_cols = list(df.columns)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not cust_col:
        st.warning("⚠️ Please select a Customer ID column in the sidebar first.")
        return
    if not date_col:
        st.warning("⚠️ Please select a Date column in the sidebar first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        channel_col = st.selectbox("📢 Channel / Source column",
                                   cat_cols if cat_cols else all_cols,
                                   key="_attr_channel")
    with c2:
        conversion_col = st.selectbox("✅ Conversion column (numeric: 1/0 or value)",
                                      num_cols if num_cols else all_cols,
                                      key="_attr_conv")

    # Check data viability
    n_channels = df[channel_col].nunique() if channel_col else 0
    n_conversions = (df[conversion_col] > 0).sum() if conversion_col else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Channels", n_channels)
    c2.metric("Conversions", f"{n_conversions:,}")
    c3.metric("Customers", f"{df[cust_col].nunique():,}" if cust_col else "—")

    if n_channels < 2:
        st.warning("Need at least 2 unique channels for attribution analysis.")
        return
    if n_conversions == 0:
        st.warning("No conversions found (all values ≤ 0). Check your conversion column.")
        return

    st.markdown("---")

    model_choice = st.radio("Attribution Model",
                            ["Compare All Models", "Last Touch", "First Touch",
                             "Linear", "Time Decay", "Position Based"],
                            horizontal=True, key="_attr_model")

    lookback = st.slider("Lookback window (days)", 7, 90, 30, key="_attr_lookback")

    if st.button("🔍 Run Attribution", use_container_width=True, type="primary"):
        with st.spinner("Calculating attribution…"):
            try:
                if model_choice == "Compare All Models":
                    comparison = compare_attribution_models(
                        df, cust_col, channel_col, conversion_col, date_col, lookback
                    )
                    st.session_state.attr_result = ('comparison', comparison)
                else:
                    model_map = {
                        "Last Touch": last_touch_attribution,
                        "First Touch": first_touch_attribution,
                        "Linear": lambda *a: linear_attribution(*a, lookback_days=lookback),
                        "Time Decay": lambda *a: time_decay_attribution(*a, lookback_days=lookback),
                        "Position Based": lambda *a: position_based_attribution(*a, lookback_days=lookback),
                    }
                    func = model_map[model_choice]
                    result = func(df, cust_col, channel_col, conversion_col, date_col)
                    st.session_state.attr_result = ('single', result)
            except Exception as e:
                st.error(f"Attribution error: {e}")
                import traceback
                st.code(traceback.format_exc())

    if 'attr_result' not in st.session_state:
        return

    mode, data = st.session_state.attr_result
    st.markdown("---")

    if mode == 'comparison':
        st.markdown("### 📊 Model Comparison")
        st.dataframe(data.round(2), use_container_width=True, hide_index=True)

        # Visual comparison
        model_cols = [c for c in data.columns if c != 'channel']
        if model_cols:
            chart_df = data.set_index('channel')[model_cols]
            st.bar_chart(chart_df)

            st.markdown("### 💡 Interpretation")
            # Find channel with biggest variance across models
            variances = chart_df.var(axis=1).sort_values(ascending=False)
            if len(variances) > 0:
                most_disputed = variances.index[0]
                vals = chart_df.loc[most_disputed]
                st.info(f"**{most_disputed}** shows the biggest difference across models "
                        f"(range: {vals.min():.1f} – {vals.max():.1f}). "
                        f"This channel's true impact is model-dependent — consider your business context.")

    elif mode == 'single':
        result = data
        st.markdown(f"### {result.model_name} Attribution")
        st.caption(result.methodology)

        c1, c2 = st.columns(2)
        c1.metric("Total Conversions Attributed", f"{result.total_conversions:,.1f}")
        c2.metric("Channels", len(result.channel_attribution))

        st.dataframe(result.channel_attribution, use_container_width=True, hide_index=True)

        # Waterfall chart
        if not result.channel_attribution.empty:
            try:
                fig = plot_attribution_waterfall(
                    result.channel_attribution,
                    channel_col='channel',
                    value_col='attributed_conversions',
                    title=f"{result.model_name} Attribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                # Fallback: bar chart
                try:
                    st.bar_chart(result.channel_attribution.set_index('channel')['attributed_conversions'])
                except Exception:
                    st.warning(f"Visualization error: {e}")

        # Top channel insight
        if len(result.channel_attribution) > 0:
            top = result.channel_attribution.iloc[0]
            st.info(f"**Top channel:** {top['channel']} — "
                    f"{top['attribution_percentage']:.1f}% of all attributed conversions")


# ──────────────────────────────────────────────────
#  ANOMALY DETECTION TAB
# ──────────────────────────────────────────────────

def render_anomaly_tab(df, mapping):
    st.subheader("🚨 Anomaly Detection")
    st.info("💡 Finds unusual patterns — spikes, drops, and odd combinations.")

    c1, c2 = st.columns(2)
    with c1:
        sensitivity = st.select_slider(
            "Detection Sensitivity",
            options=['low', 'medium', 'high'],
            value='medium',
            help="Low = Only critical | Medium = Balanced | High = Catch more"
        )
    with c2:
        anomaly_types = st.multiselect(
            "Anomaly Types",
            options=['Value Anomalies', 'Pattern Anomalies', 'Sequence Anomalies'],
            default=['Value Anomalies', 'Sequence Anomalies'],
        )

    date_col = mapping.get('date_column')

    if st.button("🔍 Detect Anomalies", use_container_width=True, type="primary"):
        h = df_hash(df)
        all_anomalies = cached_anomalies(h, df, date_col, sensitivity.lower())

        combined = []
        if 'Value Anomalies' in anomaly_types:
            combined.extend(all_anomalies['value'])
        if 'Pattern Anomalies' in anomaly_types:
            combined.extend(all_anomalies['pattern'])
        if 'Sequence Anomalies' in anomaly_types and date_col:
            combined.extend(all_anomalies['sequence'])

        # FIX: Deduplicate — group by (type, title), keep highest severity
        unique = {}
        severity_rank = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        for a in combined:
            key = (a.anomaly_type, a.title)
            if key not in unique or severity_rank.get(a.severity, 9) < severity_rank.get(unique[key].severity, 9):
                unique[key] = a

        prioritized = prioritize_anomalies(list(unique.values()))
        st.session_state.anomalies = prioritized
        st.session_state.anomaly_summary = {
            'total': len(prioritized),
            'critical': sum(1 for a in prioritized if a.severity == 'critical'),
            'high': sum(1 for a in prioritized if a.severity == 'high'),
            'medium': sum(1 for a in prioritized if a.severity == 'medium'),
            'low': sum(1 for a in prioritized if a.severity == 'low'),
        }

    if 'anomalies' in st.session_state and st.session_state.anomalies:
        anomalies = st.session_state.anomalies
        summary = st.session_state.anomaly_summary

        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", summary['total'])
        c2.metric("🔴 Critical", summary['critical'])
        c3.metric("🟠 High", summary['high'])
        c4.metric("🟡 Medium", summary['medium'])
        c5.metric("⚪ Low", summary['low'])

        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['critical', 'high', 'medium', 'low'],
            default=['critical', 'high', 'medium'],
            key="_anom_filter"
        )

        filtered = [a for a in anomalies if a.severity in severity_filter]

        if not filtered:
            st.info("No anomalies match your filter.")
        else:
            st.caption(f"Showing {min(len(filtered), 20)} of {len(filtered)} anomalies")

            for i, anomaly in enumerate(filtered[:20], 1):
                icons = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '⚪'}
                badges = {'value': '📊 VALUE', 'pattern': '🔀 PATTERN', 'sequence': '📈 SEQUENCE'}

                with st.expander(
                    f"{icons.get(anomaly.severity, '⚪')} **{i}. {anomaly.title}** | "
                    f"{badges.get(anomaly.anomaly_type, '')} | {anomaly.severity.upper()}",
                    expanded=(i <= 3)
                ):
                    st.markdown(anomaly.description)

                    # FIX: Wrap numeric conversion in try/except for non-numeric expected_value
                    try:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Detected", f"{float(anomaly.detected_value):.2f}")
                        c2.metric("Expected", f"{float(anomaly.expected_value):.2f}")
                        c3.metric("Deviation", f"{anomaly.deviation_pct:.0f}%")
                    except (TypeError, ValueError):
                        pass

                    st.warning(anomaly.business_impact)
                    for rec in anomaly.recommendations:
                        st.markdown(f"- {rec}")

        # AI interpretation of anomaly results
        top_items = "\n".join(
            f"- [{a.severity.upper()}] {a.title}: {a.description[:120]}"
            for a in anomalies[:8]
        )
        _ai_interpret(
            system=(
                "You are a data analyst. Summarize anomaly detection results clearly "
                "for a business audience. Highlight the most critical issues and suggest "
                "1-2 priority actions. Be factual — only use the data provided. "
                "Keep your response under 150 words."
            ),
            user=(
                f"Anomaly scan results: {summary['total']} anomalies found "
                f"(Critical: {summary['critical']}, High: {summary['high']}, "
                f"Medium: {summary['medium']}, Low: {summary['low']})\n\n"
                f"Top anomalies:\n{top_items}\n\n"
                "Summarize key risks and suggest 2 priority actions."
            ),
            key="ai_anomaly",
            spinner_text="Analyzing anomalies…",
        )

    elif 'anomalies' in st.session_state:
        st.success("✅ No significant anomalies detected!")


# ──────────────────────────────────────────────────
#  FULL SCAN TAB
# ──────────────────────────────────────────────────

def render_full_scan_tab(df, mapping):
    st.subheader("🔎 Full Scan — Automated Insight Discovery")
    st.markdown("""
    One-click analysis that scans your entire dataset for **positive trends, negative trends,
    unusual relationships, anomalies, and data quality issues** — all at once.
    """)

    if st.button("🚀 Run Full Scan", use_container_width=True, type="primary"):
        findings = []
        progress = st.progress(0, text="Scanning…")

        # ── 1. DATA QUALITY ──
        progress.progress(5, text="Checking data quality…")
        try:
            quality_level, warnings = check_data_quality(df)
            for w in warnings:
                findings.append({
                    'category': '🧹 Data Quality',
                    'severity': 'high' if quality_level in ('critical', 'poor') else 'medium',
                    'direction': 'neutral',
                    'title': w,
                    'detail': '',
                })
        except Exception:
            pass

        # ── 2. TREND DETECTION (numeric columns over time) ──
        progress.progress(15, text="Detecting trends…")
        date_col = mapping.get('date_column')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if date_col and date_col in df.columns and numeric_cols:
            try:
                df_sorted = df.copy()
                df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')
                df_sorted = df_sorted.dropna(subset=[date_col]).sort_values(date_col)

                for col in numeric_cols[:10]:
                    series = df_sorted[col].dropna()
                    if len(series) < 10:
                        continue

                    # Split into halves and compare
                    mid = len(series) // 2
                    first_half = series.iloc[:mid].mean()
                    second_half = series.iloc[mid:].mean()

                    if first_half == 0:
                        continue
                    pct_change = ((second_half - first_half) / abs(first_half)) * 100

                    if abs(pct_change) < 5:
                        continue

                    direction = 'positive' if pct_change > 0 else 'negative'
                    rev_col = mapping.get('revenue_column', '')
                    # Flip interpretation for cost-like columns
                    is_cost = any(kw in col.lower() for kw in ['cost', 'expense', 'discount', 'refund', 'churn', 'cancel'])
                    if is_cost:
                        direction = 'negative' if pct_change > 0 else 'positive'

                    severity = 'high' if abs(pct_change) > 25 else 'medium' if abs(pct_change) > 10 else 'low'

                    findings.append({
                        'category': '📈 Trend' if pct_change > 0 else '📉 Trend',
                        'severity': severity,
                        'direction': direction,
                        'title': f"{col}: {'↑' if pct_change > 0 else '↓'} {abs(pct_change):.1f}% (first vs second half)",
                        'detail': f"First half avg: {first_half:,.2f} → Second half avg: {second_half:,.2f}",
                    })
            except Exception:
                pass

        # ── 3. DISTRIBUTION ODDITIES ──
        progress.progress(30, text="Checking distributions…")
        for col in numeric_cols[:12]:
            series = df[col].dropna()
            if len(series) < 20:
                continue
            try:
                skewness = series.skew()
                kurtosis = series.kurtosis()

                if abs(skewness) > 3:
                    findings.append({
                        'category': '📊 Distribution',
                        'severity': 'medium',
                        'direction': 'neutral',
                        'title': f"{col}: Highly skewed distribution (skew={skewness:.1f})",
                        'detail': f"Strong {'right' if skewness > 0 else 'left'} skew — mean and median diverge significantly. "
                                  f"Mean: {series.mean():,.2f}, Median: {series.median():,.2f}",
                    })

                # Extreme outliers (>3 IQR)
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    extreme_low = (series < q1 - 3 * iqr).sum()
                    extreme_high = (series > q3 + 3 * iqr).sum()
                    total_extreme = extreme_low + extreme_high
                    pct = total_extreme / len(series) * 100

                    if total_extreme > 0 and pct > 0.5:
                        findings.append({
                            'category': '⚠️ Outliers',
                            'severity': 'medium' if pct < 5 else 'high',
                            'direction': 'neutral',
                            'title': f"{col}: {total_extreme} extreme outliers ({pct:.1f}% of data)",
                            'detail': f"Range: [{series.min():,.2f} – {series.max():,.2f}], "
                                      f"IQR: [{q1:,.2f} – {q3:,.2f}]",
                        })

                # Concentration: top value dominance
                if series.nunique() < len(series) * 0.5:
                    top_val_pct = series.value_counts(normalize=True).iloc[0] * 100
                    if top_val_pct > 50:
                        top_val = series.value_counts().index[0]
                        findings.append({
                            'category': '🏗️ Concentration',
                            'severity': 'medium',
                            'direction': 'neutral',
                            'title': f"{col}: Single value dominates ({top_val_pct:.0f}% = {top_val})",
                            'detail': "Low variance — this column may not be analytically useful.",
                        })
            except Exception:
                pass

        # ── 4. UNUSUAL CORRELATIONS ──
        progress.progress(50, text="Finding correlations…")
        if len(numeric_cols) >= 2:
            try:
                corr_cols = numeric_cols[:15]
                corr_matrix = df[corr_cols].corr(method='spearman')

                seen = set()
                for i, c1 in enumerate(corr_cols):
                    for j, c2 in enumerate(corr_cols):
                        if i >= j:
                            continue
                        r = corr_matrix.loc[c1, c2]
                        key = tuple(sorted([c1, c2]))
                        if key in seen or np.isnan(r):
                            continue
                        seen.add(key)

                        if abs(r) > 0.85:
                            findings.append({
                                'category': '🔗 Correlation',
                                'severity': 'medium',
                                'direction': 'positive' if r > 0 else 'neutral',
                                'title': f"Strong {'positive' if r > 0 else 'negative'} correlation: {c1} ↔ {c2} (ρ={r:.2f})",
                                'detail': "These columns move together strongly — possible redundancy or causal link.",
                            })
                        elif abs(r) > 0.6:
                            findings.append({
                                'category': '🔗 Correlation',
                                'severity': 'low',
                                'direction': 'neutral',
                                'title': f"Moderate {'positive' if r > 0 else 'negative'} correlation: {c1} ↔ {c2} (ρ={r:.2f})",
                                'detail': "Notable relationship worth investigating.",
                            })
            except Exception:
                pass

        # ── 5. CATEGORICAL INSIGHTS ──
        progress.progress(65, text="Scanning categories…")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        rev_col = mapping.get('revenue_column')

        for col in cat_cols[:5]:
            try:
                vc = df[col].value_counts()
                if len(vc) < 2 or len(vc) > 50:
                    continue

                # Pareto: does top 20% account for >80% of rows?
                top_n = max(1, int(len(vc) * 0.2))
                top_pct = vc.iloc[:top_n].sum() / vc.sum() * 100
                if top_pct > 80:
                    findings.append({
                        'category': '🏗️ Concentration',
                        'severity': 'low',
                        'direction': 'neutral',
                        'title': f"{col}: Pareto pattern — top {top_n} of {len(vc)} values = {top_pct:.0f}% of data",
                        'detail': f"Top: {', '.join(vc.index[:3].astype(str).tolist())}",
                    })

                # Revenue disparity across categories
                if rev_col and rev_col in df.columns and len(vc) >= 2:
                    group_means = df.groupby(col)[rev_col].mean().dropna()
                    if len(group_means) >= 2:
                        ratio = group_means.max() / group_means.min() if group_means.min() > 0 else 0
                        if ratio > 5:
                            best = group_means.idxmax()
                            worst = group_means.idxmin()
                            findings.append({
                                'category': '💰 Revenue Disparity',
                                'severity': 'high' if ratio > 10 else 'medium',
                                'direction': 'neutral',
                                'title': f"{col}: {ratio:.1f}× revenue gap between '{best}' and '{worst}'",
                                'detail': f"'{best}': avg {group_means.max():,.2f} vs '{worst}': avg {group_means.min():,.2f}",
                            })
            except Exception:
                pass

        # ── 6. ANOMALY ENGINE (reuse existing) ──
        progress.progress(80, text="Detecting anomalies…")
        try:
            from analysis_engine import detect_all_anomalies
            h = df_hash(df)
            anomalies = cached_detect_anomalies(h, df)

            unique = {}
            severity_rank = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            for a in anomalies:
                key = (a.anomaly_type, a.title)
                if key not in unique or severity_rank.get(a.severity, 9) < severity_rank.get(unique[key].severity, 9):
                    unique[key] = a

            for a in list(unique.values())[:15]:
                findings.append({
                    'category': f"🚨 {a.anomaly_type.replace('_', ' ').title()}",
                    'severity': a.severity,
                    'direction': 'negative',
                    'title': a.title,
                    'detail': a.business_impact if a.business_impact else '',
                })
        except Exception:
            pass

        # ── 7. MISSING DATA PATTERNS ──
        progress.progress(90, text="Checking missing data…")
        try:
            missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            high_missing = missing_pct[missing_pct > 10]
            for col_name, pct in high_missing.items():
                findings.append({
                    'category': '🕳️ Missing Data',
                    'severity': 'high' if pct > 50 else 'medium',
                    'direction': 'negative',
                    'title': f"{col_name}: {pct:.1f}% missing values",
                    'detail': f"{int(df[col_name].isnull().sum()):,} of {len(df):,} rows are null.",
                })
        except Exception:
            pass

        progress.progress(100, text="Scan complete!")
        st.session_state.full_scan = findings

    # ── RENDER FINDINGS ──
    if 'full_scan' not in st.session_state:
        return

    findings = st.session_state.full_scan

    if not findings:
        st.success("✅ No significant findings! Your data looks clean and stable.")
        return

    st.markdown("---")

    # Summary metrics
    n_positive = sum(1 for f in findings if f['direction'] == 'positive')
    n_negative = sum(1 for f in findings if f['direction'] == 'negative')
    n_neutral = sum(1 for f in findings if f['direction'] == 'neutral')
    n_high = sum(1 for f in findings if f['severity'] in ('high', 'critical'))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Findings", len(findings))
    c2.metric("🟢 Positive", n_positive)
    c3.metric("🔴 Negative / Risks", n_negative)
    c4.metric("🔶 High Severity", n_high)

    # Filter controls
    st.markdown("---")
    fc1, fc2 = st.columns(2)
    with fc1:
        show_dir = st.multiselect("Direction", ['positive', 'negative', 'neutral'],
                                  default=['positive', 'negative', 'neutral'], key="_fs_dir")
    with fc2:
        show_sev = st.multiselect("Severity", ['critical', 'high', 'medium', 'low'],
                                  default=['critical', 'high', 'medium', 'low'], key="_fs_sev")

    filtered = [f for f in findings if f['direction'] in show_dir and f['severity'] in show_sev]

    # Sort: high severity first, then by direction (negative first)
    sev_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    dir_order = {'negative': 0, 'neutral': 1, 'positive': 2}
    filtered.sort(key=lambda f: (sev_order.get(f['severity'], 9), dir_order.get(f['direction'], 9)))

    # Group by category
    from collections import OrderedDict
    grouped = OrderedDict()
    for f in filtered:
        cat = f['category']
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(f)

    for category, items in grouped.items():
        st.markdown(f"### {category}")
        for item in items:
            sev = item['severity']
            direction = item['direction']

            # Color coding
            if direction == 'positive':
                icon = "🟢"
            elif direction == 'negative':
                icon = "🔴"
            else:
                icon = "🟡"

            sev_badge = f"**`{sev.upper()}`**" if sev in ('critical', 'high') else f"`{sev}`"

            st.markdown(f"{icon} {sev_badge} — **{item['title']}**")
            if item['detail']:
                st.caption(item['detail'])

        st.markdown("")  # spacing

    # Export findings as table
    with st.expander("📋 Export findings as table"):
        findings_df = pd.DataFrame(filtered)[['category', 'severity', 'direction', 'title', 'detail']]
        st.dataframe(findings_df, use_container_width=True, hide_index=True)

        csv_buf = findings_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download as CSV", csv_buf, "full_scan_findings.csv", "text/csv")

    # AI summary of full scan
    high_items = [f for f in findings if f['severity'] in ('critical', 'high')][:10]
    pos_items  = [f for f in findings if f['direction'] == 'positive'][:5]

    high_text = "\n".join(f"- [{f['severity'].upper()}] {f['title']}" for f in high_items) or "None"
    pos_text  = "\n".join(f"- {f['title']}" for f in pos_items) or "None"

    _ai_interpret(
        system=(
            "You are a data analyst. Summarize an automated dataset scan for a business audience "
            "in 3-5 bullet points. Group by: key risks, key opportunities, and data quality notes. "
            "Be factual — only use the data provided. Keep under 200 words."
        ),
        user=(
            f"Full dataset scan: {len(findings)} findings "
            f"(Positive: {n_positive}, Negative/Risks: {n_negative}, High-severity: {n_high})\n\n"
            f"High-severity issues:\n{high_text}\n\n"
            f"Positive findings:\n{pos_text}\n\n"
            "Provide a 3-5 bullet summary covering key risks, opportunities, and data quality issues."
        ),
        key="ai_full_scan",
        spinner_text="Summarizing scan results…",
    )



# ──────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
