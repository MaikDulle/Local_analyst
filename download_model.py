#!/usr/bin/env python3
"""
Download a small GGUF model for Local Analyst's AI interpretation.
Models are saved to the models/ folder and auto-detected by the app.

Usage:
    python download_model.py              # interactive menu
    python download_model.py --model 1   # download model #1 directly
"""

import sys
import os
import argparse
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# Available models — ordered by size (smallest first)
AVAILABLE_MODELS = [
    {
        "name": "Qwen 2.5 · 1.5B Instruct · Q4_K_M",
        "description": "~1.0 GB  |  Fast on CPU  |  Good analytical writing  ✓ Recommended",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.0,
    },
    {
        "name": "Llama 3.2 · 1B Instruct · Q4_K_M",
        "description": "~0.8 GB  |  Fastest on CPU  |  Concise outputs",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.8,
    },
    {
        "name": "Phi-3 Mini · 4K Instruct · Q4",
        "description": "~2.2 GB  |  Best quality  |  Slower on CPU",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "size_gb": 2.2,
    },
]


def download(model: dict, dest_dir: Path) -> Path:
    """Download a model file with a progress bar. Resumes partial downloads."""
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' not installed. Run: pip install requests")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / model["filename"]

    # Resume support
    resume_byte = dest_path.stat().st_size if dest_path.exists() else 0
    headers = {"Range": f"bytes={resume_byte}-"} if resume_byte else {}

    print(f"\nDownloading: {model['name']}")
    print(f"  → {dest_path}")
    print(f"  Source: {model['url']}\n")

    with requests.get(model["url"], headers=headers, stream=True, timeout=30) as r:
        if r.status_code == 416:
            # Already fully downloaded
            print("File already complete.")
            return dest_path

        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0)) + resume_byte
        downloaded = resume_byte
        mode = "ab" if resume_byte else "wb"
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = int(pct / 2)
                        mb = downloaded / (1024 ** 2)
                        total_mb = total / (1024 ** 2)
                        print(
                            f"\r  [{'█' * bar}{'░' * (50 - bar)}] "
                            f"{pct:.1f}%  {mb:.0f}/{total_mb:.0f} MB",
                            end="", flush=True
                        )

    print(f"\n\n✓ Saved to: {dest_path}")
    return dest_path


def check_llama_cpp() -> bool:
    try:
        import llama_cpp  # noqa
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Download a GGUF model for Local Analyst")
    parser.add_argument("--model", type=int, help="Model number to download (1-based)")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    print("=" * 60)
    print("  Local Analyst — Model Downloader")
    print("=" * 60)

    if not check_llama_cpp():
        print("\nWARNING: llama-cpp-python is not installed.")
        print("   Install the pre-built wheel (no C++ compiler needed):\n")
        print("     pip install llama-cpp-python --only-binary=llama-cpp-python \\")
        print("       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n")
        print("   Then re-run this script to download a model.")
        print()

    print("\nAvailable models:\n")
    for i, m in enumerate(AVAILABLE_MODELS, 1):
        already = (MODELS_DIR / m["filename"]).exists()
        status = " ✓ already downloaded" if already else ""
        print(f"  [{i}] {m['name']}{status}")
        print(f"      {m['description']}")
        print()

    if args.list:
        return

    if args.model:
        choice = args.model
    else:
        try:
            choice = int(input("Enter model number to download [1]: ").strip() or "1")
        except (ValueError, EOFError):
            choice = 1

    if choice < 1 or choice > len(AVAILABLE_MODELS):
        print(f"Invalid choice. Pick 1–{len(AVAILABLE_MODELS)}.")
        sys.exit(1)

    model = AVAILABLE_MODELS[choice - 1]
    dest = MODELS_DIR / model["filename"]

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 ** 2)
        print(f"\nModel already exists ({size_mb:.0f} MB): {dest}")
        ans = input("Re-download? [y/N]: ").strip().lower()
        if ans != "y":
            print("Skipped.")
            return

    try:
        download(model, MODELS_DIR)
        print("\nDone! Open the app and go to sidebar → AI Settings → Local LLM.")
        print("The model will be auto-detected.\n")
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. Run again to resume.")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
