"""
Local LLM integration for Local Analyst.
Uses llama-cpp-python to run GGUF models directly — no Ollama server needed.

Install:
    pip install llama-cpp-python

Recommended small GGUF models (download from HuggingFace):
    - Phi-3-mini-4k-instruct Q4_K_M  (~2.2 GB)  best quality for analysis
      https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
    - TinyLlama-1.1B-Chat Q4_K_M     (~600 MB)  fastest, CPU-friendly
      https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
    - Qwen2.5-0.5B-Instruct Q4_K_M   (~400 MB)  smallest footprint
      https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF

Usage:
    from ai.local_llm import LocalLLM, is_available

    if is_available():
        llm = LocalLLM("path/to/model.gguf")
        text = llm.chat(
            system="You are a marketing analyst.",
            user="Explain this A/B test result: lift +12%, p=0.03"
        )
"""

from pathlib import Path
from typing import Optional, List

# Default models folder: <project_root>/models/
_MODELS_DIR = Path(__file__).parent.parent / "models"

# ── module-level singleton ──
_llm_instance: Optional["LocalLLM"] = None
_llm_model_path: Optional[str] = None


def find_local_models(models_dir: Path = _MODELS_DIR) -> List[Path]:
    """Return all .gguf files found in the models directory, sorted by name."""
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.gguf"))


def default_model_path() -> Optional[str]:
    """Return the path to the first available model, or None."""
    models = find_local_models()
    return str(models[0]) if models else None


def is_available() -> bool:
    """Return True if llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def install_instructions() -> str:
    return (
        "**Local LLM not installed.**\n\n"
        "Install the pre-built wheel (no C++ compiler needed):\n\n"
        "```\npip install llama-cpp-python "
        "--only-binary=llama-cpp-python "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n```"
    )


class LocalLLM:
    """
    Thin wrapper around llama-cpp-python for CPU-based local inference.
    Initialised lazily and reused as a singleton via `get_llm()`.
    """

    # Chat prompt templates keyed by model family (detected from filename)
    _TEMPLATES = {
        "phi":      "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
        "tinyllama": "<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n",
        "qwen":     "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        "llama":    "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]",
        "mistral":  "<s>[INST] {system}\n\n{user} [/INST]",
        "default":  "### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n",
    }

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = str(path)
        self._template_key = self._detect_template(path.name.lower())

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=42,        # fixed seed → reproducible outputs
            verbose=False,
        )

    def _detect_template(self, filename: str) -> str:
        for key in ("phi", "tinyllama", "qwen", "llama", "mistral"):
            if key in filename:
                return key
        return "default"

    def generate(self, prompt: str, max_tokens: int = 350, temperature: float = 0.1) -> str:
        """Raw completion — returns generated text only.

        temperature=0.1 keeps outputs near-deterministic and reduces hallucinations.
        seed=42 (set at init) ensures the same prompt always gives the same output.
        """
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "<|end|>", "<|im_end|>", "\n\n\n"],
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    def chat(self, system: str, user: str, max_tokens: int = 350) -> str:
        """Chat-style call using the model's instruct template."""
        template = self._TEMPLATES[self._template_key]
        prompt = template.format(system=system, user=user)
        return self.generate(prompt, max_tokens=max_tokens)


def get_llm(model_path: str) -> "LocalLLM":
    """
    Return the singleton LocalLLM, initialising it if the path changed.
    Raises ImportError / FileNotFoundError on misconfiguration.
    """
    global _llm_instance, _llm_model_path

    if _llm_instance is None or _llm_model_path != model_path:
        _llm_instance = LocalLLM(model_path)
        _llm_model_path = model_path

    return _llm_instance


def clear_llm():
    """Release the loaded model (frees RAM)."""
    global _llm_instance, _llm_model_path
    _llm_instance = None
    _llm_model_path = None


__all__ = [
    "LocalLLM",
    "is_available",
    "install_instructions",
    "get_llm",
    "clear_llm",
    "find_local_models",
    "default_model_path",
]
