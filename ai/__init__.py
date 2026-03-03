"""
AI interpretation layer for Local Analyst.

Backends:
  rule   — rule-based, always available (default)
  local  — llama-cpp-python GGUF model (pip install llama-cpp-python)
  ollama — Ollama server (requires ollama to be running)

AI is used ONLY for human-readable interpretation, never for calculations.
"""

from .interpreter import InterpretationResult, interpret_ab_test, interpret_cohort_retention
from .insights import Insight, generate_insights_from_timeseries, generate_insights_from_segments
from .recommendations import Recommendation, generate_ab_test_recommendations
from .local_llm import is_available as local_llm_available, install_instructions

__all__ = [
    "InterpretationResult",
    "interpret_ab_test",
    "interpret_cohort_retention",
    "Insight",
    "generate_insights_from_timeseries",
    "generate_insights_from_segments",
    "Recommendation",
    "generate_ab_test_recommendations",
    "local_llm_available",
    "install_instructions",
]
