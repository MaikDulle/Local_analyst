"""
AI interpretation module for Local Analyst.

Supports three backends:
  'rule'  — deterministic rule-based (always available, default)
  'local' — llama-cpp-python GGUF model (no server, pip install llama-cpp-python)
  'ollama'— Ollama server (requires Ollama to be running locally)

AI is used ONLY for interpretation, NEVER for calculations.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class InterpretationResult:
    """Result from AI interpretation."""
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    confidence: str
    raw_response: str


_SYSTEM_PROMPT = (
    "You are a concise marketing analytics assistant. "
    "Interpret analysis results in plain English for practitioners. "
    "Use bullet points. Be brief and actionable."
)


def _parse_llm_response(content: str) -> tuple:
    """Extract summary, findings, and recommendations from LLM output."""
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    summary = lines[0] if lines else "Analysis complete."
    bullets = [l for l in lines[1:] if l.startswith(('- ', '* ', '• ', '→'))]
    findings = bullets[:3] or lines[1:4]
    recs = bullets[3:6] or lines[4:7]
    return summary, findings or ["See full output below."], recs or ["Review the results above."]


def _ai_generate(prompt_user: str, model_path: Optional[str] = None,
                 ollama_model: str = 'llama3') -> Optional[str]:
    """
    Try local LLM first (if model_path given), then Ollama, return None on failure.
    """
    if model_path:
        try:
            from .local_llm import get_llm
            llm = get_llm(model_path)
            return llm.chat(system=_SYSTEM_PROMPT, user=prompt_user)
        except Exception as e:
            print(f"Local LLM failed: {e}")

    try:
        import ollama
        response = ollama.chat(
            model=ollama_model,
            messages=[
                {'role': 'system', 'content': _SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt_user},
            ]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Ollama failed: {e}")

    return None


def interpret_ab_test(
    variant_a_name: str,
    variant_b_name: str,
    lift_percentage: float,
    p_value: float,
    is_significant: bool,
    effect_size: str,
    sample_sizes: tuple,
    use_ai: bool = False,
    model_path: Optional[str] = None,
    ollama_model: str = 'llama3',
) -> InterpretationResult:
    """
    Interpret A/B test results.

    Args:
        variant_a_name: Control name
        variant_b_name: Treatment name
        lift_percentage: Lift percentage
        p_value: P-value
        is_significant: Statistical significance
        effect_size: Effect size interpretation
        sample_sizes: (n_a, n_b)
        use_ai: If True, try AI backends before rule-based fallback
        model_path: Path to GGUF model file (local LLM, no Ollama needed)
        ollama_model: Ollama model name (used only if use_ai=True and no model_path)

    Returns:
        InterpretationResult with interpretation
    """
    if use_ai:
        prompt = (
            f"Interpret this A/B test for a marketing team:\n"
            f"Control: {variant_a_name} | Treatment: {variant_b_name}\n"
            f"Lift: {lift_percentage:.2f}% | p-value: {p_value:.4f} | "
            f"Significant: {is_significant} | Effect size: {effect_size}\n"
            f"Samples: {sample_sizes[0]:,} vs {sample_sizes[1]:,}\n\n"
            f"Give: 1 summary sentence, 3 key findings, 3 recommendations."
        )
        raw = _ai_generate(prompt, model_path=model_path, ollama_model=ollama_model)
        if raw:
            summary, findings, recs = _parse_llm_response(raw)
            return InterpretationResult(
                summary=summary,
                key_findings=findings,
                recommendations=recs,
                confidence="AI-powered",
                raw_response=raw,
            )

    return _rule_based_ab_interpretation(
        variant_a_name, variant_b_name, lift_percentage,
        p_value, is_significant, effect_size, sample_sizes
    )


def _rule_based_ab_interpretation(
    variant_a_name: str,
    variant_b_name: str,
    lift: float,
    p_value: float,
    significant: bool,
    effect_size: str,
    sample_sizes: tuple
) -> InterpretationResult:
    """Rule-based A/B test interpretation (no AI)."""
    
    # Summary
    if significant and lift > 0:
        summary = f"{variant_b_name} shows a statistically significant improvement of {abs(lift):.1f}% over {variant_a_name}."
    elif significant and lift < 0:
        summary = f"{variant_b_name} shows a statistically significant decline of {abs(lift):.1f}% compared to {variant_a_name}."
    else:
        summary = f"No statistically significant difference detected between {variant_a_name} and {variant_b_name}."
    
    # Key findings
    findings = [
        f"Sample sizes: {sample_sizes[0]:,} ({variant_a_name}) vs {sample_sizes[1]:,} ({variant_b_name})",
        f"Effect size is {effect_size}",
        f"Statistical significance: p={p_value:.4f} ({'significant' if significant else 'not significant'})"
    ]
    
    # Recommendations
    if significant and lift > 5:
        recs = [
            f"Implement {variant_b_name} - strong positive impact",
            "Monitor metrics post-implementation to confirm sustained improvement",
            "Consider expanding test to other segments"
        ]
    elif significant and lift > 0:
        recs = [
            f"Consider implementing {variant_b_name} - modest but significant improvement",
            "Evaluate cost-benefit of implementation",
            "Test in production with gradual rollout"
        ]
    elif significant and lift < 0:
        recs = [
            f"Do NOT implement {variant_b_name} - negative impact detected",
            "Investigate why performance declined",
            "Test alternative variations"
        ]
    else:
        recs = [
            "Continue testing with larger sample size or longer duration",
            "Consider testing more distinct variations",
            "Evaluate if the metric is sensitive enough to detect meaningful change"
        ]
    
    return InterpretationResult(
        summary=summary,
        key_findings=findings,
        recommendations=recs,
        confidence="Rule-based",
        raw_response=""
    )


def interpret_cohort_retention(
    avg_retention_by_period: Dict[int, float],
    total_cohorts: int,
    use_ai: bool = False,
    model_path: Optional[str] = None,
    ollama_model: str = 'llama3',
) -> InterpretationResult:
    """
    Interpret cohort retention analysis.

    Args:
        avg_retention_by_period: Dict mapping period to retention %
        total_cohorts: Number of cohorts analyzed
        use_ai: If True, try AI backends before rule-based fallback
        model_path: Path to GGUF model file (local LLM)
        ollama_model: Ollama model name (fallback)

    Returns:
        InterpretationResult
    """
    periods = sorted(avg_retention_by_period.keys())
    retention_values = [avg_retention_by_period[p] for p in periods]
    
    # Calculate retention trend
    if len(retention_values) >= 2:
        first_retention = retention_values[0]
        last_retention = retention_values[-1]
        retention_drop = first_retention - last_retention
    else:
        retention_drop = 0
    
    # Summary
    if retention_drop > 50:
        summary = f"Steep retention decline: {retention_drop:.1f}% drop from period 0 to period {periods[-1]}."
    elif retention_drop > 30:
        summary = f"Moderate retention decline: {retention_drop:.1f}% drop over {periods[-1]} periods."
    elif retention_drop > 10:
        summary = f"Gradual retention decline: {retention_drop:.1f}% drop over {periods[-1]} periods."
    else:
        summary = f"Strong retention: only {retention_drop:.1f}% drop over {periods[-1]} periods."
    
    # Findings
    findings = [
        f"Analyzed {total_cohorts} cohorts over {len(periods)} periods",
        f"Period 0 retention: {retention_values[0]:.1f}%",
        f"Period {periods[-1]} retention: {retention_values[-1]:.1f}%"
    ]
    
    # Recommendations
    if retention_drop > 40:
        recs = [
            "Critical: Implement retention improvement program immediately",
            "Analyze reasons for churn in early periods",
            "Test re-engagement campaigns for at-risk cohorts"
        ]
    elif retention_drop > 20:
        recs = [
            "Focus on onboarding experience to improve early retention",
            "Implement targeted re-engagement campaigns",
            "Analyze successful cohorts to identify retention drivers"
        ]
    else:
        recs = [
            "Retention is strong - maintain current engagement strategies",
            "Consider loyalty programs to further improve long-term retention",
            "Benchmark against industry standards"
        ]
    
    rule_result = InterpretationResult(
        summary=summary,
        key_findings=findings,
        recommendations=recs,
        confidence="Rule-based",
        raw_response=""
    )

    if use_ai:
        prompt = (
            f"Interpret this cohort retention analysis for a marketing team:\n"
            f"Cohorts: {total_cohorts} | Periods tracked: {len(periods)}\n"
            f"Retention by period: { {p: f'{v:.1f}%' for p, v in avg_retention_by_period.items()} }\n\n"
            f"Give: 1 summary sentence, 3 key findings, 3 recommendations."
        )
        raw = _ai_generate(prompt, model_path=model_path, ollama_model=ollama_model)
        if raw:
            ai_summary, ai_findings, ai_recs = _parse_llm_response(raw)
            return InterpretationResult(
                summary=ai_summary,
                key_findings=ai_findings,
                recommendations=ai_recs,
                confidence="AI-powered",
                raw_response=raw,
            )

    return rule_result


# Export
__all__ = [
    'InterpretationResult',
    'interpret_ab_test',
    'interpret_cohort_retention',
]
