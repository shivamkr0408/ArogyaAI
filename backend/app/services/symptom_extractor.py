"""
Symptom extraction -- a thin adapter over `app.nlp.pipeline`.

Everything that used to live here (a hand-rolled stemmer, a phrase table, a
12-character negation look-behind, and an unused `rapidfuzz` import) has
moved into the staged pipeline under `app/nlp/`, where each stage is
separately testable and separately measured by `eval/eval_nlp.py`.

This module keeps the original call signature so existing callers keep
working, and keeps the optional LLM path: when an API key is configured the
LLM's symptom list is *unioned* with the local pipeline's rather than
replacing it, and the pipeline's assertion layer still gets the final say.
That ordering matters -- an LLM will happily return "fever" for "I don't
have a fever", and NegEx is the thing that catches it.
"""

import json
import logging
from typing import Optional

from ..nlp import pipeline
from ..prompts import load_prompt
from .llm_client import chat_completion

logger = logging.getLogger(__name__)


def _parse_llm_json(raw: str) -> Optional[dict]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) > 1 else ""
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM extraction response: %s", raw[:200])
        return None
    return parsed if isinstance(parsed, dict) else None


async def analyze_message(
    user_message: str, canonical_symptoms: list[str], use_llm: bool = True
) -> dict:
    """
    Full analysis of one user turn.

    Returns the legacy extraction keys plus `analysis`, the complete
    pipeline output (entities with spans, assertions, intent, temporal
    context, distress, per-stage timings).
    """
    analysis = pipeline.analyze(user_message, include_tokens=False)

    present = list(analysis.present_symptoms)
    negated = list(analysis.negated_symptoms)
    source = "local-pipeline"

    if use_llm:
        llm_symptoms = await _llm_symptoms(user_message, canonical_symptoms)
        if llm_symptoms is not None:
            source = "local-pipeline + llm"
            # The LLM may spot a phrasing the lexicon lacks, but it must not
            # resurrect something the assertion layer ruled out.
            valid = set(canonical_symptoms)
            for symptom in llm_symptoms:
                canonical = symptom.lower().strip().replace(" ", "_")
                if (
                    canonical in valid
                    and canonical not in present
                    and canonical not in negated
                ):
                    present.append(canonical)

    return {
        "extracted_symptoms": present,
        "raw_negations": negated,
        "possible_emergency_language": False,
        "source": source,
        "analysis": analysis,
    }


async def _llm_symptoms(
    user_message: str, canonical_symptoms: list[str]
) -> Optional[list[str]]:
    """The LLM's symptom list, or None when no LLM is configured/usable."""
    system_prompt = load_prompt("extract_symptoms")
    symptom_list = ", ".join(canonical_symptoms[:100])
    user_msg = (
        f"Canonical symptom list:\n{symptom_list}\n\n"
        f'User message: "{user_message}"\n\n'
        "Return ONLY the JSON object. No commentary."
    )

    raw = await chat_completion(
        system_prompt, user_msg, temperature=0.1, max_tokens=512
    )
    if not raw:
        return None

    parsed = _parse_llm_json(raw)
    if not parsed or not isinstance(parsed.get("extracted_symptoms"), list):
        return None
    return [str(s) for s in parsed["extracted_symptoms"]]


async def extract_symptoms(user_message: str, canonical_symptoms: list[str]) -> dict:
    """Backwards-compatible entry point used by the original orchestrator."""
    return await analyze_message(user_message, canonical_symptoms)
