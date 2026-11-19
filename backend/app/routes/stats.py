"""Model and corpus facts for the 'under the hood' panel."""

import os

from fastapi import APIRouter

from ..schemas import StatsResponse
from ..services import predictor, ayurveda_lookup, retriever
from ..services.llm_client import llm_enabled
from ..safety.red_flags import RED_FLAG_PATTERNS

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rag", "knowledge_base")

router = APIRouter(tags=["stats"])


@router.get("/safety/categories")
def safety_categories():
    """
    The red-flag categories evaluated on every message.

    The doctor-ready report states which checks were run, not just that a
    check happened, so a clinician can see what the tool did and did not
    look for. Read from RED_FLAG_PATTERNS so the list cannot drift.
    """
    return {
        "categories": [
            {"key": key, "description": defn.get("description", key)}
            for key, defn in RED_FLAG_PATTERNS.items()
        ]
    }


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    kb_documents = 0
    if os.path.isdir(KB_DIR):
        kb_documents = len([f for f in os.listdir(KB_DIR) if f.endswith(".md")])

    kb_chunks = 0
    try:
        from ..services.retriever import _collection

        if _collection is not None:
            kb_chunks = _collection.count()
    except Exception:
        pass

    # Read the model identity from the loaded artifact rather than hardcoding
    # it. This panel used to claim "Multinomial Naive Bayes" while the
    # deployed estimator was Bernoulli, and AROGYA_MODEL can switch the
    # variant at startup, so a literal here is guaranteed to drift.
    pretty = {
        "BernoulliNB": "Bernoulli Naive Bayes",
        "MultinomialNB": "Multinomial Naive Bayes",
    }
    try:
        raw = (predictor.get_metrics() or {}).get("model") or type(predictor.get_model()).__name__
    except Exception:
        raw = "Naive Bayes"
    model_type = pretty.get(raw, raw)

    return StatsResponse(
        model_type=model_type,
        diseases=len(predictor.get_disease_symptom_matrix()) or 0,
        symptoms=len(predictor.get_feature_order()),
        kb_documents=kb_documents,
        kb_chunks=kb_chunks,
        embedding_model="all-MiniLM-L6-v2",
        red_flag_patterns=len(RED_FLAG_PATTERNS),
        ayurveda_entries=len(getattr(ayurveda_lookup, "_kb", {})),
        llm_mode="LLM-assisted" if llm_enabled() else "Local fuzzy matching (no LLM key)",
    )
