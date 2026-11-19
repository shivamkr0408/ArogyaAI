"""
NLP inspection and evaluation endpoints.

`/api/nlp/analyze` runs the full pipeline over an arbitrary sentence and
returns every intermediate result: the normalized text with each rewrite,
the token table with Porter stems, recognized entities with their character
spans and the strategy that found them, the ConText assertion axes and the
cues that set them, temporal expressions, intent posterior, distress score
and per-stage timings.

It exists so the pipeline can be *demonstrated* rather than described -- the
NLP Lab panel in the UI calls it live, and it is the endpoint to open in a
viva when asked "how does your negation detection actually work?".
"""

import json
import os

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..nlp import lexicon, negation, pipeline
from ..nlp.tokenizer import tokenize_sentences
from ..services import explain as explain_svc
from ..services import predictor

router = APIRouter(tags=["nlp"])

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval", "results")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    use_semantic: bool = False


class ExplainRequest(BaseModel):
    symptoms: list[str] = []
    negated: list[str] = []
    condition: str | None = None


@router.post("/nlp/analyze")
def analyze_text(request: AnalyzeRequest):
    """Full linguistic analysis of one sentence, with every stage exposed."""
    analysis = pipeline.analyze(
        request.text, use_semantic=request.use_semantic, include_tokens=True
    )
    payload = analysis.as_dict()

    # Cue spans per sentence, so the UI can underline negation scope.
    cue_detail = []
    for sentence in tokenize_sentences(analysis.normalized):
        cues = negation.find_cues(sentence.tokens)
        for cue in cues:
            scope_tokens = sentence.tokens[cue.scope_start : cue.scope_end]
            cue_detail.append(
                {
                    **cue.as_dict(),
                    "sentence": sentence.index,
                    "scope_text": " ".join(t.lower for t in scope_tokens),
                    "scope_start": scope_tokens[0].start if scope_tokens else None,
                    "scope_end": scope_tokens[-1].end if scope_tokens else None,
                }
            )
    payload["cues"] = cue_detail
    return payload


@router.get("/nlp/capabilities")
def nlp_capabilities():
    """What the pipeline knows -- sizes of each lexical resource."""
    return {
        "stages": [
            {
                "name": "Normalization",
                "algorithms": [
                    "Unicode NFC",
                    "contraction expansion",
                    "elongation collapsing",
                    "code-mixed (Hinglish/Devanagari) rewriting",
                    "Damerau-Levenshtein spell correction",
                ],
                "detail": "Offset-mapped, so entity spans still point at the original text",
            },
            {
                "name": "Tokenization",
                "algorithms": ["offset-preserving tokenizer", "Porter stemmer (1980)"],
                "detail": "Verified against Porter's published vocabulary/output pair",
            },
            {
                "name": "Entity recognition",
                "algorithms": [
                    "longest-match dictionary",
                    "rapidfuzz token ratio",
                    "MiniLM cosine similarity",
                ],
                "detail": "Cascade; the embedding tier only runs when the first two find nothing",
            },
            {
                "name": "Assertion",
                "algorithms": ["NegEx (2001)", "ConText (2009)"],
                "detail": "Three axes: negation, temporality, experiencer",
            },
            {
                "name": "Temporal",
                "algorithms": ["TIMEX-style duration/frequency/trend normalization"],
                "detail": "Durations resolved to days and bucketed onto intake options",
            },
            {
                "name": "Intent",
                "algorithms": ["TF-IDF word+char n-grams", "logistic regression"],
                "detail": "High-precision regex rules run before the model",
            },
            {
                "name": "Distress",
                "algorithms": ["weighted lexicon with negation and intensifiers"],
                "detail": "Drives an empathetic opening line at the elevated/high bands",
            },
        ],
        "resources": {
            "canonical_symptoms": len(lexicon.canonical_symptoms()),
            "surface_phrases": len(lexicon.phrase_index()),
            "code_mixed_phrases": len(lexicon.CODE_MIXED),
            "disease_phrases": len(lexicon.disease_index()),
            "spell_vocabulary": len(lexicon.spell_vocabulary()),
            "severity_modifiers": len(lexicon.SEVERITY_MODIFIERS),
            "body_sites": len(lexicon.BODY_SITES),
            "negation_cues": len(negation.PRE_NEGATION) + len(negation.POST_NEGATION),
            "pseudo_negation_cues": len(negation.PSEUDO_NEGATION),
            "termination_cues": len(negation.TERMINATION),
            "hypothetical_cues": len(negation.HYPOTHETICAL),
            "historical_cues": len(negation.HISTORICAL_PRE) + len(negation.HISTORICAL_POST),
            "family_cues": len(negation.FAMILY),
            "uncertainty_cues": len(negation.UNCERTAINTY),
        },
    }


@router.post("/nlp/explain")
def explain_prediction(request: ExplainRequest):
    """Per-symptom Naive Bayes attribution for an arbitrary symptom set."""
    feature_order = predictor.get_feature_order()
    flags = {
        s.lower().strip().replace(" ", "_"): True
        for s in request.symptoms
        if s.lower().strip().replace(" ", "_") in feature_order
    }
    if not flags:
        return {"error": "No recognized symptoms supplied", "predictions": []}

    predictions = predictor.predict(flags)
    explanation = explain_svc.explain(
        flags,
        top_disease=request.condition,
        negated_symptoms=[
            s.lower().strip().replace(" ", "_") for s in request.negated
        ],
    )
    return {
        "symptom_flags": sorted(flags),
        "predictions": predictions,
        "explanation": explanation,
        "summary": explain_svc.summarize(explanation),
    }


def _read_results(filename: str):
    path = os.path.join(EVAL_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


@router.get("/evaluation")
def evaluation():
    """
    Every measured result the project has, served from the files the
    evaluation scripts wrote. Nothing here is hand-entered.
    """
    model_comparison = _read_results("model_comparison.json")
    nlp_eval = _read_results("nlp_eval.json")
    rag_eval = _read_results("rag_eval.json")
    deployed = predictor.get_metrics()
    variants = predictor.get_all_metrics()

    missing = [
        name
        for name, value in (
            ("model_comparison", model_comparison),
            ("nlp_eval", nlp_eval),
            ("rag_eval", rag_eval),
        )
        if value is None
    ]

    return {
        "deployed_model": {
            "estimator": type(predictor.get_model()).__name__
            if predictor.get_model() is not None
            else None,
            "metrics": deployed,
        },
        "variants": variants,
        "model_comparison": model_comparison,
        "nlp_components": nlp_eval,
        "retrieval": rag_eval,
        "missing": missing,
        "how_to_regenerate": {
            "model_comparison": "python eval/compare_models.py",
            "nlp_components": "python eval/eval_nlp.py",
            "retrieval": "python eval/eval_rag.py",
            "deployed_model": "python ml/train_model.py",
        },
    }
