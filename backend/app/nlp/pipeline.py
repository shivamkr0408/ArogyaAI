"""
The composed NLP pipeline.

One call, one structured analysis of a user turn. Every stage is timed, so
the `/api/nlp/analyze` endpoint and the NLP Lab panel can show the real
cost of each stage rather than a diagram of what supposedly happens.

Stage order matters:

    normalize -> tokenize -> recognize entities -> assert (NegEx/ConText)
              -> temporal -> intent -> distress

Normalization has to run first because negation depends on contractions
being expanded ("don't" -> "do not"), and entity recognition depends on
code-mixed text already being rewritten. Assertion has to run after
recognition because it needs entity spans to compute scope over.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from . import intent as intent_module
from . import ner, sentiment, temporal
from .tokenizer import tokenize_sentences


@dataclass
class StageTiming:
    name: str
    ms: float
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "ms": round(self.ms, 2), "detail": self.detail}


@dataclass
class Analysis:
    """Everything the pipeline knows about one user turn."""

    original: str
    normalized: str
    tokens: list[dict] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    temporal: list[dict] = field(default_factory=list)
    normalization_edits: list[dict] = field(default_factory=list)
    intent: dict = field(default_factory=dict)
    distress: dict = field(default_factory=dict)
    timings: list[dict] = field(default_factory=list)

    # Derived, and what the orchestrator actually consumes.
    present_symptoms: list[str] = field(default_factory=list)
    negated_symptoms: list[str] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    duration_days: Optional[float] = None
    duration_bucket: Optional[str] = None
    trend: Optional[str] = None
    frequency: Optional[str] = None
    severity: Optional[str] = None

    @property
    def total_ms(self) -> float:
        return round(sum(t["ms"] for t in self.timings), 2)

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "tokens": self.tokens,
            "sentences": self.sentences,
            "entities": self.entities,
            "temporal": self.temporal,
            "normalization_edits": self.normalization_edits,
            "intent": self.intent,
            "distress": self.distress,
            "timings": self.timings,
            "present_symptoms": self.present_symptoms,
            "negated_symptoms": self.negated_symptoms,
            "conditions": self.conditions,
            "duration_days": self.duration_days,
            "duration_bucket": self.duration_bucket,
            "trend": self.trend,
            "frequency": self.frequency,
            "severity": self.severity,
            "total_ms": self.total_ms,
        }


_SEVERITY_RANK = {"mild": 1, "moderate": 2, "severe": 3}


def analyze(
    text: str,
    use_semantic: Optional[bool] = None,
    include_tokens: bool = True,
) -> Analysis:
    """
    Run the full pipeline over one user message.

    `include_tokens` is off for the hot chat path (the orchestrator does not
    need the token table) and on for the NLP Lab endpoint.
    """
    text = text or ""
    timings: list[StageTiming] = []

    def _timed(name: str, fn, detail_fn=None):
        start = time.perf_counter()
        value = fn()
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(
            StageTiming(name, elapsed, detail_fn(value) if detail_fn else "")
        )
        return value

    # Recognition internally normalizes, tokenizes and asserts; it is timed
    # as one stage here and broken out below from its own result.
    recognition = _timed(
        "Normalize + recognize entities",
        lambda: ner.recognize(text, use_semantic=use_semantic),
        lambda r: "%d entit%s, %d rewrite(s)"
        % (
            len(r.entities),
            "y" if len(r.entities) == 1 else "ies",
            len(r.normalized.edits),
        ),
    )

    intent_result = _timed(
        "Intent classification",
        lambda: intent_module.classify(text),
        lambda r: "%s (%.0f%%, %s)"
        % (r["intent"], r["confidence"] * 100, r["source"]),
    )

    distress_result = _timed(
        "Distress scoring",
        lambda: sentiment.score(text),
        lambda r: "%s (%.1f)" % (r.level, r.score),
    )

    analysis = Analysis(
        original=text,
        normalized=recognition.normalized.text,
    )

    if include_tokens:
        sentences = tokenize_sentences(recognition.normalized.text)
        analysis.sentences = [s.text for s in sentences]
        analysis.tokens = [
            {
                "text": tok.text,
                "stem": tok.stem,
                "start": tok.start,
                "end": tok.end,
                "sentence": tok.sentence_index,
            }
            for sent in sentences
            for tok in sent.tokens
        ]

    analysis.entities = [e.as_dict() for e in recognition.entities]
    analysis.temporal = [e.as_dict() for e in recognition.temporal]
    analysis.normalization_edits = [e.as_dict() for e in recognition.normalized.edits]
    analysis.intent = intent_result
    analysis.distress = distress_result.as_dict()
    analysis.timings = [t.as_dict() for t in timings]

    analysis.present_symptoms = recognition.present_symptoms
    analysis.negated_symptoms = recognition.negated_symptoms
    analysis.conditions = recognition.mentioned_conditions

    primary = temporal.primary_duration(recognition.temporal)
    if primary:
        analysis.duration_days = primary.value
        analysis.duration_bucket = primary.normalized

    trend_expr = temporal.first_of(recognition.temporal, "trend")
    if trend_expr:
        analysis.trend = trend_expr.normalized

    frequency_expr = temporal.first_of(recognition.temporal, "frequency")
    if frequency_expr:
        analysis.frequency = frequency_expr.normalized

    # The strongest severity modifier attached to any present symptom.
    severities = [
        e.severity
        for e in recognition.symptoms
        if e.severity and e.is_present
    ]
    if severities:
        analysis.severity = max(severities, key=lambda s: _SEVERITY_RANK.get(s, 0))

    return analysis


def warm_up() -> dict:
    """
    Pre-build the lazy pieces (intent model, embedding index) at startup so
    the first user turn is not the one that pays for them.
    """
    from . import lexicon

    status = {}
    start = time.perf_counter()
    status["phrases"] = len(lexicon.phrase_index())
    status["diseases"] = len(lexicon.disease_index())
    status["intent_model"] = intent_module._get_model() is not None
    status["semantic_ner"] = ner.warm_semantic_index()
    status["ms"] = round((time.perf_counter() - start) * 1000, 1)
    return status
