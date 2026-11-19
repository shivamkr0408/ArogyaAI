"""
Arogya AI NLP pipeline.

A staged, inspectable clinical-text pipeline built on named algorithms
rather than a single opaque call:

    lexicon     canonical vocabulary, colloquial English, code-mixed Hinglish
    tokenizer   offset-preserving tokenization + Porter (1980) stemmer
    normalizer  elongation, contractions, code-mixed rewriting, spell repair
    negation    NegEx (2001) extended to ConText (2009) -- 3 assertion axes
    temporal    TIMEX-style duration / frequency / trend normalization
    ner         exact -> fuzzy -> embedding recognition cascade with spans
    intent      TF-IDF word+char n-grams into logistic regression
    sentiment   weighted distress lexicon with negation and intensifiers
    pipeline    composes all of the above, with per-stage timings

`pipeline.analyze(text)` is the single entry point used by the orchestrator
and by the /api/nlp/analyze endpoint.
"""

from . import (  # noqa: F401
    intent,
    lexicon,
    negation,
    ner,
    normalizer,
    pipeline,
    sentiment,
    temporal,
    tokenizer,
)
from .pipeline import analyze, warm_up  # noqa: F401

__all__ = [
    "analyze",
    "warm_up",
    "intent",
    "lexicon",
    "negation",
    "ner",
    "normalizer",
    "pipeline",
    "sentiment",
    "temporal",
    "tokenizer",
]
