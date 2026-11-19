"""
Clinical entity recognition over the normalized text.

Three recognition strategies run in a cascade, each cheaper and more precise
than the next is expensive and more permissive. Every entity carries the
strategy that found it and a confidence, so the UI can show *how* a symptom
was recognized rather than just asserting it:

1. **exact** -- longest-match dictionary lookup against the phrase index,
   over both surface and Porter-lemma forms. Confidence 1.0.
2. **fuzzy** -- bounded Damerau-Levenshtein / token ratio for phrasings the
   dictionary does not list verbatim. This is what `rapidfuzz` was imported
   for in the original extractor but never actually used.
3. **embedding** -- MiniLM cosine similarity against the canonical symptom
   phrases, for paraphrases no string metric can reach ("I feel like my head
   is going to explode" -> headache). Runs only when the first two passes
   come up empty, so the common path stays fast, and it reuses the
   sentence-transformer the RAG retriever already has in memory rather than
   loading a second copy.

Alongside symptoms, the recognizer emits SEVERITY and BODY_SITE entities and
attaches each symptom's ConText assertion, so the caller can tell an
affirmed, patient-attributed, current symptom from a negated or inherited
one.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

from . import lexicon, negation, sentiment, temporal
from .negation import Assertion
from .normalizer import NormalizedText, normalize
from .tokenizer import Sentence, Token, lemma, tokenize_sentences

logger = logging.getLogger(__name__)

# Tunables. The fuzzy cutoffs are deliberately strict: a false symptom is
# far more costly here than a missed one, because it feeds a classifier.
FUZZY_CUTOFF = 88.0
FUZZY_CUTOFF_SHORT = 93.0
EMBEDDING_CUTOFF = 0.62
MAX_FUZZY_NGRAM = 3
MAX_EMBEDDING_NGRAM = 5

SEMANTIC_ENABLED = os.getenv("AROGYA_SEMANTIC_NER", "1") not in ("0", "false", "False")

# Words that cannot carry an entity on their own, used to skip pointless
# candidate windows in the fuzzy and embedding passes.
_WINDOW_STOP = frozenset("""
i me my you he she it we they the a an and or but if then of in on at to from
by for with is am are was were be been being do does did have has had will
would can could may might must no not so very really just also too about
since while when what how why there here that this these those get got go
feel feels feeling felt think
""".split())

# Single words that express distress about the consultation rather than a
# symptom. Defined in sentiment.py so the two layers stay in step.
_DISTRESS_WORDS = sentiment.META_DISTRESS_WORDS

# Punctuation that breaks the association between a modifier and a symptom:
# in "really worried - high fever", "really" modifies "worried", not the fever.
_SEVERITY_BARRIERS = frozenset(",;:-–—.!?()")


@dataclass
class Entity:
    """One recognized mention, located in both the original and normalized text."""

    label: str  # SYMPTOM | SEVERITY | BODY_SITE | TEMPORAL
    text: str  # surface form in the normalized text
    original_text: str  # surface form in the user's original text
    canonical: str  # model feature name for SYMPTOM, label value otherwise
    char_start: int  # offset into the ORIGINAL text
    char_end: int
    norm_start: int
    norm_end: int
    strategy: str  # exact | fuzzy | embedding | lexicon
    confidence: float
    assertion: Optional[Assertion] = None
    severity: Optional[str] = None
    duration_days: Optional[float] = None
    matched_phrase: str = ""

    @property
    def is_present(self) -> bool:
        """Affirmed, current, and about the patient."""
        if self.label not in ("SYMPTOM", "DISEASE"):
            return True
        return self.assertion is None or self.assertion.is_clinically_present

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "text": self.text,
            "original_text": self.original_text,
            "canonical": self.canonical,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
            "matched_phrase": self.matched_phrase,
            "severity": self.severity,
            "duration_days": self.duration_days,
            "assertion": self.assertion.as_dict() if self.assertion else None,
            "is_present": self.is_present,
        }


# ---------------------------------------------------------------------------
# Embedding tier -- lazily initialized, shared with the RAG retriever.
# ---------------------------------------------------------------------------

_embedding_model = None
_symptom_matrix = None
_symptom_keys: tuple[str, ...] = ()


def _get_embedding_model():
    """
    Reuse the retriever's SentenceTransformer when it is already loaded.

    Loading a second MiniLM would double both start-up time and RSS for no
    benefit, so this only falls back to its own instance if RAG is disabled.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from ..services import retriever

        shared = getattr(retriever, "_model", None)
        if shared is not None:
            _embedding_model = shared
            logger.info("Semantic NER is reusing the retriever's embedding model")
            return _embedding_model
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Semantic NER loaded its own embedding model")
    except Exception as exc:
        logger.warning("Semantic NER unavailable: %s", exc)
        _embedding_model = None
    return _embedding_model


def _get_symptom_embeddings():
    """Pre-encode one descriptive phrase per canonical symptom."""
    global _symptom_matrix, _symptom_keys
    if _symptom_matrix is not None:
        return _symptom_keys, _symptom_matrix

    model = _get_embedding_model()
    if model is None:
        return (), None

    synonyms = lexicon.clinical_synonyms()
    keys: list[str] = []
    phrases: list[str] = []
    for symptom in lexicon.canonical_symptoms():
        readable = symptom.replace("_", " ")
        extras = synonyms.get(symptom, [])[:2]
        description = readable
        if extras:
            description = f"{readable}, also described as {', '.join(extras)}"
        keys.append(symptom)
        phrases.append(description)

    try:
        import numpy as np

        matrix = model.encode(phrases, normalize_embeddings=True)
        _symptom_matrix = np.asarray(matrix)
        _symptom_keys = tuple(keys)
    except Exception as exc:
        logger.warning("Failed to pre-encode symptom phrases: %s", exc)
        return (), None

    return _symptom_keys, _symptom_matrix


def warm_semantic_index() -> bool:
    """Pre-build the embedding tier so the first user turn is not slow."""
    if not SEMANTIC_ENABLED:
        return False
    keys, matrix = _get_symptom_embeddings()
    return matrix is not None and len(keys) > 0


# ---------------------------------------------------------------------------
# Recognition passes
# ---------------------------------------------------------------------------


def _windows(tokens: list[Token], n: int) -> list[list[Token]]:
    out = []
    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        if len({t.sentence_index for t in window}) > 1:
            continue
        out.append(window)
    return out


def _surface(window: list[Token]) -> str:
    return " ".join(t.lower for t in window)


def _lemma_surface(window: list[Token]) -> str:
    return " ".join(t.stem for t in window)


def _exact_pass(
    tokens: list[Token], claimed: set[int]
) -> list[tuple[list[Token], str, str, float, str]]:
    """Longest-match dictionary lookup on surface and lemma forms."""
    index = lexicon.phrase_index()
    # A lemma-keyed view, so "aching joints" reaches "joint pain" aliases.
    lemma_index: dict[str, str] = {}
    for phrase, symptom in index.items():
        key = " ".join(lemma(w) for w in phrase.split())
        lemma_index.setdefault(key, symptom)

    hits = []
    max_n = min(lexicon.max_phrase_length(), len(tokens))
    for n in range(max_n, 0, -1):
        for window in _windows(tokens, n):
            if any(t.index in claimed for t in window):
                continue
            surface = _surface(window)
            symptom = index.get(surface)
            strategy = "exact"
            confidence = 1.0
            matched = surface
            if symptom is None:
                lemma_form = _lemma_surface(window)
                symptom = lemma_index.get(lemma_form)
                if symptom is not None:
                    matched = lemma_form
                    confidence = 0.95
            if symptom is None:
                continue
            hits.append((window, symptom, strategy, confidence, matched))
            claimed.update(t.index for t in window)
    return hits


def _fuzzy_pass(
    tokens: list[Token], claimed: set[int]
) -> list[tuple[list[Token], str, str, float, str]]:
    """Approximate matching for phrasings the dictionary lacks verbatim."""
    index = lexicon.phrase_index()
    choices = list(index.keys())
    hits = []

    for n in range(min(MAX_FUZZY_NGRAM, len(tokens)), 0, -1):
        for window in _windows(tokens, n):
            if any(t.index in claimed for t in window):
                continue
            surface = _surface(window)
            content = [w for w in surface.split() if w not in _WINDOW_STOP]
            if not content:
                continue
            # Single short tokens collide with too much of the vocabulary.
            cutoff = FUZZY_CUTOFF if len(surface) >= 6 else FUZZY_CUTOFF_SHORT
            match = process.extractOne(
                surface, choices, scorer=fuzz.ratio, score_cutoff=cutoff
            )
            if not match:
                continue
            phrase, score, _ = match
            # Reject matches that differ wildly in length -- high ratio on a
            # short string is not evidence of the same concept.
            if abs(len(phrase) - len(surface)) > max(3, len(surface) * 0.4):
                continue
            hits.append((window, index[phrase], "fuzzy", score / 100.0, phrase))
            claimed.update(t.index for t in window)
    return hits


def _embedding_pass(
    tokens: list[Token], claimed: set[int]
) -> list[tuple[list[Token], str, str, float, str]]:
    """Cosine similarity against the canonical symptom phrases."""
    keys, matrix = _get_symptom_embeddings()
    if matrix is None or not keys:
        return []

    model = _get_embedding_model()
    if model is None:
        return []

    candidates: list[list[Token]] = []
    for n in range(min(MAX_EMBEDDING_NGRAM, len(tokens)), 1, -1):
        for window in _windows(tokens, n):
            if any(t.index in claimed for t in window):
                continue
            content = [t for t in window if t.lower not in _WINDOW_STOP]
            if len(content) < 1:
                continue
            candidates.append(window)

    if not candidates:
        return []

    # Cap the work: encoding is the expensive part of this pass.
    candidates = candidates[:24]

    try:
        import numpy as np

        phrases = [_surface(w) for w in candidates]
        vectors = np.asarray(model.encode(phrases, normalize_embeddings=True))
        scores = vectors @ matrix.T
    except Exception as exc:
        logger.warning("Embedding NER pass failed: %s", exc)
        return []

    hits = []
    order = sorted(
        range(len(candidates)), key=lambda i: float(scores[i].max()), reverse=True
    )
    for i in order:
        window = candidates[i]
        if any(t.index in claimed for t in window):
            continue
        best = int(scores[i].argmax())
        score = float(scores[i][best])
        if score < EMBEDDING_CUTOFF:
            continue
        hits.append((window, keys[best], "embedding", score, keys[best].replace("_", " ")))
        claimed.update(t.index for t in window)
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class Recognition:
    normalized: NormalizedText
    sentences: list[Sentence]
    entities: list[Entity] = field(default_factory=list)
    temporal: list = field(default_factory=list)

    @property
    def symptoms(self) -> list[Entity]:
        return [e for e in self.entities if e.label == "SYMPTOM"]

    @property
    def present_symptoms(self) -> list[str]:
        """Canonical names that should reach the classifier, de-duplicated."""
        seen: list[str] = []
        for entity in self.symptoms:
            if entity.is_present and entity.canonical not in seen:
                seen.append(entity.canonical)
        return seen

    @property
    def diseases(self) -> list[Entity]:
        return [e for e in self.entities if e.label == "DISEASE"]

    @property
    def mentioned_conditions(self) -> list[dict]:
        """
        Named conditions with who they belong to, for the consultation record.

        An affirmed, patient-attributed condition is a comorbidity; one with a
        family cue is family history.
        """
        out = []
        for entity in self.diseases:
            assertion = entity.assertion
            out.append(
                {
                    "condition": entity.canonical,
                    "attribution": (
                        "family"
                        if assertion and assertion.experiencer == "other"
                        else "patient"
                    ),
                    "status": assertion.label if assertion else "affirmed",
                    "text": entity.original_text,
                }
            )
        return out

    @property
    def negated_symptoms(self) -> list[str]:
        seen: list[str] = []
        for entity in self.symptoms:
            if (
                entity.assertion
                and entity.assertion.negation == "negated"
                and entity.canonical not in seen
            ):
                seen.append(entity.canonical)
        return seen


def _sentence_for(sentences: list[Sentence], token: Token) -> Optional[Sentence]:
    for sent in sentences:
        if sent.index == token.sentence_index:
            return sent
    return None


def recognize(text: str, use_semantic: Optional[bool] = None) -> Recognition:
    """
    Run the full recognition cascade over `text`.

    The returned entities carry offsets into the *original* string, even
    where normalization rewrote the span (so "pet dard" highlights correctly
    after being rewritten to "stomach pain").
    """
    normalized = normalize(text)
    sentences = tokenize_sentences(normalized.text)
    tokens = [tok for sent in sentences for tok in sent.tokens]

    result = Recognition(normalized=normalized, sentences=sentences)
    if not tokens:
        return result

    claimed: set[int] = set()
    raw_hits = _exact_pass(tokens, claimed)
    raw_hits += _fuzzy_pass(tokens, claimed)

    semantic = SEMANTIC_ENABLED if use_semantic is None else use_semantic
    if semantic and not raw_hits:
        raw_hits += _embedding_pass(tokens, claimed)

    # Cue detection is per sentence, and reused across every entity in it.
    cues_by_sentence = {
        sent.index: negation.find_cues(sent.tokens) for sent in sentences
    }

    # Token positions occupied by a cue phrase. A single word that is doing
    # cue duty should not also be reported as an entity: in "should I be
    # worried", "worried" is the hypothetical trigger, not an anxiety symptom.
    cue_tokens: set[int] = set()
    for sent in sentences:
        for cue in cues_by_sentence.get(sent.index, []):
            for pos in range(cue.token_start, cue.token_end):
                if pos < len(sent.tokens):
                    cue_tokens.add(sent.tokens[pos].index)

    temporal_expressions = temporal.extract(normalized.text)
    result.temporal = temporal_expressions
    primary = temporal.primary_duration(temporal_expressions)

    for window, canonical, strategy, confidence, matched in raw_hits:
        if len(window) == 1 and window[0].index in cue_tokens:
            continue
        # A lone distress word is the user commenting on the consultation
        # ("I'm really worried"), not reporting a clinical symptom. The
        # distress layer already consumes it; letting it also set the anxiety
        # feature measurably skews the posterior.
        if len(window) == 1 and window[0].lower in _DISTRESS_WORDS:
            continue
        norm_start, norm_end = window[0].start, window[-1].end
        orig_start, orig_end = normalized.to_original_span(norm_start, norm_end)
        sentence = _sentence_for(sentences, window[0])
        assertion = Assertion()
        if sentence is not None:
            assertion = negation.classify(
                sentence,
                norm_start,
                norm_end,
                cues=cues_by_sentence.get(sentence.index),
            )

        entity = Entity(
            label="SYMPTOM",
            text=normalized.text[norm_start:norm_end],
            original_text=normalized.original[orig_start:orig_end],
            canonical=canonical,
            char_start=orig_start,
            char_end=orig_end,
            norm_start=norm_start,
            norm_end=norm_end,
            strategy=strategy,
            confidence=confidence,
            assertion=assertion,
            matched_phrase=matched,
            severity=_severity_near(tokens, window, claimed, normalized.text),
            duration_days=primary.value if primary else None,
        )
        result.entities.append(entity)

    result.entities.extend(
        _disease_entities(tokens, normalized, sentences, cues_by_sentence, claimed)
    )
    # Modifier entities inside a symptom's own span are part of its name, not
    # separate mentions: "high" in "high fever" must not also be reported as a
    # severity, or the UI highlights the same text twice.
    occupied = [(e.norm_start, e.norm_end) for e in result.entities]
    for modifier in _modifier_entities(tokens, normalized):
        if any(
            modifier.norm_start < end and start < modifier.norm_end
            for start, end in occupied
        ):
            continue
        result.entities.append(modifier)

    result.entities.extend(_temporal_entities(temporal_expressions, normalized))
    result.entities.sort(key=lambda e: (e.char_start, e.char_end))
    return result


def _severity_near(
    tokens: list[Token], window: list[Token], claimed: set[int], text: str
) -> Optional[str]:
    """
    The severity modifier governing this mention, if any.

    Looks three tokens to the left ("really bad headache") and one to the
    right ("headache, severe"), skipping tokens that belong to another
    entity -- otherwise the "high" in "high fever and a cough" would be read
    as a severity modifier on the cough as well.
    """
    first = window[0].index
    last = window[-1].index
    own = {t.index for t in window}
    best: Optional[tuple[float, str]] = None
    for tok in tokens:
        if tok.sentence_index != window[0].sentence_index:
            continue
        if tok.index in claimed and tok.index not in own:
            continue
        if tok.index in own:
            continue
        distance = first - tok.index
        if not (0 < distance <= 3 or 0 < tok.index - last <= 1):
            continue
        hit = lexicon.SEVERITY_MODIFIERS.get(tok.lower)
        if not hit:
            continue
        # Do not reach across punctuation.
        if tok.index < first:
            between = text[tok.end : window[0].start]
        else:
            between = text[window[-1].end : tok.start]
        if any(ch in _SEVERITY_BARRIERS for ch in between):
            continue
        if best is None or hit[1] > best[0]:
            best = (hit[1], hit[0])
    return best[1] if best else None


def _disease_entities(
    tokens: list[Token],
    normalized: NormalizedText,
    sentences: list[Sentence],
    cues_by_sentence: dict,
    claimed: set[int],
) -> list[Entity]:
    """
    Recognize named conditions ("I am diabetic", "my father has asthma").

    These never enter the symptom feature vector -- they are comorbidity and
    family-history context, and the assertion axes are what tell the two
    apart.
    """
    index = lexicon.disease_index()
    if not index:
        return []

    max_n = min(max((len(p.split()) for p in index), default=1), len(tokens))
    out: list[Entity] = []
    local_claimed: set[int] = set()

    for n in range(max_n, 0, -1):
        for window in _windows(tokens, n):
            if any(t.index in local_claimed for t in window):
                continue
            surface = _surface(window)
            disease = index.get(surface)
            if disease is None:
                continue
            norm_start, norm_end = window[0].start, window[-1].end
            orig_start, orig_end = normalized.to_original_span(norm_start, norm_end)
            sentence = _sentence_for(sentences, window[0])
            assertion = Assertion()
            if sentence is not None:
                assertion = negation.classify(
                    sentence,
                    norm_start,
                    norm_end,
                    cues=cues_by_sentence.get(sentence.index),
                )
            out.append(
                Entity(
                    label="DISEASE",
                    text=normalized.text[norm_start:norm_end],
                    original_text=normalized.original[orig_start:orig_end],
                    canonical=disease,
                    char_start=orig_start,
                    char_end=orig_end,
                    norm_start=norm_start,
                    norm_end=norm_end,
                    strategy="exact",
                    confidence=1.0,
                    assertion=assertion,
                    matched_phrase=surface,
                )
            )
            local_claimed.update(t.index for t in window)
    return out


def _modifier_entities(tokens: list[Token], normalized: NormalizedText) -> list[Entity]:
    """Standalone SEVERITY and BODY_SITE entities, for display."""
    out: list[Entity] = []
    for tok in tokens:
        for table, label in (
            (lexicon.SEVERITY_MODIFIERS, "SEVERITY"),
            (lexicon.BODY_SITES, "BODY_SITE"),
        ):
            hit = table.get(tok.lower)
            if hit is None:
                continue
            canonical = hit[0] if isinstance(hit, tuple) else hit
            orig_start, orig_end = normalized.to_original_span(tok.start, tok.end)
            out.append(
                Entity(
                    label=label,
                    text=tok.lower,
                    original_text=normalized.original[orig_start:orig_end],
                    canonical=canonical,
                    char_start=orig_start,
                    char_end=orig_end,
                    norm_start=tok.start,
                    norm_end=tok.end,
                    strategy="lexicon",
                    confidence=1.0,
                    matched_phrase=tok.lower,
                )
            )
            break
    return out


def _temporal_entities(expressions: list, normalized: NormalizedText) -> list[Entity]:
    out: list[Entity] = []
    for expr in expressions:
        orig_start, orig_end = normalized.to_original_span(
            expr.char_start, expr.char_end
        )
        out.append(
            Entity(
                label="TEMPORAL",
                text=expr.text,
                original_text=normalized.original[orig_start:orig_end],
                canonical=expr.normalized or expr.kind,
                char_start=orig_start,
                char_end=orig_end,
                norm_start=expr.char_start,
                norm_end=expr.char_end,
                strategy="rule",
                confidence=1.0,
                matched_phrase=expr.kind,
                duration_days=expr.value,
            )
        )
    return out
