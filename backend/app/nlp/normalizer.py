"""
Text normalization with an offset map back to the original string.

The problem this solves: to recognize symptoms reliably we want to rewrite
the input ("pet dard se pareshan hu", "I have a feaver and coughhh") into
clean clinical English. But the UI highlights entities *in the text the user
typed*, so a naive `str.replace` would destroy the character offsets the
highlighting depends on.

So every rewrite is recorded as an :class:`Edit`, and the result keeps a
segment table that maps any span in the normalized string back to the span
it came from in the original. Four normalization passes run, all in a single
sweep so the offset map never has to be composed:

1. **Elongation collapsing** -- "paaaain" -> "pain" (informal text).
2. **Contraction expansion** -- "don't" -> "do not", which is what makes
   negation detection work at all.
3. **Code-mixed rewriting** -- "pet dard" -> "stomach pain" (longest match
   wins, so "tez bukhar" beats "bukhar").
4. **Spell correction** -- bounded Damerau-Levenshtein against the medical
   vocabulary only, so a typo can only be pulled towards a clinical term and
   ordinary English words are left alone.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

from . import lexicon
from .tokenizer import tokenize

_ELONGATION_RE = re.compile(r"([a-z])\1{2,}")


@dataclass
class Edit:
    """One rewrite, with both the original and normalized span it occupies."""

    kind: str  # elongation | contraction | code_mixed | spelling
    original: str
    replacement: str
    orig_start: int
    orig_end: int
    norm_start: int = 0
    norm_end: int = 0

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "original": self.original,
            "replacement": self.replacement,
            "orig_start": self.orig_start,
            "orig_end": self.orig_end,
        }


@dataclass
class NormalizedText:
    original: str
    text: str
    edits: list[Edit] = field(default_factory=list)
    # (norm_start, norm_end, orig_start, orig_end) for every emitted run.
    segments: list[tuple[int, int, int, int]] = field(default_factory=list)

    def to_original_span(self, norm_start: int, norm_end: int) -> tuple[int, int]:
        """
        Map a span in the normalized text back to the original string.

        A span that overlaps a rewrite returns the full original extent of
        that rewrite, which is what highlighting wants: if "pet dard" became
        "stomach pain", matching "stomach pain" should light up "pet dard".
        """
        starts: list[int] = []
        ends: list[int] = []
        for n_start, n_end, o_start, o_end in self.segments:
            if n_start < norm_end and norm_start < n_end:
                # Unchanged runs are 1:1, so the offset can be interpolated
                # instead of widening to the whole run.
                if (n_end - n_start) == (o_end - o_start):
                    starts.append(o_start + max(0, norm_start - n_start))
                    ends.append(o_start + min(n_end - n_start, norm_end - n_start))
                else:
                    starts.append(o_start)
                    ends.append(o_end)
        if not starts:
            return (0, 0)
        return (min(starts), max(ends))

    def edits_of(self, kind: str) -> list[Edit]:
        return [e for e in self.edits if e.kind == kind]


def _collapse_elongation(word: str) -> Optional[str]:
    """"coughhh" -> "cough". Returns None when nothing was elongated."""
    if not _ELONGATION_RE.search(word):
        return None
    collapsed_single = _ELONGATION_RE.sub(r"\1", word)
    collapsed_double = _ELONGATION_RE.sub(r"\1\1", word)
    known = (
        set(lexicon.spell_vocabulary())
        | lexicon.PROTECTED_WORDS
        | lexicon.HINDI_FUNCTION_WORDS
    )
    for candidate in (collapsed_single, collapsed_double):
        if candidate in known:
            return candidate
    return collapsed_single


def _spell_correct(word: str) -> Optional[tuple[str, int]]:
    """
    Correct `word` to the nearest medical-vocabulary term, or None.

    The distance budget scales with length so that short words -- where one
    edit can reach a dozen unrelated terms -- are left alone.
    """
    if not word.isalpha():
        return None
    if lexicon.is_protected(word):
        return None

    # Explicit table first: it covers short typos that bounded edit distance
    # deliberately refuses to touch.
    direct = lexicon.MISSPELLINGS.get(word)
    if direct and direct != word:
        return (direct, 1)

    if len(word) < 4:
        return None

    vocabulary = lexicon.spell_vocabulary()
    if word in vocabulary:
        return None

    budget = 2 if len(word) >= 7 else 1
    match = process.extractOne(
        word,
        vocabulary,
        scorer=DamerauLevenshtein.distance,
        score_cutoff=budget,
    )
    if not match:
        return None
    candidate, distance, _ = match
    # Guard against pulling a word to a much shorter/longer term.
    if abs(len(candidate) - len(word)) > budget:
        return None
    return (candidate, int(distance))


def _code_mixed_candidates(text_lower: str, tokens) -> list[Edit]:
    """Greedy longest-match sweep for romanized-Hindi / Devanagari phrases."""
    edits: list[Edit] = []
    consumed: set[int] = set()
    phrases = lexicon.code_mixed_phrases()
    max_len = max((len(p.split()) for p, _ in phrases), default=1)

    by_length: dict[int, dict[str, str]] = {}
    for phrase, replacement in phrases:
        by_length.setdefault(len(phrase.split()), {})[phrase] = replacement

    for n in range(max_len, 0, -1):
        table = by_length.get(n)
        if not table:
            continue
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            if any(t.index in consumed for t in window):
                continue
            if len({t.sentence_index for t in window}) > 1:
                continue
            surface = " ".join(t.lower for t in window)
            replacement = table.get(surface)
            if replacement is None:
                continue
            edits.append(
                Edit(
                    kind="code_mixed",
                    original=text_lower[window[0].start : window[-1].end],
                    replacement=replacement,
                    orig_start=window[0].start,
                    orig_end=window[-1].end,
                )
            )
            consumed.update(t.index for t in window)
    return edits


def normalize(text: str) -> NormalizedText:
    """Run every normalization pass and return the aligned result."""
    if not text:
        return NormalizedText(original=text, text="", edits=[], segments=[])

    # NFC keeps Devanagari combining marks together. Guard the offset
    # contract: if the form changes length, keep the original string.
    folded = unicodedata.normalize("NFC", text)
    base = folded if len(folded) == len(text) else text
    lowered = base.lower()

    tokens = tokenize(lowered)
    candidates: list[Edit] = _code_mixed_candidates(lowered, tokens)
    claimed = {
        idx
        for edit in candidates
        for idx in range(edit.orig_start, edit.orig_end)
    }

    for tok in tokens:
        if any(i in claimed for i in range(tok.start, tok.end)):
            continue
        if tok.is_numeric:
            continue

        word = tok.lower
        # 1. elongation
        collapsed = _collapse_elongation(word)
        if collapsed and collapsed != word:
            candidates.append(
                Edit("elongation", word, collapsed, tok.start, tok.end)
            )
            word = collapsed

        # 2. contractions
        expansion = lexicon.CONTRACTIONS.get(word)
        if expansion:
            candidates.append(
                Edit("contraction", tok.lower, expansion, tok.start, tok.end)
            )
            continue

        # 3. spelling
        correction = _spell_correct(word)
        if correction:
            candidate, distance = correction
            candidates.append(
                Edit("spelling", tok.lower, candidate, tok.start, tok.end)
            )

    # Resolve overlaps: longest span first, then leftmost, then by pass order.
    priority = {"code_mixed": 0, "contraction": 1, "elongation": 2, "spelling": 3}
    candidates.sort(
        key=lambda e: (
            -(e.orig_end - e.orig_start),
            e.orig_start,
            priority.get(e.kind, 9),
        )
    )
    accepted: list[Edit] = []
    taken: set[int] = set()
    for edit in candidates:
        span = range(edit.orig_start, edit.orig_end)
        if any(i in taken for i in span):
            continue
        accepted.append(edit)
        taken.update(span)
    accepted.sort(key=lambda e: e.orig_start)

    # Emit the normalized string, recording the segment table as we go.
    out: list[str] = []
    segments: list[tuple[int, int, int, int]] = []
    cursor = 0
    norm_len = 0
    for edit in accepted:
        if edit.orig_start > cursor:
            chunk = lowered[cursor : edit.orig_start]
            out.append(chunk)
            segments.append((norm_len, norm_len + len(chunk), cursor, edit.orig_start))
            norm_len += len(chunk)
        edit.norm_start = norm_len
        out.append(edit.replacement)
        norm_len += len(edit.replacement)
        edit.norm_end = norm_len
        segments.append((edit.norm_start, edit.norm_end, edit.orig_start, edit.orig_end))
        cursor = edit.orig_end

    if cursor < len(lowered):
        chunk = lowered[cursor:]
        out.append(chunk)
        segments.append((norm_len, norm_len + len(chunk), cursor, len(lowered)))

    return NormalizedText(
        original=text,
        text="".join(out),
        edits=accepted,
        segments=segments,
    )
