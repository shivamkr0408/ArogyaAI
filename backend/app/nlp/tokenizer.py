"""
Offset-preserving tokenization + the Porter (1980) stemming algorithm.

Why offsets matter here: every downstream stage (negation scope, entity
highlighting in the UI, temporal attachment) needs to know *where* in the
original string a token came from. The previous extractor threw the
positions away by doing ``set(re.findall(...))``, which is why negation
could only be approximated with a 12-character look-behind.

The stemmer is a full implementation of Porter's algorithm rather than the
earlier five-line suffix chopper, so "aching" / "aches" / "ached" all
collapse onto one form before lexicon lookup.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

_VOWELS = frozenset("aeiou")

# A token is a maximal run of letters (Latin or Devanagari, for code-mixed
# input), or a number such as "38.5" / "102".
_TOKEN_RE = re.compile(
    r"[A-Za-zऀ-ॿ]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?"
)
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]*|\n+")


@dataclass
class Token:
    text: str
    start: int
    end: int
    index: int
    lower: str = ""
    stem: str = ""
    is_numeric: bool = False
    sentence_index: int = 0

    def __post_init__(self):
        if not self.lower:
            self.lower = self.text.lower()
        self.is_numeric = bool(re.fullmatch(r"\d+(?:\.\d+)?", self.text))
        if not self.stem:
            self.stem = self.lower if self.is_numeric else lemma(self.lower)


@dataclass
class Sentence:
    text: str
    start: int
    end: int
    index: int
    tokens: list = field(default_factory=list)


def split_sentences(text: str) -> list[Sentence]:
    """Split on terminal punctuation and newlines, keeping character offsets."""
    out: list[Sentence] = []
    for match in _SENT_RE.finditer(text):
        raw = match.group(0)
        if not raw.strip():
            continue
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        start = match.start() + lead
        end = match.end() - trail
        out.append(Sentence(text=text[start:end], start=start, end=end, index=len(out)))
    if not out and text.strip():
        lead = len(text) - len(text.lstrip())
        stripped = text.strip()
        out.append(Sentence(text=stripped, start=lead, end=lead + len(stripped), index=0))
    return out


def tokenize(text: str) -> list[Token]:
    """Tokenize the whole string, tagging each token with its sentence index."""
    sentences = split_sentences(text)
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(text):
        s_idx = 0
        for sent in sentences:
            if sent.start <= match.start() < sent.end:
                s_idx = sent.index
                break
        tokens.append(
            Token(
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                index=len(tokens),
                sentence_index=s_idx,
            )
        )
    return tokens


def tokenize_sentences(text: str) -> list[Sentence]:
    """Sentences with their tokens already attached."""
    sentences = split_sentences(text)
    by_index: dict[int, list[Token]] = {s.index: [] for s in sentences}
    for tok in tokenize(text):
        by_index.setdefault(tok.sentence_index, []).append(tok)
    for sent in sentences:
        sent.tokens = by_index.get(sent.index, [])
    return sentences


def ngrams(tokens: list[Token], n: int) -> list[tuple[str, int, int, list[Token]]]:
    """
    Contiguous n-grams that do not straddle a sentence boundary.

    Returns (surface, char_start, char_end, member_tokens) so a phrase match
    can be reported with the exact span it covered in the original text.
    """
    out = []
    for i in range(len(tokens) - n + 1):
        window = tokens[i : i + n]
        if len({t.sentence_index for t in window}) > 1:
            continue
        surface = " ".join(t.lower for t in window)
        out.append((surface, window[0].start, window[-1].end, window))
    return out


# ---------------------------------------------------------------------------
# Porter stemmer (Porter, 1980), "An algorithm for suffix stripping".
# ---------------------------------------------------------------------------


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(word: str) -> int:
    """m -- the number of vowel-consonant sequences in ``word``."""
    n = len(word)
    i = 0
    count = 0
    while i < n and _is_consonant(word, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(word, i):
            i += 1
        if i >= n:
            break
        while i < n and _is_consonant(word, i):
            i += 1
        count += 1
    return count


def _contains_vowel(word: str) -> bool:
    return any(not _is_consonant(word, i) for i in range(len(word)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """*o -- ends consonant-vowel-consonant, where the last is not w, x or y."""
    n = len(word)
    if n < 3:
        return False
    return (
        _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _replace(word: str, suffix: str, replacement: str, min_measure: int) -> Optional[str]:
    """Apply suffix -> replacement when the remaining stem has m > min_measure."""
    if not word.endswith(suffix):
        return None
    stem_part = word[: len(word) - len(suffix)]
    if _measure(stem_part) > min_measure:
        return stem_part + replacement
    return word


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("logi", "log"),
]

_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
    "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]

def stem(word: str) -> str:
    """
    Porter-stem a single lowercase word.

    This is the unmodified algorithm -- no clinical special-casing -- so it
    can be checked against Porter's published vocabulary/output pair. Domain
    irregulars are handled one layer up, in :func:`lemma`.
    """
    word = word.lower()
    if len(word) <= 2 or not word.isalpha():
        return word

    # --- Step 1a: plurals --------------------------------------------------
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # --- Step 1b: -eed / -ed / -ing ---------------------------------------
    step1b_applied = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed"):
        if _contains_vowel(word[:-2]):
            word = word[:-2]
            step1b_applied = True
    elif word.endswith("ing"):
        if _contains_vowel(word[:-3]):
            word = word[:-3]
            step1b_applied = True

    if step1b_applied:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"

    # --- Step 1c: terminal y -> i -----------------------------------------
    if word.endswith("y") and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    # --- Steps 2 and 3: derivational suffixes ------------------------------
    for suffix, replacement in _STEP2:
        result = _replace(word, suffix, replacement, 0)
        if result is not None:
            word = result
            break

    for suffix, replacement in _STEP3:
        result = _replace(word, suffix, replacement, 0)
        if result is not None:
            word = result
            break

    # --- Step 4: strip residual suffixes -----------------------------------
    for suffix in _STEP4:
        if word.endswith(suffix):
            stem_part = word[: len(word) - len(suffix)]
            if _measure(stem_part) > 1:
                word = stem_part
            break
    else:
        if word.endswith("ion"):
            stem_part = word[:-3]
            if _measure(stem_part) > 1 and stem_part.endswith(("s", "t")):
                word = stem_part

    # --- Steps 5a and 5b: tidy up ------------------------------------------
    if word.endswith("e"):
        trimmed = word[:-1]
        measure = _measure(trimmed)
        if measure > 1 or (measure == 1 and not _ends_cvc(trimmed)):
            word = trimmed

    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]

    return word


# ---------------------------------------------------------------------------
# Clinical lemma layer, applied on top of (never inside) the Porter stemmer.
# ---------------------------------------------------------------------------

# Irregular verbs/nouns the suffix-stripping rules cannot reach, British vs
# American spellings, and pairs Porter splits apart that should land on one
# form for lexicon lookup (e.g. "itching" -> itch but "itchy" -> itchi).
_CLINICAL_LEMMA = {
    "aching": "ache", "ached": "ache", "aches": "ache", "ache": "ache",
    "achy": "ache", "achey": "ache",
    "nauseous": "nausea", "nauseated": "nausea", "nauseating": "nausea",
    "diarrhoea": "diarrhea", "diarrhoeal": "diarrhea", "diarrheal": "diarrhea",
    "itchy": "itch", "itchiness": "itch",
    "dizzy": "dizzi", "dizziness": "dizzi", "giddy": "dizzi",
    "swollen": "swell", "swelled": "swell",
    "bled": "bleed", "bleeding": "bleed", "bleeds": "bleed",
    "breathlessness": "breathless",
    "feet": "foot", "teeth": "tooth", "worse": "worsen", "worsening": "worsen",
    "faeces": "feces", "oedema": "edema", "diarrohea": "diarrhea",
    "vomitting": "vomit", "paining": "pain",
}


def lemma(word: str) -> str:
    """
    Normalize a surface form to its lookup key: a clinical override when one
    exists, otherwise the Porter stem.
    """
    word = word.lower()
    if word in _CLINICAL_LEMMA:
        return _CLINICAL_LEMMA[word]
    return stem(word)
