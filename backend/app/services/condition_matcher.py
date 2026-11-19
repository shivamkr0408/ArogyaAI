"""
Map what a person calls their condition onto one of the 41 model labels.

People do not type "Dimorphic hemmorhoids(piles)". They type "piles",
"bawaseer", "sugar", "high BP", "TB". This resolves those through four
strategies, tried in order of how much they can be trusted:

    1. exact      - the name or a curated alias, after normalising
    2. substring  - an alias appearing inside a longer sentence
    3. fuzzy      - difflib ratio, for typos ("diabetis", "tyfoid")
    4. embedding  - MiniLM cosine similarity, for paraphrases, but only
                    when the retriever has finished warming up

Everything above `AMBIGUOUS_GAP` of the runner-up is returned as a
"did you mean" list rather than guessed at, and anything below
`MIN_CONFIDENCE` is reported as no match at all. Saying "I don't cover
that" is a valid and safer answer than forcing a complaint into the
nearest of 41 labels.

No stdlib-external dependency: difflib ships with Python, and the
embedding model is the one the retriever already loads.
"""

import difflib
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

FACTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "rag", "condition_facts.json"
)

# A fuzzy score below this is not reported as a match at all.
MIN_CONFIDENCE = 0.75
# If the runner-up is within this of the leader, ask instead of guessing.
AMBIGUOUS_GAP = 0.08

_facts: list[dict] = []
_by_id: dict[str, dict] = {}
_alias_index: dict[str, str] = {}   # normalised alias -> condition id
_loaded = False


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9ऀ-ॿ\s]", " ", text)   # keep Devanagari
    return re.sub(r"\s+", " ", text).strip()


# Phrases that mean "I already know my diagnosis". Kept here rather than in
# the intent model because they need to be reliable, not learned.
_DECLARATION_PATTERNS = [
    r"\bi (?:have|has|am having|got)\b",
    r"\bi'?m\b.*\bpatient\b",
    r"\bdiagnos(?:ed|is)\b",
    r"\bdoctor (?:said|told|says)\b",
    r"\btold me i have\b",
    r"\bsuffering from\b",
    r"\bmujhe\b.*\bhai\b",          # "mujhe sugar hai"
    r"\bमुझे\b.*\bहै\b",
    r"\bknown case of\b",
    r"\bi am a\b.*\bpatient\b",
]
_DECLARATION_RE = re.compile("|".join(_DECLARATION_PATTERNS), re.I)


def load() -> None:
    global _facts, _by_id, _alias_index, _loaded
    if _loaded:
        return
    path = os.path.abspath(FACTS_PATH)
    if not os.path.exists(path):
        logger.warning("condition_facts.json not found at %s", path)
        _loaded = True
        return
    with open(path, encoding="utf-8") as fh:
        _facts = json.load(fh)
    _by_id = {f["id"]: f for f in _facts}
    _alias_index = {}
    for f in _facts:
        for alias in [f["name"], f["id"].replace("_", " "), *f.get("aliases", [])]:
            key = _norm(alias)
            if key:
                _alias_index.setdefault(key, f["id"])
    _loaded = True
    logger.info("Condition matcher: %d conditions, %d aliases", len(_facts), len(_alias_index))


def is_loaded() -> bool:
    return bool(_facts)


def all_conditions() -> list[dict]:
    load()
    return _facts


def get(condition_id: str) -> Optional[dict]:
    load()
    return _by_id.get(condition_id)


def get_by_name(name: str) -> Optional[dict]:
    load()
    cid = _alias_index.get(_norm(name))
    return _by_id.get(cid) if cid else None


def looks_like_declaration(text: str) -> bool:
    """True when the user is stating a diagnosis rather than a symptom."""
    return bool(_DECLARATION_RE.search(text or ""))


def _embedding_scores(query: str) -> dict[str, float]:
    """Cosine similarity over condition names; empty if the model is cold."""
    try:
        from . import retriever
        model = getattr(retriever, "_model", None)
        if model is None:
            return {}
        import numpy as np

        names = [f"{f['name']}. {', '.join(f.get('aliases', [])[:4])}" for f in _facts]
        mat = model.encode(names)
        vec = model.encode([query])[0]
        mat = np.asarray(mat, dtype=float)
        vec = np.asarray(vec, dtype=float)
        sims = mat @ vec / (
            np.linalg.norm(mat, axis=1) * np.linalg.norm(vec) + 1e-9
        )
        return {_facts[i]["id"]: float(sims[i]) for i in range(len(_facts))}
    except Exception as exc:                      # cold model, no numpy, etc.
        logger.debug("embedding match unavailable: %s", exc)
        return {}


def match(text: str, use_embeddings: bool = True) -> dict:
    """
    Resolve free text to a condition.

    Returns {match, candidates, confidence, strategy, ambiguous}, where
    `match` is a fact sheet or None and `candidates` holds the near-misses
    worth offering as "did you mean" chips.
    """
    load()
    q = _norm(text)
    if not q or not _facts:
        return {"match": None, "candidates": [], "confidence": 0.0,
                "strategy": "none", "ambiguous": False}

    # 1 & 2 -- exact alias, then alias contained in a longer sentence.
    if q in _alias_index:
        return {"match": _by_id[_alias_index[q]], "candidates": [],
                "confidence": 1.0, "strategy": "exact", "ambiguous": False}

    # Two-character aliases ("tb", "bp") are real condition names, so they
    # must be searchable inside a sentence; the word-boundary guard is what
    # keeps them from firing on fragments of other words.
    contained = [
        (alias, cid) for alias, cid in _alias_index.items()
        if len(alias) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q)
    ]
    if contained:
        # Prefer the longest alias: "hepatitis b" must beat "hepatitis".
        alias, cid = max(contained, key=lambda t: len(t[0]))
        return {"match": _by_id[cid], "candidates": [], "confidence": 0.95,
                "strategy": "alias-in-text", "ambiguous": False}

    # 3 -- fuzzy over every alias, keeping the best score per condition.
    scores: dict[str, float] = {}
    for alias, cid in _alias_index.items():
        r = difflib.SequenceMatcher(None, q, alias).ratio()
        if r > scores.get(cid, 0.0):
            scores[cid] = r

    # 4 -- blend in embeddings when available; they rescue paraphrases that
    # share no characters ("my chest grips when I climb stairs").
    strategy = "fuzzy"
    if use_embeddings:
        emb = _embedding_scores(text)
        if emb:
            strategy = "fuzzy+embedding"
            for cid, s in emb.items():
                scores[cid] = max(scores.get(cid, 0.0), 0.60 * s + 0.40 * scores.get(cid, 0.0))

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return {"match": None, "candidates": [], "confidence": 0.0,
                "strategy": "none", "ambiguous": False}

    top_id, top_score = ranked[0]
    near = [_by_id[cid] for cid, s in ranked[:4]
            if top_score - s <= AMBIGUOUS_GAP and s >= MIN_CONFIDENCE - 0.10]

    if top_score < MIN_CONFIDENCE:
        # Not confident enough to name anything. Offer the closest few so the
        # user can correct us, but do not claim a match.
        return {"match": None, "candidates": [_by_id[cid] for cid, _ in ranked[:3]],
                "confidence": round(top_score, 3), "strategy": strategy,
                "ambiguous": False}

    ambiguous = len(near) > 1
    return {
        "match": None if ambiguous else _by_id[top_id],
        "candidates": near if ambiguous else [],
        "confidence": round(top_score, 3),
        "strategy": strategy,
        "ambiguous": ambiguous,
    }
