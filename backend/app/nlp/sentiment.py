"""
Distress scoring -- a small, auditable lexicon rather than a black box.

A health chatbot that answers an anxious "I'm really scared, is this cancer?"
in the same flat register as "mild cough, 2 days" reads badly. This module
scores emotional distress so the orchestrator can lead with one sentence of
acknowledgement before the clinical content.

The approach is a weighted lexicon with negation and intensifier handling
(the same shape as VADER, scaled down to this domain). It is deliberately
not a trained classifier: there is no labelled distress corpus for this
project, and an interpretable lexicon is honest about what it is doing.
"""

import re
from dataclasses import dataclass

# Weighted distress terms. Positive weight = more distress.
DISTRESS_LEXICON: dict[str, float] = {
    # fear / catastrophizing
    "scared": 2.0, "terrified": 2.5, "frightened": 2.0, "afraid": 1.8,
    "panicking": 2.5, "panic": 2.0, "freaking": 2.2, "dread": 2.0,
    "cancer": 1.5, "dying": 2.5, "die": 2.2, "death": 2.0, "fatal": 2.0,
    "serious": 1.0, "emergency": 1.8, "nightmare": 1.8,
    # worry
    "worried": 1.5, "worry": 1.5, "anxious": 1.5, "anxiety": 1.3,
    "stressed": 1.3, "tense": 1.0, "nervous": 1.3, "uneasy": 1.0,
    "overthinking": 1.3, "restless": 1.0,
    "pareshan": 1.5, "ghabrahat": 1.8, "darr": 1.8, "tension": 1.3,
    # suffering intensity
    "unbearable": 2.2, "excruciating": 2.2, "agony": 2.2, "agonizing": 2.2,
    "torture": 2.0, "suffering": 1.8, "miserable": 1.8, "awful": 1.5,
    "horrible": 1.5, "terrible": 1.5, "unbearably": 2.0,
    "cant take": 2.0, "cant cope": 2.2, "cant handle": 2.0,
    "desperate": 2.0, "helpless": 2.0, "hopeless": 2.2,
    # urgency
    "urgent": 1.5, "urgently": 1.5, "immediately": 1.2, "right now": 1.0,
    "please help": 1.8, "help me": 1.8,
    # low mood
    "depressed": 1.8, "crying": 1.5, "alone": 1.0, "exhausted": 1.0,
    "giving up": 2.2, "no point": 2.0,
}

# Distress words that are commentary *about* the consultation rather than a
# reportable symptom. A lone one of these must not set a clinical feature:
# "I'm really worried, I have a fever" reports fever, not anxiety.
#
# Deliberately excludes words that are themselves clinical complaints --
# "anxious", "restless", "depressed", "insomnia", "exhausted" -- because
# "feeling anxious and restless all day" is a genuine presentation.
META_DISTRESS_WORDS: frozenset[str] = frozenset("""
worried worry worrying scared afraid frightened terrified
panicking panic freaking dread
dying die death fatal cancer serious emergency nightmare
urgent urgently immediately
desperate helpless hopeless miserable
awful horrible terrible unbearable unbearably excruciating agony agonizing
torture suffering
darr pareshan tension ghabrahat
""".split())

INTENSIFIERS: dict[str, float] = {
    "very": 1.4, "really": 1.4, "so": 1.3, "extremely": 1.7, "incredibly": 1.6,
    "super": 1.4, "too": 1.3, "absolutely": 1.5, "completely": 1.5,
    "totally": 1.4, "unbelievably": 1.6, "insanely": 1.6, "bahut": 1.4,
    "bohot": 1.4, "zyada": 1.3,
}

DAMPENERS: dict[str, float] = {
    "slightly": 0.6, "little": 0.6, "bit": 0.6, "somewhat": 0.7,
    "mildly": 0.6, "kind of": 0.7, "sort of": 0.7, "not very": 0.5,
    "thoda": 0.6,
}

NEGATORS = {"not", "no", "never", "dont", "doesnt", "isnt", "nahi", "without"}

_WORD_RE = re.compile(r"[a-z]+")

# Thresholds chosen so a single hedged "a bit worried" stays "calm" and does
# not trigger an empathetic preamble on every turn.
_ELEVATED = 1.6
_HIGH = 3.2


@dataclass
class DistressScore:
    score: float
    level: str  # calm | elevated | high
    triggers: list[str]

    @property
    def needs_acknowledgement(self) -> bool:
        return self.level in ("elevated", "high")

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "triggers": self.triggers,
            "needs_acknowledgement": self.needs_acknowledgement,
        }


def score(text: str) -> DistressScore:
    """
    Score the distress expressed in `text`.

    Each matched term is scaled by the intensifier or dampener immediately
    preceding it, and zeroed by a negator within two tokens ("I'm not
    worried" must not score as worry).
    """
    if not text:
        return DistressScore(0.0, "calm", [])

    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    total = 0.0
    triggers: list[str] = []

    # Multi-word entries first, so "cant take" is not double counted.
    for phrase, weight in DISTRESS_LEXICON.items():
        if " " not in phrase:
            continue
        if phrase in lowered:
            total += weight
            triggers.append(phrase)

    for i, word in enumerate(words):
        weight = DISTRESS_LEXICON.get(word)
        if weight is None:
            continue

        window = words[max(0, i - 2) : i]
        if any(w in NEGATORS for w in window):
            continue

        multiplier = 1.0
        if i > 0:
            multiplier *= INTENSIFIERS.get(words[i - 1], 1.0)
            multiplier *= DAMPENERS.get(words[i - 1], 1.0)

        total += weight * multiplier
        triggers.append(word)

    # Repeated punctuation and shouting are genuine signals in typed text.
    if re.search(r"[!?]{2,}", text):
        total += 0.6
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        total += 0.8

    level = "calm"
    if total >= _HIGH:
        level = "high"
    elif total >= _ELEVATED:
        level = "elevated"

    return DistressScore(total, level, triggers)


# Opening lines used when distress is detected. Deliberately short: the
# clinical content is what the user came for.
ACKNOWLEDGEMENTS = {
    "elevated": "I can hear this is worrying you, and that's completely understandable. Let's work through it together.",
    "high": "I'm sorry you're going through this - that sounds really distressing. Let's take it step by step, and if anything here points to urgent care I'll tell you straight away.",
}


def acknowledgement(level: str) -> str:
    return ACKNOWLEDGEMENTS.get(level, "")
