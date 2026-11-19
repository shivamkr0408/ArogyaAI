"""
Temporal expression recognition and normalization (a TIMEX-style layer).

Two jobs:

1. **Duration / onset** -- "since 3 days", "for the past two weeks",
   "yesterday", "3 din se" are all resolved to a number of days, so the
   follow-up engine can stop asking "how long have you had this?" when the
   user already said it in their first sentence.
2. **Frequency and trend** -- "on and off", "getting worse", "constant".
   Trend feeds the triage score, which previously only learned it by asking.

Normalized durations are also bucketed onto the exact option strings the
clinical-slot questionnaire uses, so an extracted value is indistinguishable
from a button the user tapped.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Number words, including the romanized Hindi numerals that show up in
# code-mixed input ("do din se" = "for two days").
NUMBER_WORDS: dict[str, float] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "half": 0.5, "couple": 2, "few": 3, "several": 4, "many": 5,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
    "panch": 5, "chhe": 6, "che": 6, "saat": 7, "aath": 8, "nau": 9,
    "das": 10, "dedh": 1.5,
}

# Unit -> days.
UNITS: dict[str, float] = {
    "minute": 1 / 1440, "minutes": 1 / 1440, "min": 1 / 1440, "mins": 1 / 1440,
    "hour": 1 / 24, "hours": 1 / 24, "hr": 1 / 24, "hrs": 1 / 24,
    "ghante": 1 / 24, "ghanta": 1 / 24,
    "day": 1, "days": 1, "din": 1, "dino": 1, "dinon": 1,
    "night": 1, "nights": 1, "raat": 1,
    "week": 7, "weeks": 7, "hafta": 7, "hafte": 7, "hafton": 7, "saptah": 7,
    "fortnight": 14,
    "month": 30, "months": 30, "mahina": 30, "mahine": 30, "mahinon": 30,
    "year": 365, "years": 365, "saal": 365, "varsh": 365, "baras": 365,
}

# Fixed relative references.
RELATIVE: dict[str, float] = {
    "today": 0.5, "aaj": 0.5, "this morning": 0.5, "this afternoon": 0.5,
    "tonight": 0.5, "just now": 0.04, "right now": 0.04, "a moment ago": 0.04,
    "last night": 1, "yesterday": 1, "kal": 1, "kal se": 1,
    "day before yesterday": 2, "parso": 2, "parson": 2,
    "last week": 7, "past week": 7, "pichle hafte": 7,
    "last month": 30, "past month": 30, "pichle mahine": 30,
    "last year": 365, "pichle saal": 365,
}

FREQUENCY: dict[str, str] = {
    "every day": "daily", "everyday": "daily", "daily": "daily",
    "roz": "daily", "har roz": "daily", "all the time": "continuous",
    "constantly": "continuous", "constant": "continuous",
    "nonstop": "continuous", "non stop": "continuous",
    "continuous": "continuous", "continuously": "continuous",
    "on and off": "intermittent", "off and on": "intermittent",
    "on off": "intermittent", "intermittent": "intermittent",
    "intermittently": "intermittent", "comes and goes": "intermittent",
    "now and then": "occasional", "occasionally": "occasional",
    "sometimes": "occasional", "kabhi kabhi": "occasional",
    "once a day": "daily", "twice a day": "twice_daily",
    "at night": "nocturnal", "only at night": "nocturnal",
    "in the morning": "morning", "every morning": "morning",
    "after eating": "postprandial", "after meals": "postprandial",
}

TREND: dict[str, str] = {
    "getting worse": "worsening", "got worse": "worsening",
    "becoming worse": "worsening", "worsening": "worsening",
    "worse than before": "worsening", "worse now": "worsening",
    "increasing": "worsening", "badh raha": "worsening",
    "zyada ho raha": "worsening", "deteriorating": "worsening",
    "getting better": "improving", "improving": "improving",
    "better now": "improving", "improved": "improving",
    "kam ho raha": "improving", "subsiding": "improving",
    "recovering": "improving",
    "same": "unchanged", "about the same": "unchanged",
    "no change": "unchanged", "unchanged": "unchanged",
    "stable": "unchanged", "waisa hi": "unchanged",
}


# The slot options the follow-up questionnaire offers, so an extracted
# duration can be injected as if the user had tapped the button.
DURATION_BUCKETS: list[tuple[float, str]] = [
    (1.0, "Less than a day"),
    (3.9, "1-3 days"),
    (13.9, "About a week"),
    (float("inf"), "More than 2 weeks"),
]


@dataclass
class TemporalExpression:
    kind: str  # duration | frequency | trend
    text: str
    char_start: int
    char_end: int
    value: Optional[float] = None  # normalized to days, for kind == duration
    unit: Optional[str] = None
    normalized: Optional[str] = None  # frequency/trend label, or bucket label

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "days": round(self.value, 3) if self.value is not None else None,
            "unit": self.unit,
            "normalized": self.normalized,
        }


_NUM = r"\d+(?:\.\d+)?|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
_UNIT = "|".join(sorted(UNITS, key=len, reverse=True))

# "since 3 days", "for the past two weeks", "3 din se", "about 2 hours"
_DURATION_RE = re.compile(
    r"\b(?:(?:since|for|from|past|last|about|around|nearly|almost|over)\s+)?"
    r"(?:the\s+)?(?:past\s+|last\s+)?"
    r"(?P<num>" + _NUM + r")\s*"
    # "a couple of days", "a few of weeks"
    r"(?:of\s+)?"
    r"(?P<unit>" + _UNIT + r")\b"
    r"(?:\s+(?:se|ago|back))?",
    re.IGNORECASE,
)

_RELATIVE_RE = re.compile(
    r"\b(?P<prep>since|from|starting|for|over)?\s*(?P<rel>"
    + "|".join(sorted(RELATIVE, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# These name a point in the past, not a duration. "I had dengue last year" is
# a history item; only "since last year" frames it as an ongoing duration.
# Bare "yesterday" / "last night" stay allowed, since those really are onset
# markers for a current complaint.
_NEEDS_PREPOSITION = {
    "last week", "past week", "pichle hafte",
    "last month", "past month", "pichle mahine",
    "last year", "pichle saal",
    "day before yesterday", "parso", "parson",
}

_FREQUENCY_RE = re.compile(
    r"\b(?P<freq>" + "|".join(sorted(FREQUENCY, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_TREND_RE = re.compile(
    r"\b(?P<trend>" + "|".join(sorted(TREND, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _to_number(raw: str) -> Optional[float]:
    raw = raw.strip().lower()
    try:
        return float(raw)
    except ValueError:
        return NUMBER_WORDS.get(raw)


def bucket_for_days(days: float) -> str:
    """Map a normalized duration onto a questionnaire option string."""
    for threshold, label in DURATION_BUCKETS:
        if days < threshold:
            return label
    return DURATION_BUCKETS[-1][1]


def extract(text: str) -> list[TemporalExpression]:
    """
    Find every temporal expression in `text`.

    Overlapping matches are resolved in favour of the longest span, so
    "since the past two weeks" yields one expression rather than three.
    """
    found: list[TemporalExpression] = []

    for match in _DURATION_RE.finditer(text):
        number = _to_number(match.group("num"))
        if number is None:
            continue
        unit_raw = match.group("unit").lower()
        per_day = UNITS.get(unit_raw)
        if per_day is None:
            continue
        days = number * per_day
        found.append(
            TemporalExpression(
                kind="duration",
                text=match.group(0).strip(),
                char_start=match.start(),
                char_end=match.end(),
                value=days,
                unit=unit_raw,
                normalized=bucket_for_days(days),
            )
        )

    for match in _RELATIVE_RE.finditer(text):
        label = match.group("rel").lower()
        days = RELATIVE.get(label)
        if days is None:
            continue
        if label in _NEEDS_PREPOSITION and not match.group("prep"):
            continue
        found.append(
            TemporalExpression(
                kind="duration",
                text=match.group(0).strip(),
                char_start=match.start(),
                char_end=match.end(),
                value=days,
                unit="relative",
                normalized=bucket_for_days(days),
            )
        )

    for match in _FREQUENCY_RE.finditer(text):
        label = match.group("freq").lower()
        found.append(
            TemporalExpression(
                kind="frequency",
                text=match.group(0).strip(),
                char_start=match.start(),
                char_end=match.end(),
                normalized=FREQUENCY[label],
            )
        )

    for match in _TREND_RE.finditer(text):
        label = match.group("trend").lower()
        found.append(
            TemporalExpression(
                kind="trend",
                text=match.group(0).strip(),
                char_start=match.start(),
                char_end=match.end(),
                normalized=TREND[label],
            )
        )

    # Longest span wins on overlap, within the same kind.
    found.sort(key=lambda e: (-(e.char_end - e.char_start), e.char_start))
    accepted: list[TemporalExpression] = []
    claimed: dict[str, set[int]] = {}
    for expr in found:
        taken = claimed.setdefault(expr.kind, set())
        span = range(expr.char_start, expr.char_end)
        if any(i in taken for i in span):
            continue
        accepted.append(expr)
        taken.update(span)

    accepted.sort(key=lambda e: e.char_start)
    return accepted


def primary_duration(expressions: list[TemporalExpression]) -> Optional[TemporalExpression]:
    """The longest-standing duration mentioned, which is the clinical one."""
    durations = [e for e in expressions if e.kind == "duration" and e.value is not None]
    if not durations:
        return None
    return max(durations, key=lambda e: e.value or 0)


def first_of(expressions: list[TemporalExpression], kind: str) -> Optional[TemporalExpression]:
    for expr in expressions:
        if expr.kind == kind:
            return expr
    return None
