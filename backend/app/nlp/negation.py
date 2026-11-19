"""
Assertion detection: NegEx (Chapman et al., 2001) extended to ConText
(Harkema et al., 2009).

The old extractor approximated negation by looking at the 12 characters
before a matched phrase. That fails on everything real:

    "I don't have any fever"          -> trigger is 10+ chars away
    "no cough but I do have a fever"  -> kills the fever too
    "my father has diabetes"          -> attributed to the patient
    "I had dengue last year"          -> treated as a current symptom
    "if I get a fever should I worry" -> a hypothetical becomes a symptom

ConText models three independent axes instead of one boolean:

* **Negation**   -- affirmed / negated
* **Temporality** -- recent / historical / hypothetical
* **Experiencer** -- patient / other

A symptom only reaches the classifier's feature vector when it is
*affirmed*, *recent* and about the *patient*. Each axis is decided by
matching cue phrases and propagating a bounded scope that stops at
termination terms, which is what makes "no cough but I do have a fever"
come out right.
"""

from dataclasses import dataclass, field
from typing import Optional

from .tokenizer import Sentence, Token

# NegEx's published window is six tokens; conversational text carries more
# filler ("I really do not have any kind of fever"), so this is slightly
# wider. Scope still stops early at any termination term.
DEFAULT_WINDOW = 7


# ---------------------------------------------------------------------------
# Cue lexicons. "forward" cues scope to their right, "backward" to their left.
# ---------------------------------------------------------------------------

# Matched first; they look like negations but are not, so they block a
# negation cue from firing at the same position.
PSEUDO_NEGATION = [
    "no increase", "no change", "no improvement", "not only", "not just",
    "no further", "not necessarily", "cannot be ruled out", "can not be ruled out",
    "no doubt", "not sure", "no question", "without difficulty", "without a doubt",
    "not to mention", "no wonder", "none the less", "no matter",
    # "no X" constructions that *describe* a symptom rather than deny one.
    # Without these, "no appetite" negates appetite and the loss_of_appetite
    # feature is dropped -- the exact opposite of what the user said.
    "no appetite", "no energy", "no strength", "no stamina", "no sleep",
    "no taste", "no smell", "no relief", "no urine", "no motion",
    "no bowel movement", "no interest",
]

PRE_NEGATION = [
    "no", "not", "none", "never", "without", "denies", "denied", "deny",
    "no sign of", "no signs of", "no evidence of", "no complaints of",
    "no complaint of", "absence of", "absent", "free of", "clear of",
    "rules out", "ruled out", "rule out", "negative for", "did not have",
    "do not have", "does not have", "have not had", "has not had",
    "had no", "there is no", "there was no", "i have no", "nothing",
    "no history of", "not experiencing", "not having", "no such",
    "apart from", "other than", "except for", "besides",
]

POST_NEGATION = [
    "free", "ruled out", "is ruled out", "are ruled out", "absent",
    "negative", "resolved", "gone now", "stopped", "cleared up",
    "not present", "no more",
]

# Scope stops here, so a contrast after a negation is not swallowed.
#
# Note that the coordinators "and" / "or" are deliberately NOT terminators,
# matching the published NegEx list. They coordinate items of the *same*
# polarity: "denies vomiting and nausea" negates both, so terminating on
# "and" would let nausea through as a reported symptom.
TERMINATION = [
    "but", "however", "although", "though", "except", "yet", "aside from",
    "apart from", "other than", "still", "nevertheless", "nonetheless",
    "whereas", "while", "unfortunately", "instead", "rather",
]

HYPOTHETICAL = [
    "if", "in case", "in case of", "should i", "unless", "whenever",
    "as needed", "what if", "could it be", "could this be", "might it be",
    "do i have", "am i going to", "will i get", "worried about",
    "worried i have", "afraid of", "afraid i have", "scared of",
    "scared i have", "scared that", "anxious about", "hope i do not have",
    "hope its not", "is it possible", "does this mean", "suppose",
    "in future", "to avoid", "prevent", "risk of", "chance of",
]

# Historical cues come in both directions. "history of asthma" scopes right;
# "I had dengue last year" scopes left over the entity that precedes it.
HISTORICAL_PRE = [
    "in the past", "previously", "used to", "history of", "past history",
    "when i was a child", "as a child", "since childhood", "long ago",
    "already recovered", "recovered from", "after recovering", "before this",
    "earlier this year", "had an episode of",
]

# Deliberately disjoint from HISTORICAL_PRE: a phrase listed in both would
# collapse in the length-keyed lookup and only one direction would survive.
HISTORICAL_POST = [
    "last year", "last month", "last week", "years ago", "months ago",
    "weeks ago", "a year ago", "a month ago", "back then",
    "two years back", "a while back", "when i was younger", "long back",
]

# A historical cue directly after one of these is a *duration*, not a past
# episode: "fever since last week" is ongoing, not resolved. temporal.py
# picks those up instead.
DURATION_PREFIXES = {"since", "for", "from", "past", "last", "over", "about"}

# Experiencer cues -- the symptom belongs to someone else.
FAMILY = [
    "my mother", "my father", "my mom", "my dad", "my mum", "my brother",
    "my sister", "my wife", "my husband",
    "my friend", "my colleague",
    "my grandmother", "my grandfather", "my uncle", "my aunt", "my cousin",
    "my neighbour", "my neighbor", "my partner", "my roommate",
    "family history", "someone at home", "someone in my family",
    "my family member", "a family member", "people at home", "my parents",
    "his", "her", "their",
]

# Dependants are a separate axis value, not family history.
#
# In a clinical note "my son has a fever" is somebody else's symptom. In a
# consumer symptom checker it is usually the whole reason the person opened
# the app: a parent is asking ON BEHALF of the child, who is the patient.
# Collapsing the two meant "my baby has a fever" extracted nothing at all,
# which silently disabled the infant safety rules. Keeping a distinct value
# preserves the distinction the NLP layer is making while letting the
# symptom reach the classifier.
DEPENDENT = [
    "my son", "my daughter", "my child", "my kid", "my baby", "my infant",
    "my toddler", "my little one", "my grandson", "my granddaughter",
    "mera beta", "meri beti", "mera bachcha",
]

UNCERTAINTY = [
    "maybe", "may be", "might", "might be", "possibly", "possible",
    "perhaps", "not sure", "unsure", "i think", "i guess", "i feel like",
    "could be", "probably", "sort of", "kind of", "a bit like",
    "seems like", "feels like", "i wonder", "suspect", "apparently",
    "somewhat", "slightly unsure",
]


_CUE_SETS: list[tuple[str, str, list[str]]] = [
    # (axis-kind, direction, phrases)
    ("pseudo", "none", PSEUDO_NEGATION),
    ("negation", "forward", PRE_NEGATION),
    ("negation", "backward", POST_NEGATION),
    ("termination", "none", TERMINATION),
    ("hypothetical", "forward", HYPOTHETICAL),
    ("historical", "forward", HISTORICAL_PRE),
    ("historical", "backward", HISTORICAL_POST),
    ("family", "forward", FAMILY),
    ("dependent", "forward", DEPENDENT),
    ("uncertainty", "forward", UNCERTAINTY),
]


@dataclass
class Cue:
    """A matched cue phrase with the token range its scope governs."""

    text: str
    kind: str  # pseudo | negation | termination | hypothetical | historical | family | dependent | uncertainty
    direction: str  # forward | backward | none
    token_start: int  # index within the sentence token list
    token_end: int  # exclusive
    char_start: int
    char_end: int
    scope_start: int = 0  # token indices (within sentence) the cue governs
    scope_end: int = 0

    def covers(self, token_index: int) -> bool:
        return self.scope_start <= token_index < self.scope_end

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "direction": self.direction,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class Assertion:
    """The three ConText axes, plus a single derived label for display."""

    negation: str = "affirmed"  # affirmed | negated
    temporality: str = "recent"  # recent | historical | hypothetical
    experiencer: str = "patient"  # patient | dependent | other
    certainty: str = "certain"  # certain | uncertain
    cues: list[Cue] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.negation == "negated":
            return "negated"
        if self.experiencer == "other":
            return "family"
        if self.experiencer == "dependent":
            return "dependent"
        if self.temporality == "hypothetical":
            return "hypothetical"
        if self.temporality == "historical":
            return "historical"
        if self.certainty == "uncertain":
            return "uncertain"
        return "affirmed"

    @property
    def is_clinically_present(self) -> bool:
        """
        Whether this mention should reach the classifier.

        Uncertainty is deliberately *not* disqualifying -- "I think I have a
        fever" is still a reported symptom, just a hedged one.
        """
        return (
            self.negation == "affirmed"
            and self.temporality == "recent"
            # A dependant ("my baby has a fever") is who the consultation is
            # about, so the mention counts; a third party ("my father has
            # diabetes") is history and does not.
            and self.experiencer in ("patient", "dependent")
        )

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "negation": self.negation,
            "temporality": self.temporality,
            "experiencer": self.experiencer,
            "certainty": self.certainty,
            "is_clinically_present": self.is_clinically_present,
            "cues": [c.as_dict() for c in self.cues],
        }


def _phrase_tables() -> dict[str, list[tuple[int, str, str, str]]]:
    """Group cue phrases by length so longest-match can run first."""
    table: dict[int, list[tuple[str, str, str]]] = {}
    for kind, direction, phrases in _CUE_SETS:
        for phrase in phrases:
            n = len(phrase.split())
            table.setdefault(n, []).append((phrase, kind, direction))
    return table


_PHRASES_BY_LENGTH = _phrase_tables()
_MAX_CUE_LEN = max(_PHRASES_BY_LENGTH) if _PHRASES_BY_LENGTH else 1

_DETERMINERS = {"the", "a", "an", "this", "these"}


def _is_contiguous(tokens: list[Token], start: int, length: int) -> bool:
    """
    True when the tokens are separated only by single spaces.

    Punctuation is not emitted as tokens, so a gap wider than one character
    between two words means something was dropped between them -- a comma, a
    dash, a clause boundary. A cue phrase may not span that.
    """
    for offset in range(length - 1):
        left = tokens[start + offset]
        right = tokens[start + offset + 1]
        if right.start - left.end > 1:
            return False
    return True


def _preceded_by_duration(lowered: list[str], index: int) -> bool:
    """
    True when the token at `index` is introduced by a duration preposition,
    allowing one determiner in between ("for the last month").
    """
    cursor = index - 1
    skipped = 0
    while cursor >= 0 and skipped < 2:
        word = lowered[cursor]
        if word in DURATION_PREFIXES:
            return True
        if word not in _DETERMINERS:
            return False
        cursor -= 1
        skipped += 1
    return False


def find_cues(tokens: list[Token], window: int = DEFAULT_WINDOW) -> list[Cue]:
    """
    Find every cue in one sentence's token list and resolve its scope.

    Longest phrases match first, pseudo-negations consume their tokens so a
    plain negation cannot fire inside them, and scopes are clipped at the
    nearest termination term.
    """
    n_tokens = len(tokens)
    lowered = [t.lower for t in tokens]
    consumed: set[int] = set()
    found: list[Cue] = []

    for length in range(min(_MAX_CUE_LEN, n_tokens), 0, -1):
        entries = _PHRASES_BY_LENGTH.get(length)
        if not entries:
            continue
        lookup = {phrase: (kind, direction) for phrase, kind, direction in entries}
        for i in range(n_tokens - length + 1):
            if any((i + k) in consumed for k in range(length)):
                continue
            surface = " ".join(lowered[i : i + length])
            hit = lookup.get(surface)
            if not hit:
                continue
            kind, direction = hit
            # "fever since last week" / "cough for the last month" are
            # durations, not past episodes. Look back past any determiner.
            if kind == "historical" and _preceded_by_duration(lowered, i):
                continue
            # A multi-word cue must be contiguous in the source. Punctuation is
            # not tokenized, so "worried, I have a high fever" would otherwise
            # match the cue "worried i have" straight across the comma and
            # mark the fever hypothetical.
            if length > 1 and not _is_contiguous(tokens, i, length):
                continue
            found.append(
                Cue(
                    text=surface,
                    kind=kind,
                    direction=direction,
                    token_start=i,
                    token_end=i + length,
                    char_start=tokens[i].start,
                    char_end=tokens[i + length - 1].end,
                )
            )
            # Pseudo-negations and multi-word cues block shorter cues inside
            # them ("no change" must stop "no" from negating).
            if kind == "pseudo" or length > 1:
                consumed.update(range(i, i + length))

    found.sort(key=lambda c: c.token_start)
    terminators = [c for c in found if c.kind == "termination"]

    for cue in found:
        if cue.direction == "forward":
            limit = min(n_tokens, cue.token_end + window)
            for term in terminators:
                if cue.token_end <= term.token_start < limit:
                    limit = term.token_start
                    break
            cue.scope_start, cue.scope_end = cue.token_end, limit
        elif cue.direction == "backward":
            limit = max(0, cue.token_start - window)
            for term in reversed(terminators):
                if limit <= term.token_end <= cue.token_start:
                    limit = term.token_end
                    break
            cue.scope_start, cue.scope_end = limit, cue.token_start
        else:
            cue.scope_start, cue.scope_end = cue.token_start, cue.token_end

    return found


def classify(
    sentence: Sentence,
    entity_char_start: int,
    entity_char_end: int,
    cues: Optional[list[Cue]] = None,
    window: int = DEFAULT_WINDOW,
) -> Assertion:
    """
    Decide the assertion status of an entity occupying a character span
    inside `sentence`.

    Each axis is set independently, so "my father never had asthma" comes out
    negated *and* attributed to another experiencer.
    """
    tokens = sentence.tokens
    if cues is None:
        cues = find_cues(tokens, window=window)

    # Token indices the entity occupies.
    covered = [
        idx
        for idx, tok in enumerate(tokens)
        if tok.start < entity_char_end and entity_char_start < tok.end
    ]
    if not covered:
        return Assertion()

    assertion = Assertion()
    applied: list[Cue] = []

    entity_span = set(covered)

    for cue in cues:
        if cue.kind in ("pseudo", "termination"):
            continue
        # A cue sitting *inside* the entity is part of the symptom's own
        # name, not an operator on it: "can not sleep" is insomnia, and
        # "no appetite" is loss of appetite. Without this guard the cue
        # negates the very entity it helps spell.
        if entity_span & set(range(cue.token_start, cue.token_end)):
            continue
        if not any(cue.covers(idx) for idx in covered):
            continue
        applied.append(cue)
        if cue.kind == "negation":
            assertion.negation = "negated"
        elif cue.kind == "hypothetical":
            assertion.temporality = "hypothetical"
        elif cue.kind == "historical":
            # A hypothetical reading wins: "if it comes back like last year".
            if assertion.temporality != "hypothetical":
                assertion.temporality = "historical"
        elif cue.kind == "dependent":
            assertion.experiencer = "dependent"
            assertion.cues.append(cue)
        elif cue.kind == "family":
            assertion.experiencer = "other"
        elif cue.kind == "uncertainty":
            assertion.certainty = "uncertain"

    assertion.cues = applied
    return assertion
