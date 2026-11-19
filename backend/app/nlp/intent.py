"""
Intent classification for the dialogue manager.

The state machine used to assume every user turn was a symptom description.
That is wrong the moment a result is on screen: "what should I eat?" reset
the consultation and came back with "I didn't catch a specific symptom
there", because the message contains no symptom. Classifying the *intent*
first lets the orchestrator answer from data it already has.

Model: TF-IDF over word (1-2) and character (3-5) n-grams, unioned, into a
multinomial logistic regression. Character n-grams carry most of the weight
on this kind of input -- short, misspelled, code-mixed -- because they
degrade gracefully where whole-word features miss entirely.

High-precision regex rules run *before* the model for the few intents where
a mistake is expensive (a bare "no" answering a yes/no question, or someone
asking for emergency help).
"""

import logging
import os
import re
from functools import lru_cache

from . import intent_corpus as _corpus
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "ml", "model", "intent_clf.joblib"
)

INTENTS = [
    "greeting",
    "describe_symptom",
    "affirm",
    "deny",
    "ask_diet",
    "ask_doctor",
    "ask_alternatives",
    "ask_medicine",
    "ask_prognosis",
    "thanks",
    "out_of_scope",
]

# Human-readable descriptions, surfaced in the NLP Lab panel.
INTENT_LABELS = {
    "greeting": "Greeting / conversation opener",
    "describe_symptom": "Describing a symptom",
    "affirm": "Answering yes",
    "deny": "Answering no",
    "ask_diet": "Asking what to eat or avoid",
    "ask_doctor": "Asking whether to see a doctor",
    "ask_alternatives": "Asking about other possible conditions",
    "ask_medicine": "Asking about medication",
    "ask_prognosis": "Asking how long recovery takes",
    "thanks": "Expressing thanks / closing",
    "out_of_scope": "Outside the scope of the assistant",
}


# The corpus lives in intent_corpus.py; see the sizing note there.
TRAINING_EXAMPLES = _corpus.TRAINING_EXAMPLES


# ---------------------------------------------------------------------------
# High-precision rules, applied before the statistical model.
# ---------------------------------------------------------------------------

_AFFIRM_RE = re.compile(
    r"^(y|ya|yes|yeah|yep|yup|sure|ok|okay|correct|right|true|haan|han|ji|"
    r"haan ji|bilkul|a little|a little bit|a bit|sometimes|yes please)"
    r"[\s.!,]*$",
    re.IGNORECASE,
)
_DENY_RE = re.compile(
    r"^(n|no|nope|nah|never|none|nothing|negative|not really|no way|"
    r"nahi|nahin|bilkul nahi|no thanks|nothing else|no other symptoms|"
    r"not at all|no i dont|no i do not)[\s.!,]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^(thanks|thank you|thanks a lot|thank u|ty|dhanyawad|shukriya|bye|"
    r"goodbye|good bye|ok thanks|okay thanks)[\s.!,]*$",
    re.IGNORECASE,
)


def _rule_intent(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    if _AFFIRM_RE.match(stripped):
        return "affirm"
    if _DENY_RE.match(stripped):
        return "deny"
    if _THANKS_RE.match(stripped):
        return "thanks"
    return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_pipeline():
    """Word + character n-gram TF-IDF into multinomial logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=8.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(examples: Optional[list[tuple[str, str]]] = None):
    """Fit the intent pipeline on the seed corpus."""
    examples = examples or TRAINING_EXAMPLES
    texts = [t for t, _ in examples]
    labels = [l for _, l in examples]
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    return pipeline


def train_and_save(path: str = MODEL_PATH):
    import joblib

    pipeline = train()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    return pipeline


@lru_cache(maxsize=1)
def _get_model():
    """
    Load the saved pipeline, training one in memory if the artifact is
    missing. Fitting this corpus takes well under a second, so a fresh
    checkout still gets intent classification without a build step.
    """
    try:
        import joblib

        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.warning("Could not load intent model, retraining: %s", exc)
    try:
        return train()
    except Exception as exc:
        logger.warning("Intent model unavailable: %s", exc)
        return None


def classify(text: str) -> dict:
    """
    Return the predicted intent with its confidence and the full posterior.

    `source` says whether a rule or the model decided, which the NLP Lab
    panel displays.
    """
    text = (text or "").strip()
    if not text:
        return {
            "intent": "out_of_scope",
            "confidence": 0.0,
            "source": "empty",
            "scores": {},
        }

    ruled = _rule_intent(text)
    if ruled:
        return {
            "intent": ruled,
            "confidence": 1.0,
            "source": "rule",
            "scores": {ruled: 1.0},
        }

    model = _get_model()
    if model is None:
        return {
            "intent": "describe_symptom",
            "confidence": 0.0,
            "source": "fallback",
            "scores": {},
        }

    try:
        probabilities = model.predict_proba([text])[0]
        classes = list(model.classes_)
        ranked = sorted(zip(classes, probabilities), key=lambda kv: -kv[1])
        return {
            "intent": ranked[0][0],
            "confidence": float(ranked[0][1]),
            "source": "model",
            "scores": {c: round(float(p), 4) for c, p in ranked[:5]},
        }
    except Exception as exc:
        logger.warning("Intent classification failed: %s", exc)
        return {
            "intent": "describe_symptom",
            "confidence": 0.0,
            "source": "error",
            "scores": {},
        }


def is_question_about_result(intent: str) -> bool:
    """Intents that should be answered from the existing assessment."""
    return intent in {
        "ask_diet",
        "ask_doctor",
        "ask_alternatives",
        "ask_medicine",
        "ask_prognosis",
    }
