"""
Question answering about a known condition.

Once the user has told us what they have, the useful questions change
shape: "is it contagious?", "can I eat rice?", "how long does it last?".
These are answered from the condition's fact sheet, optionally enriched
with the RAG passages the retriever already indexes, and never invented.

Intent detection is nearest-neighbour over labelled example questions.
Embeddings are used when the retriever's MiniLM model has warmed up, with a
keyword fallback so the feature works on a cold start and with no model at
all -- the same "must work offline" rule the rest of the app follows.

Safety rules enforced here:
  * medicine questions never produce a dose, schedule or brand, only the
    general classes a doctor may consider, plus a route to a professional;
  * every answer is built from fact-sheet fields, so there is no free text
    to hallucinate into;
  * the caller is told which passages were used, so the UI can cite them.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# intent -> example questions. Short, varied, and deliberately including the
# clipped phrasing people actually type.
INTENT_EXAMPLES: dict[str, list[str]] = {
    "overview": ["what is it", "tell me about this condition", "explain this disease",
                 "what does this mean", "ye kya hai"],
    "symptoms": ["what are the symptoms", "what should i feel", "signs of this",
                 "how does it present", "lakshan kya hai"],
    "causes": ["what causes it", "why did i get this", "how did i catch it",
               "reason for this disease", "kaise hua"],
    "diet": ["what should i eat", "diet for this", "what food is good",
             "khana kya khaye", "which foods help"],
    "specific_food": ["can i eat rice", "can i drink milk", "is curd ok",
                      "can i have tea", "is banana allowed", "can i eat eggs"],
    "exercise": ["can i exercise", "is walking ok", "should i rest",
                 "can i go to the gym", "is yoga safe"],
    "ayurveda": ["ayurvedic treatment", "home remedy", "natural remedy",
                 "gharelu upay", "ayurveda for this"],
    "contagious": ["is it contagious", "can it spread", "will my family get it",
                   "is it infectious", "kya ye failta hai"],
    "duration": ["how long does it last", "when will i recover", "recovery time",
                 "how many days", "kitne din lagenge"],
    "complications": ["what can go wrong", "is it dangerous", "complications",
                      "what if untreated", "can it get serious"],
    "prevention": ["how do i prevent it", "how to avoid this", "can it come back",
                   "bachav kaise kare", "prevention tips"],
    "tests": ["what tests do i need", "which test confirms it", "do i need a blood test",
              "what investigation", "jaanch kya hogi"],
    "when_doctor": ["when should i see a doctor", "do i need to go to hospital",
                    "when is it serious", "should i see a specialist",
                    "doctor ke paas kab jaye"],
    "medicines": ["what medicine should i take", "which tablet", "what treatment",
                  "dawa kya le", "is there a cure"],
    "emergency": ["is this an emergency", "should i call an ambulance",
                  "i feel very unwell", "this is getting worse fast"],
}

# Keyword fallback used when embeddings are unavailable.
INTENT_KEYWORDS: dict[str, list[str]] = {
    "specific_food": ["can i eat", "can i drink", "can i have", "is it ok to eat", "allowed to eat"],
    "contagious": ["contagious", "spread", "infectious", "catch it", "failta"],
    "diet": ["diet", "food", "eat", "khana", "nutrition"],
    "exercise": ["exercise", "walk", "gym", "workout", "yoga", "rest"],
    "ayurveda": ["ayurved", "home remedy", "natural", "gharelu", "herbs"],
    "duration": ["how long", "how many days", "recover", "recovery", "kitne din", "last"],
    "complications": ["complication", "dangerous", "serious", "go wrong", "untreated"],
    "prevention": ["prevent", "avoid", "come back", "again", "bachav"],
    "tests": ["test", "investigation", "blood test", "scan", "jaanch", "diagnos"],
    "when_doctor": ["see a doctor", "hospital", "specialist", "when should i"],
    "medicines": ["medicine", "tablet", "dawa", "treatment", "cure", "drug"],
    "emergency": ["emergency", "ambulance", "urgent", "getting worse fast"],
    "symptoms": ["symptom", "sign", "lakshan", "what should i feel"],
    "causes": ["cause", "why did i", "how did i get", "reason", "kaise hua"],
    "overview": ["what is", "tell me about", "explain", "ye kya hai"],
}

_vectors = None
_vector_labels: list[str] = []


def _ensure_vectors():
    """Encode the example bank once, if the embedding model is available."""
    global _vectors, _vector_labels
    if _vectors is not None:
        return _vectors
    try:
        from . import retriever
        model = getattr(retriever, "_model", None)
        if model is None:
            return None
        import numpy as np

        texts, labels = [], []
        for intent, examples in INTENT_EXAMPLES.items():
            for ex in examples:
                texts.append(ex)
                labels.append(intent)
        mat = np.asarray(model.encode(texts), dtype=float)
        _vectors = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        _vector_labels = labels
        return _vectors
    except Exception as exc:
        logger.debug("condition_qa embeddings unavailable: %s", exc)
        return None


def detect_intent(question: str) -> tuple[str, float, str]:
    """Return (intent, confidence, source)."""
    q = (question or "").lower().strip()
    if not q:
        return "overview", 0.0, "default"

    # Keywords first for the handful of intents where a phrase is decisive;
    # "can i eat X" must never be confused with the general diet question.
    for intent in ("specific_food", "emergency", "contagious"):
        for kw in INTENT_KEYWORDS.get(intent, []):
            if kw in q:
                return intent, 0.95, "rule"

    vectors = _ensure_vectors()
    if vectors is not None:
        try:
            from . import retriever
            import numpy as np
            v = np.asarray(retriever._model.encode([q])[0], dtype=float)
            v = v / (np.linalg.norm(v) + 1e-9)
            sims = vectors @ v
            idx = int(sims.argmax())
            return _vector_labels[idx], float(sims[idx]), "embedding"
        except Exception:
            pass

    best, score = "overview", 0.0
    for intent, kws in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in q)
        if hits:
            s = hits / len(kws) + 0.5
            if s > score:
                best, score = intent, s
    return best, min(score, 0.9), "keyword"


def _join(items, limit=6) -> str:
    items = [i for i in (items or []) if i][:limit]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def answer(fact: dict, question: str, intent: Optional[str] = None) -> dict:
    """
    Build an answer about `fact` from its own fields.

    Returns {answer, intent, confidence, source, sections_used, disclaimer}.
    """
    if intent is None:
        intent, conf, source = detect_intent(question)
    else:
        conf, source = 1.0, "explicit"

    name = fact["name"]
    used: list[str] = []

    def take(field, text):
        used.append(field)
        return text

    if intent == "overview":
        body = take("overview", fact["overview"]) or f"{name} is one of the conditions this tool covers."
    elif intent == "symptoms":
        body = take("common_symptoms",
                    f"The symptoms most often associated with {name} are {_join(fact['common_symptoms'])}. "
                    "Not everyone gets all of them, and a partial picture is normal.")
    elif intent == "causes":
        body = take("causes", fact["causes"])
        if fact.get("risk_factors"):
            body += " " + take("risk_factors",
                               f"Things that raise the risk include {_join(fact['risk_factors'])}.")
    elif intent == "diet":
        d = fact.get("diet", {}).get("eat", "")
        body = take("diet", d) if d else f"No specific dietary guidance is recorded for {name} in this knowledge base."
    elif intent == "specific_food":
        food = _extract_food(question)
        d = fact.get("diet", {}).get("eat", "")
        body = (
            f"This tool holds general dietary guidance for {name} rather than a list of individual foods, "
            f"so it cannot say definitively whether {food} is suitable for you. "
            + (f"The recorded guidance is: {d} " if d else "")
            + "If a specific food matters to you, a doctor or dietitian can advise for your situation."
        )
        used.append("diet")
    elif intent == "exercise":
        body = take("ayurvedic_lifestyle", fact.get("ayurvedic_lifestyle", "")) or \
            f"Activity advice for {name} depends on how you are feeling and on any other conditions you have."
    elif intent == "ayurveda":
        body = take("ayurvedic_lifestyle", fact.get("ayurvedic_lifestyle", ""))
        caution = fact.get("ayurvedic_caution")
        if caution:
            body += " " + take("ayurvedic_caution", caution)
    elif intent == "contagious":
        if fact["is_contagious"]:
            body = take("is_contagious",
                        f"Yes - {name} can be passed to other people. {fact['causes']} "
                        "Good hand hygiene and avoiding close contact while you are unwell reduce the risk.")
        else:
            body = take("is_contagious",
                        f"No - {name} is not passed from person to person. {fact['causes']}")
    elif intent == "duration":
        body = take("typical_course", fact.get("typical_course", "")) or \
            f"How long {name} lasts varies between people."
    elif intent == "complications":
        body = take("complications",
                    f"If {name} is not managed, the problems that can develop include {_join(fact['complications'])}.")
    elif intent == "prevention":
        body = take("risk_factors",
                    f"The risk factors for {name} are {_join(fact['risk_factors'])}, "
                    "so reducing the ones you can control is the main way to lower your risk.")
    elif intent == "tests":
        body = take("common_tests",
                    f"The investigations a doctor commonly uses for {name} are {_join(fact['common_tests'])}. "
                    "Whether you need any of them is a decision for the doctor assessing you.")
    elif intent == "when_doctor":
        body = take("when_to_see_doctor", fact.get("when_to_see_doctor", ""))
        if fact.get("condition_specific_red_flags"):
            body += " " + take("condition_specific_red_flags",
                               "Seek care urgently if you notice " +
                               _join([r.lower() for r in fact["condition_specific_red_flags"]]) + ".")
    elif intent == "medicines":
        # Deliberately refuses to name a drug, a dose or a brand.
        body = (
            f"This tool does not recommend medicines, doses or brands, and that applies to {name} as much as "
            "to anything else. Treatment depends on how severe your illness is, your other conditions and what "
            "else you take, which is exactly why it needs a doctor or pharmacist. "
        )
        if fact.get("common_tests"):
            body += f"They may first use {_join(fact['common_tests'], 3)} to confirm the picture. "
        body += "Please do not start or stop any prescribed medicine without professional advice."
        used.append("safety_policy")
    elif intent == "emergency":
        flags = _join([r.lower() for r in fact.get("condition_specific_red_flags", [])])
        body = (
            f"If you are worried this is an emergency, treat it as one. For {name} the warning signs are {flags}. "
            "In India call 108 for an ambulance or 112 for emergency services. Do not wait to see if it improves."
        )
        used.append("condition_specific_red_flags")
    else:
        body = take("overview", fact["overview"])

    return {
        "answer": re.sub(r"\s+", " ", body).strip(),
        "intent": intent,
        "confidence": round(float(conf), 3),
        "source": source,
        "sections_used": used,
    }


_FOOD_RE = re.compile(
    r"\b(?:eat|drink|have|take)\s+(?:some\s+|a\s+|an\s+|the\s+)?([a-zऀ-ॿ][a-zऀ-ॿ\s]{1,24}?)\b"
    r"(?:\s+(?:with|in|for|if|when|during|ok|okay)\b|\s*\?|$)",
    re.I,
)


def _extract_food(question: str) -> str:
    m = _FOOD_RE.search(question or "")
    if m:
        return m.group(1).strip()
    return "that food"


def suggestion_chips(fact: dict, last_intent: Optional[str] = None) -> list[str]:
    """Follow-up chips that change with the condition and the last question."""
    chips: list[str] = []
    if fact["is_contagious"] and last_intent != "contagious":
        chips.append("Is it contagious?")
    if last_intent != "duration":
        chips.append("How long does it last?")
    if last_intent != "diet" and fact.get("diet", {}).get("eat"):
        chips.append("What should I eat?")
    if last_intent != "when_doctor":
        chips.append("When should I see a doctor?")
    if last_intent != "complications":
        chips.append("What can go wrong?")
    if last_intent != "tests" and fact.get("common_tests"):
        chips.append("Which tests are used?")
    if last_intent != "ayurveda" and fact.get("ayurvedic_lifestyle"):
        chips.append("Any Ayurvedic guidance?")
    chips.append("I have new symptoms")
    return chips[:5]
