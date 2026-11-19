"""
Structured follow-up engine.

The original flow asked a clarifying question only when the Naive Bayes
posterior was ambiguous, so a confident first message went straight to a
result with no consultation at all. This module adds a proper intake:
a small set of clinical "slots" (duration, intensity, progression, ...)
that a real triage conversation always covers, interleaved with the
model-driven discriminating-symptom questions.

Each question carries its own quick-reply options so the frontend can
render contextual buttons instead of a fixed Yes/No/Not-sure row.
"""

from typing import Optional

# Minimum questions to ask before showing a result, even when the model is
# already confident -- a one-shot answer reads as a guess, not a consultation.
MIN_FOLLOWUPS = 3
MAX_FOLLOWUPS = 6


class Slot:
    def __init__(self, key, question, options, requires=None, adds=None,
                 when_candidates=None):
        self.key = key
        self.question = question
        self.options = options
        # Only ask this slot when one of these symptoms is present.
        self.requires = requires or []
        # Map of answer -> symptom to add to the feature vector.
        self.adds = adds or {}
        # Only ask this slot when one of these conditions is currently among
        # the model's leading candidates. Empty means "always applicable".
        self.when_candidates = set(when_candidates or [])


SLOTS: list[Slot] = [
    Slot(
        key="duration",
        question="How long have you been feeling this way?",
        options=["Less than a day", "1-3 days", "About a week", "More than 2 weeks"],
    ),
    Slot(
        key="intensity",
        question="How intense would you say it feels overall?",
        options=["Mild - I can carry on normally", "Moderate - it's distracting", "Severe - hard to function"],
    ),
    Slot(
        key="progression",
        question="Since it started, is it getting better, worse, or staying about the same?",
        options=["Getting worse", "About the same", "Getting better"],
    ),
    Slot(
        key="fever_reading",
        question="Have you been able to measure your temperature?",
        options=["Above 102°F / 39°C", "100-102°F / 38-39°C", "Below 100°F / 38°C", "Haven't measured"],
        requires=["fever", "high_fever", "mild_fever", "chills", "shivering", "sweating"],
        adds={"Above 102°F / 39°C": "high_fever", "100-102°F / 38-39°C": "high_fever", "Below 100°F / 38°C": "mild_fever"},
    ),
    Slot(
        key="exposure",
        question="Has anything happened recently that might be connected?",
        options=[
            "Travelled recently",
            "Ate outside / street food",
            "Someone at home is unwell",
            "Started a new medicine",
            "Nothing I can think of",
        ],
    ),
    Slot(
        key="appetite_sleep",
        question="How have your appetite and sleep been through this?",
        options=["Both are fine", "Eating less than usual", "Sleeping badly", "Both are affected"],
        adds={"Eating less than usual": "loss_of_appetite", "Both are affected": "loss_of_appetite"},
    ),
]

# ---------------------------------------------------------------------------
# Condition-specific follow-ups.
#
# The generic slots above are the same for everyone, which is fine for
# context but useless for telling apart conditions whose symptom pictures
# overlap almost entirely. The hepatitis variants are the clearest case:
# they share fatigue, nausea, vomiting and joint pain, and what actually
# separates them is an exposure or a history, not another symptom. Asking
# "are you also experiencing nausea?" cannot help there; asking what the
# person was exposed to can.
#
# Each of these is gated on the conditions currently leading the
# differential, so a febrile presentation is never asked about inhalers.
# ---------------------------------------------------------------------------

HEPATIC = [
    "Hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
    "Alcoholic hepatitis", "Jaundice", "Chronic cholestasis",
]
RESPIRATORY = [
    "Asthma", "Bronchial Asthma", "COPD", "Pneumonia", "Tuberculosis",
    "Common Cold",
]
FEBRILE = ["Malaria", "Dengue", "Typhoid", "Gastroenteritis"]
DERMATOLOGIC = ["Fungal infection", "Psoriasis", "Impetigo", "Drug Reaction",
                "Chicken pox"]

CONDITION_SLOTS: list[Slot] = [
    # ---- hepatic ----------------------------------------------------------
    Slot(
        key="hepatic_exposure",
        requires=["yellowing_of_eyes","yellowing_of_skin","dark_urine","pale_stools","abdominal_pain","nausea","vomiting","fatigue","loss_of_appetite","itching"],
        question="Has any of these happened in the last few weeks?",
        options=[
            "Ate street or outside food",
            "Drank water I wasn't sure about",
            "Had an injection, tattoo or piercing",
            "Had a blood transfusion",
            "None of these",
        ],
        when_candidates=HEPATIC,
        adds={
            "Ate street or outside food": "recent_contaminated_food",
            "Drank water I wasn't sure about": "recent_contaminated_water",
            "Had an injection, tattoo or piercing": "unsafe_injection_exposure",
            "Had a blood transfusion": "blood_transfusion_history",
        },
    ),
    Slot(
        key="hepatic_history",
        requires=["yellowing_of_eyes","yellowing_of_skin","dark_urine","pale_stools","abdominal_pain","nausea","vomiting","fatigue","loss_of_appetite","itching"],
        question="Do either of these apply to you?",
        options=[
            "I've been told I have hepatitis B",
            "I drink alcohol most days",
            "Neither",
        ],
        when_candidates=HEPATIC,
        adds={
            "I've been told I have hepatitis B": "known_hepatitis_b",
            "I drink alcohol most days": "heavy_alcohol_use",
        },
    ),
    Slot(
        key="stool_colour",
        requires=["yellowing_of_eyes","yellowing_of_skin","dark_urine","abdominal_pain","nausea","vomiting","fatigue","loss_of_appetite","itching"],
        question="Have your stools changed colour at all?",
        options=["Pale or clay-coloured", "Normal", "Haven't noticed"],
        when_candidates=HEPATIC,
        adds={"Pale or clay-coloured": "pale_stools"},
    ),

    # ---- respiratory ------------------------------------------------------
    Slot(
        key="breathing_pattern",
        requires=["cough","breathlessness","wheezing","chest_tightness","chest_pain","continuous_sneezing","congestion"],
        question="When is your breathing or coughing at its worst?",
        options=[
            "At night or early morning",
            "Around dust, cold air or exercise",
            "It's much the same all the time",
            "Not sure",
        ],
        when_candidates=RESPIRATORY,
        adds={
            "At night or early morning": "night_time_symptoms",
            "Around dust, cold air or exercise": "symptoms_with_triggers",
        },
    ),
    Slot(
        key="sputum",
        requires=["cough","breathlessness","wheezing","chest_tightness","congestion"],
        question="When you cough, does anything come up?",
        options=[
            "Thick yellow or green phlegm",
            "Phlegm most mornings, for months now",
            "Dry cough, nothing comes up",
        ],
        when_candidates=RESPIRATORY,
        adds={
            "Thick yellow or green phlegm": "productive_sputum",
            "Phlegm most mornings, for months now": "chronic_productive_cough",
        },
    ),
    Slot(
        key="resp_history",
        requires=["cough","breathlessness","wheezing","chest_tightness","chest_pain"],
        question="Do any of these apply?",
        options=[
            "I smoke, or used to",
            "This cough has lasted more than two weeks",
            "I've been sweating heavily at night",
            "None of these",
        ],
        when_candidates=RESPIRATORY,
        adds={
            "I smoke, or used to": "smoking_history",
            "This cough has lasted more than two weeks": "cough_over_two_weeks",
            "I've been sweating heavily at night": "night_sweats",
        },
    ),
    Slot(
        key="inhaler",
        requires=["breathlessness","wheezing","chest_tightness","cough"],
        question="Do you use an inhaler, and is it helping?",
        options=[
            "Using it, but it's not helping much",
            "Using it, and it helps",
            "I don't use an inhaler",
        ],
        when_candidates=["Asthma", "Bronchial Asthma", "COPD"],
        adds={"Using it, but it's not helping much": "reliever_not_helping"},
    ),

    # ---- febrile ----------------------------------------------------------
    # `requires` matters as much as `when_candidates` here: this question
    # only makes sense once a fever has actually been reported, and its
    # answers add symptoms, so asking it unprompted invents evidence.
    Slot(
        key="febrile_pattern",
        question="Alongside the fever, are you noticing any of these?",
        options=[
            "Shaking chills, then heavy sweating",
            "A rash on my skin",
            "Severe joint or muscle pain",
            "Constipation and tummy pain",
            "None of these",
        ],
        when_candidates=FEBRILE,
        requires=["fever", "high_fever", "mild_fever", "moderate_fever",
                  "chills", "shivering", "sweating"],
        adds={
            "Shaking chills, then heavy sweating": "chills",
            "A rash on my skin": "skin_rash",
            "Severe joint or muscle pain": "joint_pain",
            "Constipation and tummy pain": "constipation",
        },
    ),

    # ---- dermatologic -----------------------------------------------------
    Slot(
        key="skin_appearance",
        requires=["skin_rash","itching","blister","skin_peeling","dischromic_patches","red_sore_around_nose","nodal_eruptions"],
        question="What does the affected skin look like?",
        options=[
            "Silvery, scaly patches",
            "Blisters with a yellow crust",
            "Discoloured patches",
            "Raised red bumps",
        ],
        when_candidates=DERMATOLOGIC,
        adds={
            "Silvery, scaly patches": "silver_like_dusting",
            "Blisters with a yellow crust": "yellow_crust_sortation",
            "Discoloured patches": "dischromic_patches",
            "Raised red bumps": "blister",
        },
    ),
]

SLOT_BY_KEY = {s.key: s for s in SLOTS + CONDITION_SLOTS}

# Human-readable labels used in the consultation report.
SLOT_LABELS = {
    "duration": "Duration",
    "intensity": "Reported intensity",
    "progression": "Trend",
    "fever_reading": "Temperature",
    "exposure": "Recent exposure",
    "appetite_sleep": "Appetite & sleep",
    "hepatic_exposure": "Recent exposure",
    "hepatic_history": "Relevant history",
    "stool_colour": "Stool colour",
    "breathing_pattern": "When symptoms are worst",
    "sputum": "Sputum",
    "resp_history": "Smoking / cough history",
    "inhaler": "Inhaler response",
    # Labels describe the question that was actually asked, so a result
    # never reports "Fever pattern" for an answer about something else.
    "febrile_pattern": "Alongside the fever",
    "skin_appearance": "Appearance of the skin",
}


def _slot_applies(slot: Slot, symptoms: list[str]) -> bool:
    if not slot.requires:
        return True
    return any(s in symptoms for s in slot.requires)


def next_slot(filled: dict, symptoms: list[str]) -> Optional[Slot]:
    """The next unanswered clinical slot that applies to this presentation."""
    for slot in SLOTS:
        if slot.key in filled:
            continue
        if _slot_applies(slot, symptoms):
            return slot
    return None


def next_condition_slot(
    filled: dict, candidates: list[str], symptoms: list[str] | None = None
) -> Optional[Slot]:
    """
    The next unanswered condition-specific question, given the conditions
    currently leading the differential AND what the user has actually
    reported.

    Both gates matter. Checking only the candidate list asked a fever
    question of someone who had never mentioned a fever -- a constipation
    and stomach-pain complaint reached Gastroenteritis in the differential,
    was asked "is anything else going along with the fever?", and the answer
    then *added* chills to the feature vector, pushing the result to Malaria.
    A question must never introduce a symptom the presentation does not
    already support.
    """
    leading = set(candidates)
    present = symptoms or []
    for slot in CONDITION_SLOTS:
        if slot.key in filled:
            continue
        if not (slot.when_candidates & leading):
            continue
        if not _slot_applies(slot, present):
            continue
        return slot
    return None


def record_answer(slot_key: str, answer: str, filled: dict) -> Optional[str]:
    """
    Store a slot answer. Returns a symptom name to add to the feature
    vector when the answer implies one (e.g. a 103°F reading -> high_fever).
    """
    filled[slot_key] = answer
    slot = SLOT_BY_KEY.get(slot_key)
    if not slot:
        return None
    for option, symptom in slot.adds.items():
        if option.lower() in answer.lower() or answer.lower() in option.lower():
            return symptom
    return None


def symptom_question(symptom: str) -> dict:
    """A discriminating-symptom question with its own quick replies."""
    label = symptom.replace("_", " ")
    return {
        "question": f"Are you also experiencing {label}?",
        "options": ["Yes", "No", "A little", "Not sure"],
        "topic": f"symptom:{symptom}",
    }


def total_planned(symptoms: list[str]) -> int:
    """How many questions this intake expects to ask, for the progress bar."""
    applicable = sum(1 for s in SLOTS if _slot_applies(s, symptoms))
    return max(MIN_FOLLOWUPS, min(MAX_FOLLOWUPS, applicable))


def triage(disease_severity: str, filled: dict) -> dict:
    """
    Combine the condition's baseline severity with what the follow-up
    answers revealed into a single actionable recommendation.
    """
    score = {"low": 0, "medium": 2, "high": 4}.get(disease_severity, 2)

    intensity = (filled.get("intensity") or "").lower()
    if intensity.startswith("severe"):
        score += 2
    elif intensity.startswith("moderate"):
        score += 1

    duration = (filled.get("duration") or "").lower()
    if "more than 2 weeks" in duration:
        score += 2
    elif "about a week" in duration:
        score += 1

    if (filled.get("progression") or "").lower().startswith("getting worse"):
        score += 2

    fever = (filled.get("fever_reading") or "").lower()
    if "above 102" in fever:
        score += 2

    if score >= 7:
        return {
            "level": "urgent",
            "label": "See a doctor today",
            "detail": "Your answers point to a presentation that should be assessed in person without delay.",
        }
    if score >= 4:
        return {
            "level": "soon",
            "label": "Book an appointment in the next 1-2 days",
            "detail": "This is worth a professional opinion soon, though it does not read as an emergency.",
        }
    return {
        "level": "self_care",
        "label": "Self-care and monitoring is reasonable",
        "detail": "Keep an eye on your symptoms and see a doctor if they worsen or persist.",
    }
