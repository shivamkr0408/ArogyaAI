import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


RED_FLAG_PATTERNS = {
    "cardiac": {
        "description": "Possible cardiac emergency",
        "required": ["chest_pain"],
        "any_of": ["left_arm_pain", "jaw_pain", "sweating", "shortness_of_breath", "nausea", "dizziness"],
    },
    "stroke": {
        "description": "Possible stroke symptoms",
        "any_of": [
            "sudden_severe_headache", "worst_headache", "one_sided_weakness",
            "one_sided_numbness", "slurred_speech", "sudden_vision_loss",
            "sudden_confusion", "facial_drooping",
        ],
    },
    "respiratory_distress": {
        "description": "Severe respiratory distress",
        "required": ["difficulty_breathing"],
        "any_of": ["cannot_speak_full_sentences", "blue_lips", "blue_face", "severe_chest_tightness"],
    },
    "severe_bleeding": {
        "description": "Uncontrolled or severe bleeding",
        "any_of": [
            "uncontrolled_bleeding", "vomiting_blood", "coughing_blood",
            "heavy_bleeding", "blood_in_stool",
        ],
    },
    "anaphylaxis": {
        "description": "Possible severe allergic reaction",
        "required": ["breathing_difficulty"],
        "any_of": ["throat_swelling", "tongue_swelling", "hives", "known_allergen_exposure"],
    },
    "severe_abdominal": {
        "description": "Severe abdominal emergency",
        "required": ["severe_abdominal_pain"],
        "any_of": ["abdominal_rigidity", "inability_to_move", "vomiting_blood"],
    },
    "obstetric_emergency": {
        "description": "Obstetric emergency",
        "required": ["pregnant"],
        "any_of": ["heavy_vaginal_bleeding", "severe_abdominal_pain", "severe_headache_pregnancy"],
    },
    "mental_health_crisis": {
        "description": "Mental health crisis - potential self-harm or harm to others",
        "any_of": [
            "intent_to_harm_self", "suicidal_thoughts", "want_to_die",
            "intent_to_harm_others", "threatening_violence",
        ],
    },
}

EMERGENCY_KEYWORDS = {
    "chest_pain": ["chest pain", "chest pressure", "chest tightness", "heart attack", "chest feels heavy"],
    "left_arm_pain": ["left arm pain", "arm numbness", "tingling in arm", "arm pain"],
    "jaw_pain": ["jaw pain", "jaw ache", "pain in jaw"],
    "sweating": ["sweating", "cold sweat", "clammy", "perspiring heavily"],
    "shortness_of_breath": ["shortness of breath", "can't breathe", "hard to breathe", "breathing difficulty", "out of breath"],
    "nausea": ["nausea", "feeling sick", "want to vomit"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "faint", "feeling faint"],
    "sudden_severe_headache": ["worst headache", "sudden severe headache", "thunderclap headache", "worst headache of my life"],
    "worst_headache": ["worst headache of my life", "unbearable headache", "excruciating headache"],
    "one_sided_weakness": ["one side weak", "weakness on one side", "half body weak", "left side weak", "right side weak"],
    "one_sided_numbness": ["numbness on one side", "one side numb", "half body numb"],
    "slurred_speech": ["slurred speech", "can't speak properly", "difficulty speaking", "speech is slurred"],
    "sudden_vision_loss": ["sudden vision loss", "can't see", "vision went blurry suddenly", "lost vision"],
    "sudden_confusion": ["sudden confusion", "confused suddenly", "can't think clearly"],
    "facial_drooping": ["face drooping", "face droop", "mouth drooping", "one side of face dropped"],
    "difficulty_breathing": ["can't breathe", "severe difficulty breathing", "can't get air", "suffocating", "struggling to breathe"],
    "cannot_speak_full_sentences": ["can't speak in sentences", "can't talk properly", "only getting few words out"],
    "blue_lips": ["blue lips", "lips turning blue", "lips are blue"],
    "blue_face": ["face turning blue", "face is blue", "bluish face"],
    "severe_chest_tightness": ["severe chest tightness", "chest feels very tight"],
    "uncontrolled_bleeding": ["uncontrolled bleeding", "can't stop bleeding", "bleeding won't stop", "gushing blood"],
    "vomiting_blood": ["vomiting blood", "throwing up blood", "blood in vomit", "hematemesis"],
    "coughing_blood": ["coughing blood", "blood in sputum", "hemoptysis", "coughing up blood"],
    "heavy_bleeding": ["heavy bleeding", "soaking through pads", "hemorrhage"],
    "blood_in_stool": ["blood in stool", "bloody stool", "black tarry stool", "melena"],
    "breathing_difficulty": ["difficulty breathing", "can't breathe", "trouble breathing", "respiratory distress"],
    "throat_swelling": ["throat swelling", "throat closing", "throat feels tight", "can't swallow"],
    "tongue_swelling": ["tongue swelling", "tongue is swollen", "tongue feels big"],
    "hives": ["hives", "allergic rash", "widespread rash", "itchy welts"],
    "known_allergen_exposure": ["ate something allergic", "allergic reaction", "exposed to allergen"],
    "severe_abdominal_pain": ["severe stomach pain", "extreme abdominal pain", "worst stomach pain", "intense belly pain"],
    "abdominal_rigidity": ["stomach is rigid", "abdomen is rigid", "hard stomach", "rigid abdomen"],
    "inability_to_move": ["can't move", "can't move at all", "paralyzed"],
    "pregnant": ["pregnant", "i'm pregnant", "expecting", "pregnancy"],
    "heavy_vaginal_bleeding": ["heavy vaginal bleeding", "severe bleeding down there", "hemorrhage during pregnancy"],
    "severe_headache_pregnancy": ["severe headache during pregnancy", "preeclampsia symptoms"],
    "intent_to_harm_self": ["want to kill myself", "end my life", "suicide", "harm myself", "don't want to live", "want to die"],
    "suicidal_thoughts": ["suicidal", "suicide plan", "thinking about suicide"],
    "want_to_die": ["want to die", "wish i were dead", "life is not worth living"],
    "intent_to_harm_others": ["want to hurt someone", "going to kill someone", "threatening to hurt", "plan to attack"],
    "threatening_violence": ["going to hurt others", "will harm others", "revenge violence"],
}


# The ML feature vocabulary and the red-flag vocabulary were written
# separately, so accumulated symptoms like "breathlessness" never lined up
# with the rule key "shortness_of_breath". This maps one onto the other so
# symptoms gathered over several turns still reach the safety rules.
ML_SYMPTOM_ALIASES = {
    "breathlessness": ["shortness_of_breath", "breathing_difficulty"],
    "chest_tightness": ["chest_pain"],
    "bloody_stool": ["blood_in_stool"],
    "blurred_vision": ["sudden_vision_loss"],
    "weakness_in_limbs": ["one_sided_weakness"],
    "weakness_of_body": ["one_sided_weakness"],
    "altered_sensorium": ["sudden_confusion"],
    "slurred_speech": ["slurred_speech"],
    "coma": ["sudden_confusion"],
    "stomach_bleeding": ["uncontrolled_bleeding"],
    "blood_in_sputum": ["coughing_blood"],
    "rusty_sputum": ["coughing_blood"],
}


def _words_in(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def _proximity_match(text: str, phrase_words: set[str], window: int = 6) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    if not phrase_words:
        return False
    if len(words) < len(phrase_words):
        return False
    for i in range(len(words) - window + 1):
        if phrase_words.issubset(set(words[i:i + window])):
            return True
    return False


def check_red_flags(symptom_list: list[str], raw_text: str = "") -> Optional[dict]:
    normalized_symptoms = [s.lower().strip() for s in symptom_list]
    raw_lower = raw_text.lower()

    matched_keywords = {}

    def _match_phrase(symptom_key: str, phrase: str) -> bool:
        if phrase in raw_lower:
            return True
        phrase_words = _words_in(phrase)
        if len(phrase_words) < 2:
            return False
        window = 4 if len(phrase_words) == 2 else 6
        return _proximity_match(raw_text, phrase_words, window)

    for symptom_key, keywords in EMERGENCY_KEYWORDS.items():
        for kw in keywords:
            if _match_phrase(symptom_key, kw):
                matched_keywords[symptom_key] = True
                break

    for symptom in normalized_symptoms:
        canonical = symptom.replace(" ", "_")
        for kw_key in EMERGENCY_KEYWORDS:
            if symptom == kw_key or canonical == kw_key:
                matched_keywords[kw_key] = True
        for alias in ML_SYMPTOM_ALIASES.get(canonical, []):
            matched_keywords[alias] = True

    matched_patterns = []
    for pattern_name, pattern_def in RED_FLAG_PATTERNS.items():
        required = pattern_def.get("required", [])
        any_of = pattern_def.get("any_of", [])

        required_met = all(k in matched_keywords for k in required)
        any_of_met = any(k in matched_keywords for k in any_of)

        if required and any_of:
            if required_met and any_of_met:
                matched_patterns.append(pattern_name)
        elif required and not any_of:
            if required_met:
                matched_patterns.append(pattern_name)
        elif any_of and not required:
            if any_of_met:
                matched_patterns.append(pattern_name)

    if matched_patterns:
        primary_pattern = matched_patterns[0]
        return {
            "matched_pattern": primary_pattern,
            "description": RED_FLAG_PATTERNS[primary_pattern]["description"],
            "all_patterns": matched_patterns,
        }

    return None
