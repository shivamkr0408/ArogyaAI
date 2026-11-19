import logging
from typing import Optional

logger = logging.getLogger(__name__)

DISEASE_SEVERITY = {
    "Fungal infection": "low",
    "Allergy": "low",
    "GERD": "low",
    "Chronic cholestasis": "medium",
    "Drug Reaction": "medium",
    "Peptic ulcer diseae": "medium",
    "AIDS": "high",
    "Diabetes": "high",
    "Gastroenteritis": "low",
    "Bronchial Asthma": "medium",
    "Hypertension": "high",
    "Migraine": "low",
    "Cervical spondylosis": "low",
    "Paralysis (brain hemorrhage)": "high",
    "Jaundice": "medium",
    "Malaria": "high",
    "Chicken pox": "low",
    "Dengue": "high",
    "Typhoid": "medium",
    "Hepatitis A": "medium",
    "Hepatitis B": "medium",
    "Hepatitis C": "medium",
    "Hepatitis D": "high",
    "Hepatitis E": "medium",
    "Alcoholic hepatitis": "high",
    "Tuberculosis": "high",
    "Common Cold": "low",
    "Pneumonia": "high",
    "Dimorphic hemmorhoids(piles)": "low",
    "Heart attack": "high",
    "Vertigo": "low",
    "Urinary tract infection": "low",
    "Psoriasis": "low",
    "Impetigo": "low",
    "Arthritis": "medium",
    "Gout": "medium",
    "Asthma": "medium",
    "COPD": "high",
    "Anxiety": "low",
    "Depression": "medium",
    "Diabetic foot": "high",
}

DEFAULT_SEVERITY = "medium"


def lookup(disease: str) -> str:
    key = disease.strip()
    if key in DISEASE_SEVERITY:
        return DISEASE_SEVERITY[key]
    for stored in DISEASE_SEVERITY:
        if stored.lower() in key.lower() or key.lower() in stored.lower():
            return DISEASE_SEVERITY[stored]
    return DEFAULT_SEVERITY


def get_all() -> dict:
    return DISEASE_SEVERITY