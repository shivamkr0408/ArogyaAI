import csv
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_kb: dict[str, dict] = {}
_loaded = False

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rag", "ayurveda_kb.csv")


def load_kb():
    global _kb, _loaded

    if not os.path.exists(KB_PATH):
        logger.warning("Ayurveda KB not found")
        return

    with open(KB_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row.get("disease", "").strip().lower()
            if disease:
                _kb[disease] = {
                    "dosha_imbalance": row.get("likely_dosha_imbalance", "Vata-Pitta"),
                    "diet": row.get("dietary_guidance", "Warm, simple, freshly cooked food"),
                    "lifestyle": row.get("lifestyle_guidance", "Regular sleep, gentle exercise, stress reduction"),
                    "general_herbs_note": row.get("general_herbs_note", ""),
                    "caution": row.get("caution_note", "Consult an Ayurvedic practitioner before following any herbal regimen."),
                }

    _loaded = True
    logger.info(f"Loaded Ayurveda KB with {len(_kb)} entries")


def is_loaded() -> bool:
    return _loaded


def lookup(disease: str) -> Optional[dict]:
    key = disease.strip().lower()
    if key in _kb:
        return _kb[key]

    for kb_key in _kb:
        if kb_key in key or key in kb_key:
            return _kb[kb_key]

    return {
        "dosha_imbalance": "General imbalance",
        "diet": "Warm, simple, freshly cooked food; avoid cold, oily, and heavy meals. Stay hydrated with warm water or herbal teas.",
        "lifestyle": "Maintain regular sleep schedule, practice gentle exercise like walking or yoga, and focus on stress reduction through meditation or deep breathing.",
        "general_herbs_note": "Ayurvedic traditions recommend warming spices like ginger, cumin, and turmeric for general wellness support.",
        "caution": "This is general wellness information only. Consult a qualified Ayurvedic practitioner or healthcare professional before making any health decisions.",
    }
