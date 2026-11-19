"""
Assemble rag/condition_facts.json for all 41 conditions.

Most of a fact sheet already exists somewhere in this repo; only the fields
in condition_profiles.py had to be written by hand. This script merges:

    rag/condition_profiles.py        curated: aliases, causes, risk factors,
                                     contagiousness, complications, red flags,
                                     questions for the doctor, test names
    rag/knowledge_base/*.md          overview / typical course / when to see
                                     a doctor, reused verbatim so the fact
                                     sheet and the RAG passages agree
    ml/generate_data.py DISEASES     common_symptoms, in prevalence order
    backend/app/services/severity.py urgency tier
    rag/ayurveda_kb.csv              diet and ayurvedic lifestyle

Keeping it generated rather than hand-written means the fact sheets cannot
drift from the knowledge base or the model's own symptom profiles.

Run:  backend/venv/Scripts/python.exe rag/build_facts.py
"""

import csv
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

OUT = os.path.join(HERE, "condition_facts.json")
KB_DIR = os.path.join(HERE, "knowledge_base")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


profiles = _load("condition_profiles", os.path.join(HERE, "condition_profiles.py")).PROFILES
gen = _load("generate_data", os.path.join(ROOT, "ml", "generate_data.py"))
sev = _load("severity", os.path.join(ROOT, "backend", "app", "services", "severity.py"))

LABELS = json.load(open(os.path.join(ROOT, "ml", "model", "label_classes.json"), encoding="utf-8"))

# The label set carries two typos from the original public dataset; the
# knowledge base and the Ayurveda table use the corrected spellings.
SLUG_FIX = {
    "Dimorphic hemmorhoids(piles)": "hemorrhoids",
    "Peptic ulcer diseae": "peptic_ulcer_disease",
}
AYUR_FIX = {
    "Dimorphic hemmorhoids(piles)": "hemorrhoids",
    "Peptic ulcer diseae": "peptic ulcer disease",
}

URGENCY = {
    "low": "routine",
    "medium": "see a doctor soon",
    "high": "needs prompt medical attention",
}


def slug(name: str) -> str:
    return SLUG_FIX.get(name) or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def severity_map() -> dict:
    for value in vars(sev).values():
        if isinstance(value, dict) and "Malaria" in value:
            return value
    return {}


def kb_sections(name: str) -> dict:
    """Pull overview / course / when-to-see-doctor out of the KB article."""
    path = os.path.join(KB_DIR, slug(name) + ".md")
    if not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8").read()
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and not p.startswith("#")]
    if not paras:
        return {}
    doctor = next(
        (p for p in reversed(paras)
         if re.search(r"see a doctor|seek (prompt |immediate )?medical|emergency", p, re.I)),
        paras[-1],
    )
    course = next(
        (p for p in paras[1:]
         if re.search(r"symptom|typically|usually|recover|last", p, re.I) and p is not doctor),
        paras[1] if len(paras) > 1 else "",
    )
    selfcare = next(
        (p for p in paras if re.search(r"self-care|rest|fluids|hydrat|manage at home", p, re.I)),
        "",
    )
    return {
        "overview": paras[0],
        "typical_course": course,
        "self_care": selfcare,
        "when_to_see_doctor": doctor,
    }


def ayurveda_rows() -> dict:
    rows = {}
    with open(os.path.join(HERE, "ayurveda_kb.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["disease"].strip().lower()] = row
    return rows


def main():
    sevmap = severity_map()
    ayur = ayurveda_rows()
    sheets, problems = [], []

    for name in LABELS:
        prof = profiles.get(name)
        if not prof:
            problems.append(f"no curated profile for {name!r}")
            continue

        kb = kb_sections(name)
        if not kb:
            problems.append(f"no knowledge-base article for {name!r}")

        a = ayur.get(AYUR_FIX.get(name, name.lower()), {})
        tier = sevmap.get(name, "medium")

        sheets.append({
            "id": slug(name),
            "name": name,
            "aliases": prof["aliases"],
            "urgency": URGENCY.get(tier, "see a doctor soon"),
            "severity_tier": tier,
            "overview": kb.get("overview", ""),
            "common_symptoms": [s.replace("_", " ") for s in gen.DISEASES.get(name, [])],
            "causes": prof["causes"],
            "risk_factors": prof["risk_factors"],
            "is_contagious": prof["is_contagious"],
            "typical_course": kb.get("typical_course", ""),
            "self_care": kb.get("self_care", ""),
            "diet": {
                "eat": a.get("dietary_guidance", ""),
                "avoid": "",   # the KB states guidance as prose; no separate avoid list
            },
            "ayurvedic_lifestyle": a.get("lifestyle_guidance", ""),
            "ayurvedic_caution": a.get("caution_note", ""),
            "complications": prof["complications"],
            "condition_specific_red_flags": prof["red_flags"],
            "when_to_see_doctor": kb.get("when_to_see_doctor", ""),
            "questions_to_ask_doctor": prof["ask_doctor"],
            "common_tests": prof["tests"],
            "sources": prof["sources"],
        })

    json.dump(sheets, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"Wrote {OUT}")
    print(f"  conditions      : {len(sheets)} / {len(LABELS)}")
    print(f"  contagious      : {sum(1 for s in sheets if s['is_contagious'])}")
    print(f"  with KB overview: {sum(1 for s in sheets if s['overview'])}")
    print(f"  with ayurveda   : {sum(1 for s in sheets if s['ayurvedic_lifestyle'])}")
    unverified = [s['name'] for s in sheets
                  if any('needs verification' in x for x in s['sources'])]
    print(f"  sources flagged 'needs verification': {unverified or 'none'}")
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print("   -", p)
        sys.exit(1)


if __name__ == "__main__":
    main()
