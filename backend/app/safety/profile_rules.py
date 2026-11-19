"""
Red-flag rules that depend on who the patient is.

The base rules in red_flags.py are deliberately person-independent: a
crushing chest pain is an emergency whoever reports it. But some
presentations are only dangerous for particular people, and treating
everyone identically is the wrong kind of safe -- it under-triages infants,
older adults and pregnant women.

These rules are additive. They can escalate a message to an emergency, and
they can add a warning to an otherwise ordinary reply. They never
de-escalate: nothing here can cancel a red flag that the base layer raised.

The profile is held on the user's device and sent with the turn; nothing is
stored server-side. An absent profile simply means none of these fire.
"""

from typing import Optional

# Age bands the UI offers. Kept coarse on purpose: a symptom checker has no
# business asking for a date of birth.
INFANT = "under_1"
CHILD = "1_12"
TEEN = "13_17"
ADULT = "18_64"
OLDER = "65_plus"

FEVER_KEYS = {"fever", "high_fever", "mild_fever", "moderate_fever", "chills", "shivering"}
BREATH_KEYS = {"breathlessness", "difficulty_breathing", "fast_breathing", "chest_tightness"}
ABDO_KEYS = {"abdominal_pain", "stomach_pain", "vomiting"}
BLEED_KEYS = {"bloody_stool", "vomiting_blood", "coughing_blood", "uncontrolled_bleeding"}
HEAD_KEYS = {"headache", "blurred_vision", "dizziness", "face_swelling", "swelling"}


def _has(symptoms: set[str], keys: set[str]) -> bool:
    return bool(symptoms & keys)


def evaluate(profile: Optional[dict], symptoms: list[str], text: str = "") -> list[dict]:
    """
    Return the profile-specific concerns raised by this presentation.

    Each entry is {level, title, detail, rule}. `level` is "emergency" when
    the combination warrants immediate care regardless of what the
    classifier thinks, and "caution" when it should be surfaced prominently
    but is not by itself an emergency.
    """
    if not profile:
        return []

    s = {x.lower().strip() for x in (symptoms or [])}
    age = (profile.get("age_group") or "").strip()
    pregnant = bool(profile.get("pregnant"))
    conditions = {c.lower() for c in (profile.get("known_conditions") or [])}
    out: list[dict] = []

    # --- infants ---------------------------------------------------------
    # A fever in a baby under one is assessed in person, always. There is no
    # safe home-triage answer for it.
    if age == INFANT and _has(s, FEVER_KEYS):
        out.append({
            "level": "emergency",
            "rule": "infant_fever",
            "title": "A fever in a baby under 1 needs to be seen today",
            "detail": (
                "Babies under a year old can become seriously unwell very quickly and show "
                "few outward signs. Please have the baby assessed in person now rather than "
                "relying on any online tool."
            ),
        })
    if age == INFANT and _has(s, BREATH_KEYS):
        out.append({
            "level": "emergency",
            "rule": "infant_breathing",
            "title": "Breathing difficulty in a baby is an emergency",
            "detail": "Seek immediate in-person care. Do not wait to see if it settles.",
        })

    # --- pregnancy --------------------------------------------------------
    if pregnant:
        if _has(s, ABDO_KEYS):
            out.append({
                "level": "emergency",
                "rule": "pregnancy_abdominal",
                "title": "Abdominal pain in pregnancy should be assessed now",
                "detail": (
                    "Abdominal pain or persistent vomiting during pregnancy needs same-day "
                    "assessment by a doctor or midwife."
                ),
            })
        if _has(s, HEAD_KEYS):
            out.append({
                "level": "emergency",
                "rule": "pregnancy_preeclampsia",
                "title": "Headache, visual changes or swelling in pregnancy",
                "detail": (
                    "This combination can indicate raised blood pressure in pregnancy, which "
                    "is assessed urgently. Please contact your maternity service now."
                ),
            })
        if _has(s, FEVER_KEYS):
            out.append({
                "level": "caution",
                "rule": "pregnancy_fever",
                "title": "Fever in pregnancy is assessed sooner",
                "detail": "Contact your doctor or midwife today rather than waiting it out.",
            })

    # --- older adults -----------------------------------------------------
    if age == OLDER:
        if _has(s, FEVER_KEYS) and _has(s, BREATH_KEYS):
            out.append({
                "level": "emergency",
                "rule": "elderly_fever_breathless",
                "title": "Fever with breathlessness over 65 needs prompt assessment",
                "detail": (
                    "Chest infections can progress quickly in older adults and often present "
                    "without the usual warning signs. Please be seen today."
                ),
            })
        elif _has(s, FEVER_KEYS):
            out.append({
                "level": "caution",
                "rule": "elderly_fever",
                "title": "Fever over 65 is worth a same-day opinion",
                "detail": "Older adults can be more unwell than they feel. Do not delay if it worsens.",
            })

    # --- young children ---------------------------------------------------
    if age == CHILD and _has(s, FEVER_KEYS) and _has(s, BREATH_KEYS):
        out.append({
            "level": "emergency",
            "rule": "child_fever_breathless",
            "title": "Fever with breathing difficulty in a child",
            "detail": "This needs in-person assessment now.",
        })

    # --- known conditions that change the stakes --------------------------
    if "diabetes" in conditions and _has(s, {"slow_healing", "foul_smelling_wound", "numbness", "swelling"}):
        out.append({
            "level": "caution",
            "rule": "diabetic_foot_risk",
            "title": "A wound or numbness in the feet with diabetes",
            "detail": (
                "Foot problems in diabetes can deteriorate without pain as a warning. Have it "
                "looked at promptly rather than treating it at home."
            ),
        })
    if _has(s, BLEED_KEYS):
        out.append({
            "level": "emergency",
            "rule": "any_bleeding",
            "title": "Bleeding needs urgent assessment",
            "detail": "Please seek immediate in-person care.",
        })

    # Deduplicate by rule, keeping the first (emergency rules are listed first).
    seen, unique = set(), []
    for item in out:
        if item["rule"] in seen:
            continue
        seen.add(item["rule"])
        unique.append(item)
    return unique


def highest_level(concerns: list[dict]) -> Optional[str]:
    if any(c["level"] == "emergency" for c in concerns):
        return "emergency"
    if concerns:
        return "caution"
    return None
