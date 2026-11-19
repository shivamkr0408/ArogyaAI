"""
"I know my condition" mode.

Deliberately kept as its own router rather than folded into /chat: the
symptom flow is the part of the app that must not regress, and condition
mode needs none of its state machine. The two meet only through the bridges
the UI offers ("Learn about X" / "I have new symptoms").

Safety: every message that arrives here is put through the same red-flag
scan as the chat route, before any condition content is returned.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ..safety import red_flags
from ..services import condition_matcher, condition_qa, retriever

logger = logging.getLogger(__name__)
router = APIRouter(tags=["condition"])

EMERGENCY_NOTE = (
    "In India call 108 for an ambulance or 112 for emergency services."
)
DISCLAIMER = (
    "This is general information, not a diagnosis. Please consult a qualified "
    "healthcare professional about your own care."
)
# Conditions that are emergencies by nature: naming one must lead with the
# emergency card rather than a tidy fact sheet.
EMERGENCY_CONDITIONS = {"heart_attack", "paralysis_brain_hemorrhage"}


class MatchRequest(BaseModel):
    text: str


class AskRequest(BaseModel):
    condition_id: str
    question: str
    intent: Optional[str] = None


def _red_flag_block(text: str) -> Optional[dict]:
    """
    Run the deterministic safety scan on the raw message.

    check_red_flags() takes (symptom_list, raw_text) and matches keywords and
    proximity patterns against the raw text, so an empty symptom list is the
    correct call here: condition mode has no accumulated symptom vector, but
    the message itself must still be screened on every turn.
    """
    try:
        hit = red_flags.check_red_flags([], text)
    except Exception as exc:
        logger.warning("red-flag scan unavailable: %s", exc)
        return None
    if not hit:
        return None
    return {
        "emergency": True,
        "pattern": hit.get("pattern") or hit.get("description", "emergency"),
        "description": hit.get("description", "Possible medical emergency"),
        "message": (
            "What you have described could be a medical emergency. Please seek "
            "immediate in-person care now. " + EMERGENCY_NOTE
        ),
        "disclaimer": DISCLAIMER,
    }


def _summary(fact: dict) -> dict:
    """The small shape the UI needs for a chip or a card header."""
    return {
        "id": fact["id"],
        "name": fact["name"],
        "urgency": fact["urgency"],
        "is_contagious": fact["is_contagious"],
    }


@router.get("/condition/list")
def list_conditions():
    """All 41, for the picker and for 'did you mean' fallbacks."""
    return {"conditions": [_summary(f) for f in condition_matcher.all_conditions()]}


@router.get("/condition/{condition_id}")
def get_condition(condition_id: str):
    fact = condition_matcher.get(condition_id)
    if not fact:
        return {"found": False, "condition": None, "disclaimer": DISCLAIMER}
    return {
        "found": True,
        "condition": fact,
        "is_emergency_condition": condition_id in EMERGENCY_CONDITIONS,
        "emergency_note": EMERGENCY_NOTE if condition_id in EMERGENCY_CONDITIONS else None,
        "chips": condition_qa.suggestion_chips(fact),
        "disclaimer": DISCLAIMER,
    }


@router.post("/condition/match")
def match_condition(req: MatchRequest):
    """
    Resolve free text to one of the 41 conditions.

    Three outcomes the UI must distinguish: a confident match, an ambiguous
    one worth asking about, and no match at all -- which is answered
    honestly rather than by forcing the nearest label.
    """
    blocked = _red_flag_block(req.text)
    if blocked:
        return {"status": "emergency", **blocked}

    result = condition_matcher.match(req.text)
    fact = result["match"]

    if fact:
        return {
            "status": "matched",
            "condition": fact,
            "confidence": result["confidence"],
            "strategy": result["strategy"],
            "is_emergency_condition": fact["id"] in EMERGENCY_CONDITIONS,
            "emergency_note": EMERGENCY_NOTE if fact["id"] in EMERGENCY_CONDITIONS else None,
            "chips": condition_qa.suggestion_chips(fact),
            "disclaimer": DISCLAIMER,
        }

    if result["ambiguous"]:
        return {
            "status": "ambiguous",
            "candidates": [_summary(f) for f in result["candidates"]],
            "confidence": result["confidence"],
            "message": "I want to be sure I have the right one. Did you mean:",
            "disclaimer": DISCLAIMER,
        }

    return {
        "status": "not_found",
        "candidates": [_summary(f) for f in result["candidates"]],
        "confidence": result["confidence"],
        "message": (
            "That is not one of the 41 conditions this tool covers, so I would rather "
            "say so than guess. You can search the knowledge base, pick from the closest "
            "matches below, or describe your symptoms instead - but for a condition "
            "outside this list, a doctor is the right source."
        ),
        "disclaimer": DISCLAIMER,
    }


@router.post("/condition/ask")
def ask_condition(req: AskRequest):
    """Answer a question about an already-identified condition."""
    blocked = _red_flag_block(req.question)
    if blocked:
        return {"status": "emergency", **blocked}

    fact = condition_matcher.get(req.condition_id)
    if not fact:
        return {"status": "not_found", "message": "I do not have a fact sheet for that condition.",
                "disclaimer": DISCLAIMER}

    result = condition_qa.answer(fact, req.question, req.intent)

    # Enrich with the passages the retriever already indexes, so the answer
    # can cite where it came from. Absence is fine: the fact sheet alone is
    # a complete answer, the passages are corroboration.
    passages = []
    try:
        if retriever.is_ready():
            for chunk in retriever.retrieve(fact["name"], req.question, k=2) or []:
                passages.append({
                    "id": chunk.get("id") or chunk.get("source") or fact["id"],
                    "source": chunk.get("source") or fact["name"],
                    "text": (chunk.get("text") or "")[:400],
                })
    except Exception as exc:
        logger.debug("retrieval for condition answer failed: %s", exc)

    return {
        "status": "ok",
        "condition": _summary(fact),
        "answer": result["answer"],
        "intent": result["intent"],
        "confidence": result["confidence"],
        "intent_source": result["source"],
        "sections_used": result["sections_used"],
        "passages": passages,
        "chips": condition_qa.suggestion_chips(fact, result["intent"]),
        "disclaimer": DISCLAIMER,
    }
