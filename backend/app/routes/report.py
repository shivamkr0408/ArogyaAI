"""Consultation summary a user can save, print, or take to a doctor."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ..db import get_db
from ..models import Session
from ..schemas import ReportResponse, MessageOut
from ..services.followup import SLOT_LABELS

router = APIRouter(tags=["report"])


def _build_markdown(session: Session, prediction: dict | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    symptoms = [s.replace("_", " ") for s in (session.accumulated_symptoms or {}).keys()]
    slots = session.clinical_slots or {}

    lines = [
        "# Arogya AI — Consultation Summary",
        "",
        f"**Generated:** {stamp}  ",
        f"**Session:** `{session.id}`  ",
        f"**Conversation turns:** {session.turn_count or 0}",
        "",
        "> This summary was produced by an educational AI tool. It is **not** a medical",
        "> diagnosis. Please share it with a licensed healthcare professional rather than",
        "> acting on it alone.",
        "",
        "## Reported symptoms",
        "",
    ]
    lines += [f"- {s}" for s in symptoms] or ["_None recorded._"]

    if slots:
        lines += ["", "## Consultation answers", ""]
        lines += ["| Question | Answer |", "| --- | --- |"]
        lines += [f"| {SLOT_LABELS.get(k, k)} | {v} |" for k, v in slots.items()]

    if prediction:
        lines += [
            "",
            "## Assessment",
            "",
            f"**Most likely condition:** {prediction.get('top_condition')} "
            f"({prediction.get('confidence', 0) * 100:.0f}% confidence)  ",
            f"**Baseline severity:** {prediction.get('severity', 'unknown')}",
            "",
        ]
        triage = prediction.get("triage")
        if triage:
            lines += [f"**Recommended next step:** {triage['label']} — {triage['detail']}", ""]

        alts = prediction.get("alternatives") or []
        if alts:
            lines += ["**Other possibilities considered:**", ""]
            lines += [
                f"- {a['disease']} ({a['probability'] * 100:.0f}%)" for a in alts
            ]
            lines.append("")

        if prediction.get("reasoning_summary"):
            lines += ["**Reasoning:**", "", prediction["reasoning_summary"], ""]

        watch = prediction.get("watch_for") or []
        if watch:
            lines += ["## Seek care promptly if you notice", ""]
            lines += [f"- {w}" for w in watch]
            lines.append("")

        ayur = prediction.get("ayurvedic_guidance")
        if ayur:
            lines += [
                "## Ayurvedic lifestyle guidance",
                "",
                f"- **Dosha pattern:** {ayur.get('dosha_imbalance', '')}",
                f"- **Diet:** {ayur.get('diet', '')}",
                f"- **Lifestyle:** {ayur.get('lifestyle', '')}",
                f"- **Note:** {ayur.get('general_herbs_note', '')}",
                f"- **Caution:** {ayur.get('caution', '')}",
                "",
            ]

        sources = prediction.get("evidence_sources") or []
        if sources:
            lines += ["## Evidence sources", ""]
            lines += [f"- `{s}`" for s in sources]
            lines.append("")

    lines += ["## Transcript", ""]
    for m in session.messages:
        who = "You" if m.role == "user" else "Arogya AI"
        lines.append(f"**{who}:** {m.content}")
        lines.append("")

    return "\n".join(lines)


@router.get("/session/{session_id}/report", response_model=ReportResponse)
def get_report(session_id: str, db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # The most recent assistant message carrying a prediction is the assessment.
    prediction = None
    for m in reversed(session.messages):
        if m.role == "assistant" and (m.metadata_ or {}).get("prediction"):
            prediction = m.metadata_["prediction"]
            break

    return ReportResponse(
        session_id=session.id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        duration_turns=session.turn_count or 0,
        symptoms=[s.replace("_", " ") for s in (session.accumulated_symptoms or {}).keys()],
        clinical_context={SLOT_LABELS.get(k, k): v for k, v in (session.clinical_slots or {}).items()},
        prediction=prediction,
        transcript=[
            MessageOut(
                role=m.role,
                content=m.content,
                metadata_=None,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in session.messages
        ],
        markdown=_build_markdown(session, prediction),
    )
