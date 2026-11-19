from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from ..db import get_db
from ..models import Session
from ..schemas import SessionCreateResponse, ChatHistoryResponse, MessageOut

router = APIRouter(tags=["session"])


@router.post("/session", response_model=SessionCreateResponse)
def create_session(db: DBSession = Depends(get_db)):
    session = Session()
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionCreateResponse(session_id=session.id)


@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str, db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[],
            state="greeting",
            accumulated_symptoms={},
        )

    messages = [
        MessageOut(
            role=m.role,
            content=m.content,
            metadata_=m.metadata_,
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
        for m in session.messages
    ]

    return ChatHistoryResponse(
        session_id=session.id,
        messages=messages,
        state=session.state or "greeting",
        accumulated_symptoms=session.accumulated_symptoms or {},
    )
