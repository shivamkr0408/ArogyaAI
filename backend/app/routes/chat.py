from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ..db import get_db
from ..models import Session, Message
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = Message(
        session_id=session.id,
        role="user",
        content=request.message,
    )
    db.add(user_msg)
    db.commit()

    from ..services.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    profile = request.profile.model_dump() if request.profile else None
    response = await orchestrator.process_turn(session, request.message, db, profile)

    assistant_msg = Message(
        session_id=session.id,
        role="assistant",
        content=response.reply_text,
        metadata_=response.model_dump(exclude={"reply_text"}),
    )
    db.add(assistant_msg)
    db.commit()

    return response
