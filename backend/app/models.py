from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from .db import Base


def generate_uuid():
    return str(uuid.uuid4())


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    state = Column(String, default="GREETING")
    accumulated_symptoms = Column(JSON, default=dict)
    turn_count = Column(Integer, default=0)
    # Answers to the structured follow-up questions (duration, intensity, ...).
    clinical_slots = Column(JSON, default=dict)
    # Number of follow-up questions asked so far this consultation.
    followup_asked = Column(Integer, default=0)
    # Topic of the question we are currently awaiting an answer to.
    pending_topic = Column(String, nullable=True)
    # Symptoms the user explicitly denied. Kept rather than merely dropped:
    # BernoulliNB scores absence, and the summary should say what was ruled out.
    denied_symptoms = Column(JSON, default=dict)
    # The last assessment, so a follow-up question ("what should I eat?") can
    # be answered without re-running the whole consultation.
    last_prediction = Column(JSON, nullable=True)
    # Conditions the user named (comorbidity or family history).
    mentioned_conditions = Column(JSON, default=list)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("Session", back_populates="messages")
