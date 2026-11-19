from pydantic import BaseModel
from typing import Optional
from enum import Enum


class SessionState(str, Enum):
    GREETING = "greeting"
    SYMPTOM_COLLECTION = "symptom_collection"
    CLARIFYING = "clarifying"
    RESULT = "result"
    EMERGENCY = "emergency"


class HealthProfile(BaseModel):
    """Held on the user's device and sent with the turn; never stored."""
    age_group: Optional[str] = None
    sex: Optional[str] = None
    pregnant: bool = False
    known_conditions: list[str] = []
    allergies: list[str] = []


class ProfileConcern(BaseModel):
    level: str
    rule: str
    title: str
    detail: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    profile: Optional[HealthProfile] = None


class Alternative(BaseModel):
    disease: str
    probability: float


class AyurvedicGuidance(BaseModel):
    dosha_imbalance: str
    diet: str
    lifestyle: str
    general_herbs_note: str
    caution: str


class Triage(BaseModel):
    level: str
    label: str
    detail: str


class EvidenceChunk(BaseModel):
    """One retrieved knowledge-base passage, with its similarity score."""

    source_file: str
    disease_slug: str = ""
    similarity: Optional[float] = None
    excerpt: str = ""


class SymptomContribution(BaseModel):
    """One symptom's weight in the Naive Bayes decision."""

    symptom: str
    label: str
    log_likelihood: float
    discrimination: Optional[float] = None
    support_share: float = 0.0


class Explanation(BaseModel):
    """Exact per-symptom decomposition of the prediction."""

    condition: str
    compared_with: Optional[str] = None
    model: str
    weight_meaning: str = ""
    log_prior: Optional[float] = None
    contributions: list[SymptomContribution] = []
    top_discriminating: list[SymptomContribution] = []
    denial_evidence: list[dict] = []
    expected_but_not_reported: list[dict] = []
    note: str = ""


class Prediction(BaseModel):
    top_condition: str
    confidence: float
    severity: str = "medium"
    alternatives: list[Alternative]
    reasoning_summary: str
    evidence_sources: list[str]
    evidence: list[EvidenceChunk] = []
    explanation: Optional[Explanation] = None
    ayurvedic_guidance: Optional[AyurvedicGuidance] = None
    triage: Optional[Triage] = None
    clinical_context: dict = {}
    denied_symptoms: list[str] = []
    mentioned_conditions: list[dict] = []
    watch_for: list[str] = []
    # True when the evidence does not support naming one condition;
    # the client shows the differential instead of a headline result.
    abstained: bool = False
    abstain_reason: Optional[str] = None
    disclaimer: str = (
        "This is an educational, informational tool and not a medical diagnosis. "
        "Please consult a licensed healthcare professional."
    )


class PipelineStage(BaseModel):
    """One step of the per-turn pipeline, surfaced for the 'how it works' panel."""

    name: str
    status: str  # ok | skipped | triggered
    detail: str = ""
    ms: float = 0.0


class EntityOut(BaseModel):
    """A recognized entity, located in the user's original text."""

    label: str
    text: str
    original_text: str = ""
    canonical: str
    char_start: int
    char_end: int
    strategy: str
    confidence: float
    severity: Optional[str] = None
    assertion: Optional[dict] = None
    is_present: bool = True


class NlpTrace(BaseModel):
    """Per-turn NLP output, surfaced in the chat and the NLP Lab panel."""

    normalized: str = ""
    entities: list[EntityOut] = []
    normalization_edits: list[dict] = []
    intent: dict = {}
    distress: dict = {}
    duration_days: Optional[float] = None
    duration_bucket: Optional[str] = None
    trend: Optional[str] = None
    frequency: Optional[str] = None
    severity: Optional[str] = None
    timings: list[dict] = []
    total_ms: float = 0.0


class ChatResponse(BaseModel):
    reply_text: str
    state: str
    extracted_symptoms: list[str] = []
    denied_symptoms: list[str] = []
    intent: Optional[str] = None
    nlp: Optional[NlpTrace] = None
    turn_count: int = 0
    prediction: Optional[Prediction] = None
    matched_pattern: Optional[str] = None
    emergency_numbers_note: Optional[str] = None
    suggested_replies: list[str] = []
    question_topic: Optional[str] = None
    # Why this particular question was chosen (expected information gain).
    question_rationale: Optional[str] = None
    # Red-flag concerns that depend on who the patient is.
    profile_concerns: list[ProfileConcern] = []
    information_gain: Optional[float] = None
    followup_asked: int = 0
    followup_total: int = 0
    pipeline: list[PipelineStage] = []


class SessionCreateResponse(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_store_ready: bool


class MessageOut(BaseModel):
    role: str
    content: str
    metadata_: Optional[dict] = None
    created_at: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageOut]
    state: str
    accumulated_symptoms: dict


class StatsResponse(BaseModel):
    """Model/corpus facts for the 'under the hood' panel."""

    model_type: str
    diseases: int
    symptoms: int
    kb_documents: int
    kb_chunks: int
    embedding_model: str
    red_flag_patterns: int
    ayurveda_entries: int
    llm_mode: str


class ReportResponse(BaseModel):
    session_id: str
    generated_at: str
    duration_turns: int
    symptoms: list[str]
    clinical_context: dict
    prediction: Optional[dict] = None
    transcript: list[MessageOut]
    markdown: str
