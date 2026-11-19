from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv

load_dotenv()

from .db import engine, Base
from .routes import chat, session, health, debug, report, stats, nlp, condition
from .services import predictor, ayurveda_lookup, retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)


def _migrate_sqlite_columns():
    """
    Add columns introduced after a database was first created.
    create_all() only creates missing tables, never missing columns, so an
    existing arogya.db would otherwise break on the new session fields.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "sessions" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("sessions")}
    additions = {
        "clinical_slots": "JSON",
        "followup_asked": "INTEGER DEFAULT 0",
        "pending_topic": "VARCHAR",
        "denied_symptoms": "JSON",
        "last_prediction": "JSON",
        "mentioned_conditions": "JSON",
    }
    with engine.begin() as conn:
        for name, ddl in additions.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {name} {ddl}"))


_migrate_sqlite_columns()

app = FastAPI(title="Arogya AI", version="0.1.0")

# Vite falls back to 5174+ when 5173 is taken, so allow the usual dev range.
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:5173")
ALLOWED_ORIGINS = list({CORS_ORIGIN, "http://localhost:5173", "http://localhost:5174", "http://localhost:5175"})
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(session.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(debug.router, prefix="/api")
app.include_router(report.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(nlp.router, prefix="/api")
app.include_router(condition.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """
    Load all ML artifacts and knowledge bases at startup.
    This order matters:
      1. Predictor model (required for symptom->disease mapping)
      2. Ayurveda KB (cheap CSV lookup)
      3. Vector retriever (RAG grounding for explanations)
    """
    logger.info("Arogya AI backend starting up...")

    if predictor.load_model():
        logger.info("Predictor model loaded successfully")
    else:
        logger.warning("Predictor model NOT loaded - run ml/train_model.py first")

    ayurveda_lookup.load_kb()

    # The retriever loads sentence-transformers (torch), which is slow (~20s).
    # Do it in a background thread so the server binds and starts serving
    # immediately; retrieve() already returns [] gracefully until it's ready.
    import threading

    def _load_retriever_bg():
        if retriever.init_retriever():
            logger.info("Retriever initialized successfully (background)")
        else:
            logger.warning("Retriever NOT ready - run rag/build_index.py first")

        # Warm the NLP pipeline *after* the retriever, so the semantic NER
        # tier reuses that already-loaded MiniLM instead of loading a second
        # copy. Both happen off the request path.
        try:
            from .nlp import pipeline as nlp_pipeline

            status = nlp_pipeline.warm_up()
            logger.info("NLP pipeline warm: %s", status)
        except Exception as exc:
            logger.warning("NLP warm-up failed (pipeline still works lazily): %s", exc)

    threading.Thread(target=_load_retriever_bg, name="retriever-init", daemon=True).start()
    logger.info("Retriever + NLP warming in background; server is ready to accept requests")