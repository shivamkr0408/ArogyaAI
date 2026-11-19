import logging
import json
import time
from typing import Optional

from sqlalchemy.orm import Session as DBSession

from ..models import Session
from ..schemas import (
    ChatResponse,
    Prediction,
    Alternative,
    AyurvedicGuidance,
    EntityOut,
    EvidenceChunk,
    Explanation,
    NlpTrace,
    SymptomContribution,
    Triage,
    PipelineStage,
)


def _entity_payload(entity: dict) -> dict:
    """Keep only the fields EntityOut declares."""
    allowed = {
        "label", "text", "original_text", "canonical", "char_start",
        "char_end", "strategy", "confidence", "severity", "assertion",
        "is_present",
    }
    return {k: v for k, v in entity.items() if k in allowed}
from ..services import predictor, retriever, ayurveda_lookup, severity as severity_svc
from ..services import followup as followup_svc
from ..services import infogain
from ..services import explain as explain_svc
from ..services.symptom_extractor import analyze_message
from ..services.llm_client import chat_completion
from ..safety.red_flags import check_red_flags
from ..safety import profile_rules
from ..prompts import load_prompt
from ..nlp import intent as intent_nlp
from ..nlp import sentiment as sentiment_nlp

logger = logging.getLogger(__name__)

CLARIFICATION_CAP = followup_svc.MAX_FOLLOWUPS
CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.15

# Minimum intent confidence before the dialogue policy will act on it (route
# to a canned answer, or declare a message out of scope). Below this the turn
# is handled as an ordinary symptom description, which is the safe default.
INTENT_ROUTING_THRESHOLD = 0.45

NEGATIVE_ANSWERS = {"no", "nope", "not really", "none", "no other symptoms", "nothing"}

# Symptoms worth telling the user to watch for, per severity band.
WATCH_FOR = {
    "high": [
        "Difficulty breathing or shortness of breath",
        "Chest pain or pressure",
        "Confusion, fainting, or inability to stay awake",
        "Fever that stays above 103°F / 39.5°C",
    ],
    "medium": [
        "Symptoms getting sharply worse over 24 hours",
        "Inability to keep fluids down",
        "New rash, severe pain, or persistent vomiting",
    ],
    "low": [
        "Symptoms lasting longer than a week without improvement",
        "A fever developing or climbing",
    ],
}


class Timer:
    """Records how long each pipeline stage took, for the 'how it works' panel."""

    def __init__(self):
        self.stages: list[PipelineStage] = []
        self._t0 = time.perf_counter()

    def mark(self, name: str, status: str = "ok", detail: str = ""):
        now = time.perf_counter()
        self.stages.append(
            PipelineStage(name=name, status=status, detail=detail, ms=round((now - self._t0) * 1000, 1))
        )
        self._t0 = now


class Orchestrator:
    async def process_turn(
        self, session: Session, user_message: str, db: DBSession,
        profile: Optional[dict] = None,
    ) -> ChatResponse:
        trace = Timer()
        current_state = session.state or "GREETING"
        accumulated = dict(session.accumulated_symptoms or {})
        slots = dict(session.clinical_slots or {})
        turn_count = session.turn_count or 0
        asked = session.followup_asked or 0
        pending = session.pending_topic

        feature_order = predictor.get_feature_order()

        # Extraction result carried over from the greeting probe below, so a
        # first message that already describes symptoms is not analysed twice.
        preextracted = None

        if current_state == "GREETING":
            session.state = "SYMPTOM_COLLECTION"
            session.turn_count = 0
            session.clinical_slots = {}
            session.followup_asked = 0
            session.pending_topic = None
            db.commit()

            # The greeting used to be returned unconditionally, which threw
            # away the first message. Most people open by describing their
            # symptoms rather than saying hello, so they were told "tell me
            # what you're feeling" immediately after doing exactly that, and
            # had to type it again. Greet only when there is nothing
            # clinical to work with; otherwise fall through and answer.
            preextracted = await analyze_message(user_message, feature_order)
            if not preextracted.get("extracted_symptoms"):
                trace.mark("Greeting", "ok", "New consultation started")
                return ChatResponse(
                    reply_text=(
                        "Hello! I'm Arogya AI, your health guidance assistant. "
                        "Tell me what you're feeling in your own words and I'll ask a few "
                        "follow-up questions before sharing any guidance. "
                        "This is for information only — always consult a healthcare professional."
                    ),
                    state="symptom_collection",
                    turn_count=0,
                    suggested_replies=[
                        "I've had a fever and headache for two days",
                        "My stomach hurts and I feel nauseous",
                        "I have a cough that won't go away",
                    ],
                    pipeline=trace.stages,
                )
            trace.mark(
                "Greeting",
                "skipped",
                "First message already described symptoms - answering it directly",
            )
            current_state = "SYMPTOM_COLLECTION"
        previous_prediction = session.last_prediction or None

        # --- 0. Intent --------------------------------------------------
        # A question about an existing result ("what should I eat?") must not
        # be mistaken for a new symptom description, which is what the old
        # flow did -- it reset the consultation and replied "I didn't catch a
        # specific symptom there".
        intent_result = intent_nlp.classify(user_message)
        detected_intent = intent_result["intent"]
        intent_confidence = intent_result["confidence"]
        # The classifier reports a posterior; the dialogue policy decides how
        # much of it to trust. Acting on a low-confidence intent is the more
        # expensive error here -- it would answer a question the user did not
        # ask, or dismiss a real symptom as out of scope -- so routing needs
        # a floor, while an unsure turn just falls through to the normal
        # symptom path. Measured intent accuracy is in eval/results/nlp_eval.json.
        intent_trusted = (
            intent_confidence >= INTENT_ROUTING_THRESHOLD
            or intent_result["source"] == "rule"
        )
        trace.mark(
            "Intent classification",
            "ok" if intent_trusted else "skipped",
            f"{detected_intent} ({intent_confidence:.0%}, {intent_result['source']})"
            + ("" if intent_trusted else " - below routing threshold, treated as symptom description"),
        )

        if (
            current_state == "RESULT"
            and not pending
            and intent_trusted
            and intent_nlp.is_question_about_result(detected_intent)
            and previous_prediction
        ):
            trace.mark("Follow-up answer", "ok", "Answered from the existing assessment")
            return self._answer_about_result(
                session, detected_intent, previous_prediction, turn_count, trace
            )

        if detected_intent == "thanks" and intent_trusted and current_state == "RESULT":
            trace.mark("Closing", "ok", "Conversation closed")
            return ChatResponse(
                reply_text=(
                    "You're welcome - I hope you feel better soon. Remember this "
                    "was general information, not a diagnosis, so do see a "
                    "healthcare professional if anything worsens."
                ),
                state="result",
                turn_count=turn_count,
                intent=detected_intent,
                suggested_replies=["Start a new consultation"],
                pipeline=trace.stages,
            )

        if detected_intent == "out_of_scope" and intent_trusted and not pending:
            trace.mark("Scope check", "triggered", "Outside the assistant's scope")
            return ChatResponse(
                reply_text=(
                    "I can only help with describing and understanding physical "
                    "symptoms. Tell me what you're feeling - for example a fever, "
                    "headache, cough, stomach pain or fatigue - and I'll take it "
                    "from there."
                ),
                state="symptom_collection",
                turn_count=turn_count,
                intent=detected_intent,
                suggested_replies=[
                    "I have a fever and headache",
                    "My stomach hurts",
                    "I have a cough",
                ],
                pipeline=trace.stages,
            )

        # A fresh symptom query after a result clears the previous consultation.
        if current_state == "RESULT":
            accumulated, slots, asked, pending = {}, {}, 0, None
            turn_count = 0

        # --- 1. Answer to a pending follow-up question ---------------------
        answered_topic = None
        if pending:
            answered_topic = self._apply_answer(pending, user_message, accumulated, slots, feature_order)
            trace.mark("Follow-up intake", "ok", f"Recorded answer for {pending.split(':')[-1]}")

        # --- 2. NLP pipeline ----------------------------------------------
        # Reuse the greeting probe when the opening message was already
        # analysed, rather than running the whole pipeline over it twice.
        extraction = preextracted or await analyze_message(user_message, feature_order)
        extracted = extraction.get("extracted_symptoms", [])
        analysis = extraction.get("analysis")

        for neg in extraction.get("raw_negations", []):
            accumulated.pop(neg, None)

        for symptom in extracted:
            canonical = symptom.lower().strip().replace(" ", "_")
            if canonical in feature_order:
                accumulated[canonical] = True

        # Denied symptoms are recorded, not just dropped: BernoulliNB scores
        # absence, and the summary should say what was ruled out.
        denied = dict(session.denied_symptoms or {})
        for neg in extraction.get("raw_negations", []):
            if neg in feature_order:
                denied[neg] = True
        session.denied_symptoms = denied

        trace.mark(
            "Entity recognition + assertion",
            "ok",
            f"{len(extracted)} affirmed, {len(extraction.get('raw_negations', []))} denied; "
            f"{len(accumulated)} tracked in total",
        )

        # --- 2b. Context the user already volunteered ----------------------
        # No point asking "how long?" when the first message said "since 3
        # days". Extracted values are written into the same slots the
        # questionnaire fills, so downstream code cannot tell them apart.
        auto_filled = self._absorb_context(analysis, slots)
        if auto_filled:
            trace.mark(
                "Context extraction",
                "ok",
                "Captured " + ", ".join(f"{k}={v}" for k, v in auto_filled.items()),
            )

        # --- 3. Safety layer (always, before anything else is shown) -------
        emergency_check = check_red_flags(list(accumulated.keys()), user_message)

        # Person-dependent rules run alongside the base scan. They can only
        # escalate: a fever is routine in a healthy adult and an emergency in
        # a baby under one, but nothing here can cancel a base red flag.
        concerns = profile_rules.evaluate(profile, list(accumulated.keys()), user_message)
        profile_emergency = profile_rules.highest_level(concerns) == "emergency"
        if concerns:
            trace.mark(
                "Profile safety rules",
                "triggered" if profile_emergency else "ok",
                "; ".join(c["rule"] for c in concerns),
            )

        if emergency_check or profile_emergency or extraction.get("possible_emergency_language"):
            trace.mark("Red-flag safety scan", "triggered", "Emergency pattern matched — assessment halted")
            session.state = "EMERGENCY"
            session.accumulated_symptoms = accumulated
            session.clinical_slots = slots
            session.pending_topic = None
            db.commit()
            return ChatResponse(
                reply_text=(
                    (
                        f"{concerns[0]['title']}. {concerns[0]['detail']} "
                        "Please seek in-person care now rather than continuing here."
                    )
                    if (profile_emergency and not emergency_check)
                    else (
                        "What you're describing could be a medical emergency. "
                        "Please seek immediate in-person or emergency medical care now. "
                        "Do not wait for further online assessment."
                    )
                ),
                state="emergency",
                extracted_symptoms=extracted,
                denied_symptoms=sorted(denied),
                intent=detected_intent,
                nlp=self._nlp_trace(analysis),
                turn_count=turn_count + 1,
                matched_pattern=(
                    emergency_check["matched_pattern"] if emergency_check
                    else (concerns[0]["rule"] if profile_emergency else "possible_emergency_language")
                ),
                profile_concerns=concerns,
                emergency_numbers_note="In India, dial 112 for emergency services or 108 for ambulance. For other countries, use your local emergency number.",
                pipeline=trace.stages,
            )
        trace.mark("Red-flag safety scan", "ok", "No emergency patterns matched")

        turn_count += 1
        session.turn_count = turn_count
        session.accumulated_symptoms = accumulated
        session.clinical_slots = slots
        db.commit()

        if not accumulated:
            session.state = "SYMPTOM_COLLECTION"
            session.pending_topic = None
            db.commit()
            prefix = self._distress_prefix(analysis)
            return ChatResponse(
                reply_text=(
                    prefix
                    + "I didn't catch a specific symptom there. Could you describe what you're "
                    "feeling physically? For example: fever, headache, cough, stomach pain, or fatigue."
                ),
                state="symptom_collection",
                extracted_symptoms=[],
                denied_symptoms=sorted(denied),
                intent=detected_intent,
                nlp=self._nlp_trace(analysis),
                turn_count=turn_count,
                suggested_replies=["I have a fever", "I have a headache", "I feel nauseous", "I have body aches"],
                pipeline=trace.stages,
            )

        # --- 4. Prediction -------------------------------------------------
        predictions = predictor.predict(accumulated)
        if not predictions:
            return ChatResponse(
                reply_text="I'm having trouble matching your symptoms to known conditions. Could you tell me more about what you're experiencing?",
                state="symptom_collection",
                extracted_symptoms=extracted,
                denied_symptoms=sorted(denied),
                intent=detected_intent,
                nlp=self._nlp_trace(analysis),
                turn_count=turn_count,
                pipeline=trace.stages,
            )

        top_prob = predictions[0]["probability"]
        second_prob = predictions[1]["probability"] if len(predictions) > 1 else 0
        margin = top_prob - second_prob
        trace.mark(
            "Naive Bayes prediction",
            "ok",
            f"Top: {predictions[0]['disease']} at {top_prob:.0%} (margin {margin:.0%})",
        )

        # --- 5. Decide: another follow-up, or the result? ------------------
        total_planned = followup_svc.total_planned(list(accumulated.keys()))
        model_unsure = top_prob < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD
        below_minimum = asked < followup_svc.MIN_FOLLOWUPS
        under_cap = asked < CLARIFICATION_CAP

        if (below_minimum or model_unsure) and under_cap:
            # caution-level concerns still travel with the turn
            question = await self._next_question(
                predictions, accumulated, slots, user_message, answered_topic, denied
            )
            if question:
                session.state = "CLARIFYING"
                session.followup_asked = asked + 1
                session.pending_topic = question["topic"]
                db.commit()
                trace.mark("Follow-up selection", "ok", f"Asking about {question['topic'].split(':')[-1]}")
                return ChatResponse(
                    reply_text=self._distress_prefix(analysis) + question["question"],
                    state="clarifying",
                    extracted_symptoms=extracted,
                    denied_symptoms=sorted(denied),
                    intent=detected_intent,
                    nlp=self._nlp_trace(analysis),
                    turn_count=turn_count,
                    suggested_replies=question["options"],
                    question_topic=question["topic"],
                    question_rationale=question.get("rationale"),
                    information_gain=question.get("information_gain"),
                    followup_asked=asked + 1,
                    followup_total=max(total_planned, asked + 1),
                    pipeline=trace.stages,
                )

        session.pending_topic = None
        db.commit()
        return await self._build_result_response(
            session, predictions, accumulated, slots, extracted, turn_count,
            asked, total_planned, user_message, db, trace,
            analysis=analysis, detected_intent=detected_intent, denied=denied,
            concerns=concerns,
        )

    # ----------------------------------------------------------------------

    def _apply_answer(
        self, topic: str, message: str, accumulated: dict, slots: dict, feature_order: list[str]
    ) -> Optional[str]:
        """Fold the answer to a pending question into session state."""
        cleaned = message.strip().lower()

        if topic.startswith("symptom:"):
            symptom = topic.split(":", 1)[1]
            if cleaned in NEGATIVE_ANSWERS or cleaned.startswith("no"):
                accumulated.pop(symptom, None)
            elif cleaned.startswith("yes") or cleaned.startswith("a little"):
                if symptom in feature_order:
                    accumulated[symptom] = True
            return topic

        if topic.startswith("slot:"):
            slot_key = topic.split(":", 1)[1]
            implied = followup_svc.record_answer(slot_key, message.strip(), slots)
            if implied and implied in feature_order:
                accumulated[implied] = True
            return topic

        return None

    def _nlp_trace(self, analysis) -> Optional[NlpTrace]:
        """Shape the pipeline output for the client."""
        if analysis is None:
            return None
        return NlpTrace(
            normalized=analysis.normalized,
            entities=[EntityOut(**_entity_payload(e)) for e in analysis.entities],
            normalization_edits=analysis.normalization_edits,
            intent=analysis.intent,
            distress=analysis.distress,
            duration_days=analysis.duration_days,
            duration_bucket=analysis.duration_bucket,
            trend=analysis.trend,
            frequency=analysis.frequency,
            severity=analysis.severity,
            timings=analysis.timings,
            total_ms=analysis.total_ms,
        )

    def _distress_prefix(self, analysis) -> str:
        """
        One sentence of acknowledgement when the user sounds distressed.

        Only fires at the 'elevated' and 'high' bands, so a routine "mild
        cough for two days" is not met with misplaced sympathy.
        """
        if analysis is None:
            return ""
        distress = analysis.distress or {}
        if not distress.get("needs_acknowledgement"):
            return ""
        line = sentiment_nlp.acknowledgement(distress.get("level", ""))
        return (line + "\n\n") if line else ""

    def _absorb_context(self, analysis, slots: dict) -> dict:
        """
        Fold volunteered context into the clinical slots.

        Writes into the same slot keys and the same option strings the
        questionnaire uses, so `followup.next_slot()` treats an extracted
        duration exactly like a tapped button and simply does not ask again.
        """
        if analysis is None:
            return {}

        filled: dict[str, str] = {}

        if analysis.duration_bucket and "duration" not in slots:
            slots["duration"] = analysis.duration_bucket
            filled["duration"] = analysis.duration_bucket

        if analysis.trend and "progression" not in slots:
            mapped = {
                "worsening": "Getting worse",
                "improving": "Getting better",
                "unchanged": "About the same",
            }.get(analysis.trend)
            if mapped:
                slots["progression"] = mapped
                filled["progression"] = mapped

        if analysis.severity and "intensity" not in slots:
            mapped = {
                "mild": "Mild - I can carry on normally",
                "moderate": "Moderate - it's distracting",
                "severe": "Severe - hard to function",
            }.get(analysis.severity)
            if mapped:
                slots["intensity"] = mapped
                filled["intensity"] = mapped

        return filled

    def _answer_about_result(
        self,
        session: Session,
        intent_name: str,
        prediction: dict,
        turn_count: int,
        trace: Timer,
    ) -> ChatResponse:
        """
        Answer a question about the assessment already on screen.

        Everything here comes from the stored prediction -- no re-prediction,
        no new LLM call -- so the answer cannot contradict what the user is
        looking at.
        """
        condition = prediction.get("top_condition", "this condition")
        ayurveda = prediction.get("ayurvedic_guidance") or {}
        triage = prediction.get("triage") or {}
        alternatives = prediction.get("alternatives") or []
        watch_for = prediction.get("watch_for") or []

        if intent_name == "ask_diet":
            if ayurveda:
                reply = (
                    f"For **{condition}**, the Ayurvedic guidance in our knowledge "
                    f"base suggests:\n\n"
                    f"**Diet:** {ayurveda.get('diet', 'no specific guidance available')}\n\n"
                    f"**Lifestyle:** {ayurveda.get('lifestyle', '-')}\n\n"
                    f"**Note:** {ayurveda.get('general_herbs_note', '-')}\n\n"
                    f"*{ayurveda.get('caution', '')}*"
                )
            else:
                reply = (
                    f"I don't have specific dietary guidance stored for "
                    f"{condition}. In general, stay well hydrated, eat light "
                    f"and easily digestible food, and avoid alcohol while you "
                    f"recover. A doctor or dietitian can give advice tailored "
                    f"to you."
                )
        elif intent_name == "ask_doctor":
            reply = (
                f"**{triage.get('label', 'Seek professional advice')}**\n\n"
                f"{triage.get('detail', '')}\n\n"
                "Seek care sooner if you notice any of these:\n"
                + "\n".join(f"- {item}" for item in watch_for)
            )
        elif intent_name == "ask_alternatives":
            if alternatives:
                lines = "\n".join(
                    f"- **{alt['disease']}** - {alt['probability']:.0%}"
                    for alt in alternatives
                )
                reply = (
                    f"Besides {condition} "
                    f"({prediction.get('confidence', 0):.0%}), the model also "
                    f"considered:\n\n{lines}\n\n"
                    f"{prediction.get('reasoning_summary', '')}"
                )
            else:
                reply = (
                    f"{condition} was the only condition that matched your "
                    "symptoms closely enough to report."
                )
        elif intent_name == "ask_medicine":
            reply = (
                "I can't recommend medicines, dosages or prescriptions - that "
                "needs a licensed clinician who can examine you and knows your "
                "history and allergies.\n\n"
                f"What I can tell you is the recommended next step: "
                f"**{triage.get('label', 'see a healthcare professional')}**. "
                f"{triage.get('detail', '')}"
            )
        else:  # ask_prognosis
            reply = (
                f"How {condition} progresses varies from person to person, so I "
                "can't give you a timeline.\n\n"
                f"**Recommended next step:** {triage.get('label', '-')}. "
                f"{triage.get('detail', '')}\n\n"
                "Watch out for:\n"
                + "\n".join(f"- {item}" for item in watch_for)
            )

        return ChatResponse(
            reply_text=reply
            + "\n\n**Disclaimer:** This is educational information, not a "
            "medical diagnosis.",
            state="result",
            turn_count=turn_count,
            intent=intent_name,
            suggested_replies=[
                "What should I eat?",
                "When should I see a doctor?",
                "Tell me about the other possibilities",
            ],
            pipeline=trace.stages,
        )

    async def _next_question(
        self,
        predictions: list[dict],
        accumulated: dict,
        slots: dict,
        user_message: str,
        answered_topic: Optional[str],
        denied: Optional[set] = None,
    ) -> Optional[dict]:
        """
        Alternate between clinical slots and model-discriminating symptoms so
        the consultation covers both context and differential narrowing.
        """
        symptoms = list(accumulated.keys())
        # Genuinely alternate. The previous expression was
        #   prefer_slot = not answered_topic.startswith("symptom:")
        # which is True both when nothing has been asked yet AND after a slot
        # answer, so a slot was chosen on every turn and the symptom branch
        # below was unreachable. Open with a slot for context, then alternate.
        prefer_slot = (not answered_topic) or answered_topic.startswith("symptom:")

        slot = followup_svc.next_slot(slots, symptoms)
        candidates = [p["disease"] for p in predictions[:3]]
        discriminating = predictor.get_discriminating_symptoms(candidates, symptoms)

        # A condition-specific question outranks both the generic slots and
        # the discriminating-symptom question. When the leading candidates
        # are, say, the hepatitis variants, what separates them is an
        # exposure or a history -- asking about another shared symptom
        # cannot move the posterior, so ask the question that can.
        cond_slot = followup_svc.next_condition_slot(slots, candidates, symptoms)
        if cond_slot:
            return {
                "question": cond_slot.question,
                "options": cond_slot.options,
                "topic": f"slot:{cond_slot.key}",
            }

        if prefer_slot and slot:
            return {"question": slot.question, "options": slot.options, "topic": f"slot:{slot.key}"}

        # Expected information gain picks the question that should shrink the
        # posterior's entropy most, which is not always the most
        # "characteristic" symptom: one shared by all the leading candidates
        # cannot separate them, however typical it is. Falls back to the
        # static discriminating-symptom table if the estimator cannot supply
        # per-feature likelihoods.
        asked_topics = {
            k.split(":")[-1] for k in slots.keys() if isinstance(k, str)
        }
        best = infogain.best_question(
            predictor.get_model(),
            predictor.get_feature_order(),
            present=set(symptoms),
            absent=set(denied or []),
            already_asked=asked_topics,
        )
        chosen = best["symptom"] if best else (discriminating[0] if discriminating else None)

        if chosen:
            base = followup_svc.symptom_question(chosen)
            if best:
                base["information_gain"] = best["information_gain"]
                base["rationale"] = infogain.explain(best)
            phrased = await self._phrase_with_llm(
                symptoms, candidates, chosen, user_message
            )
            if phrased:
                base["question"] = phrased
            return base

        if slot:
            return {"question": slot.question, "options": slot.options, "topic": f"slot:{slot.key}"}

        return None

    async def _phrase_with_llm(
        self, symptoms: list[str], candidates: list[str], target: str, user_message: str
    ) -> Optional[str]:
        """Let the LLM word the symptom question naturally, when one is configured."""
        try:
            clarify_prompt = load_prompt("clarifying_question")
            ctx = (
                f"Current symptoms: {', '.join(symptoms)}. "
                f"Candidate conditions: {', '.join(candidates)}. "
                f"Ask about: {target.replace('_', ' ')}. User's message: {user_message}"
            )
            return await chat_completion(clarify_prompt, ctx, temperature=0.4, max_tokens=120)
        except Exception:
            return None

    async def _build_result_response(
        self,
        session: Session,
        predictions: list[dict],
        accumulated: dict,
        slots: dict,
        extracted: list[str],
        turn_count: int,
        asked: int,
        total_planned: int,
        user_message: str,
        db: DBSession,
        trace: Timer,
        analysis=None,
        detected_intent: Optional[str] = None,
        denied: Optional[dict] = None,
        concerns: Optional[list] = None,
    ) -> ChatResponse:
        denied = denied or {}
        top = predictions[0]
        alternatives = [
            Alternative(disease=p["disease"], probability=p["probability"])
            for p in predictions[1:5]   # top-5 posterior for the chart
        ]

        # Abstention. Naming one condition is a claim the evidence has to
        # support: at two known symptoms the calibrated top probability
        # averages ~0.36 and top-1 accuracy is ~0.35 (ml/calibrate.py), so
        # below the threshold the honest output is the differential plus an
        # invitation to say more, not a headline diagnosis.
        abstention = predictor.should_abstain(predictions, len(accumulated))
        if abstention["abstain"]:
            trace.mark(
                "Abstention",
                "triggered",
                f"{abstention['reason']}: top {abstention['top_probability']:.0%} "
                f"on {abstention['n_symptoms']} symptom(s)",
            )
        else:
            trace.mark(
                "Abstention",
                "ok",
                f"evidence supports naming one condition "
                f"({abstention['top_probability']:.0%} on {abstention['n_symptoms']} symptoms)",
            )

        query_text = f"{top['disease']} {' '.join(accumulated.keys())}"
        rag_chunks = retriever.retrieve(top["disease"], query_text, k=4)
        evidence_sources = list(dict.fromkeys(c.get("source_file", "") for c in rag_chunks))
        evidence = [
            EvidenceChunk(
                source_file=c.get("source_file", ""),
                disease_slug=c.get("disease_slug", ""),
                similarity=c.get("similarity"),
                excerpt=(
                    c.get("text", "")[:280]
                    + ("..." if len(c.get("text", "")) > 280 else "")
                ),
            )
            for c in rag_chunks
        ]
        context_text = "\n\n".join(c["text"] for c in rag_chunks) if rag_chunks else "No specific medical context available."
        best_similarity = max(
            (c.get("similarity") or 0.0 for c in rag_chunks), default=0.0
        )
        trace.mark(
            "RAG retrieval",
            "ok" if rag_chunks else "skipped",
            f"{len(rag_chunks)} chunk(s) from {len(evidence_sources)} source file(s)"
            + (f", best similarity {best_similarity:.2f}" if rag_chunks else ""),
        )

        # --- Naive Bayes explanation (exact, not an approximation) --------
        raw_explanation = explain_svc.explain(
            accumulated,
            top_disease=top["disease"],
            runner_up=predictions[1]["disease"] if len(predictions) > 1 else None,
            negated_symptoms=sorted(denied),
        )
        explanation = None
        if raw_explanation:
            explanation = Explanation(
                condition=raw_explanation["condition"],
                compared_with=raw_explanation.get("compared_with"),
                model=raw_explanation["model"],
                weight_meaning=raw_explanation.get("weight_meaning", ""),
                log_prior=raw_explanation.get("log_prior"),
                contributions=[
                    SymptomContribution(**c) for c in raw_explanation["contributions"]
                ],
                top_discriminating=[
                    SymptomContribution(**c)
                    for c in raw_explanation.get("top_discriminating", [])
                ],
                denial_evidence=raw_explanation.get("denial_evidence", []),
                expected_but_not_reported=raw_explanation.get(
                    "expected_but_not_reported", []
                ),
                note=raw_explanation.get("note", ""),
            )
            trace.mark(
                "Explainability",
                "ok",
                explain_svc.summarize(raw_explanation) or "Per-symptom weights computed",
            )
        else:
            trace.mark("Explainability", "skipped", "No reported symptoms to attribute")

        explain_prompt = load_prompt("explain_prediction")
        explain_input = f"""Predicted disease: {top['disease']}
Probability: {top['probability']:.2f}
Alternatives: {json.dumps([{'disease': p['disease'], 'probability': p['probability']} for p in predictions[1:3]])}
User symptoms: {', '.join(accumulated.keys())}
Consultation answers: {json.dumps(slots)}
Medical context:
{context_text}"""

        llm_explain = await chat_completion(explain_prompt, explain_input, temperature=0.3, max_tokens=512)

        # Prefer the exact per-symptom attribution over the generic sentence:
        # it names the symptoms that actually moved the decision.
        reasoning = explain_svc.summarize(raw_explanation) or (
            f"The system identified {', '.join(k.replace('_', ' ') for k in accumulated.keys())} "
            f"as key symptoms, which are most commonly associated with {top['disease']} "
            f"(confidence: {top['probability']:.0%})."
        )
        if llm_explain:
            try:
                cleaned = llm_explain.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                parsed = json.loads(cleaned)
                reasoning = parsed.get("reasoning_summary", reasoning)
                if parsed.get("evidence_sources"):
                    evidence_sources = parsed["evidence_sources"]
            except json.JSONDecodeError:
                reasoning = llm_explain.strip()[:500]
            trace.mark("Explanation", "ok", "Grounded in retrieved knowledge-base context")
        else:
            trace.mark("Explanation", "skipped", "No LLM configured — using template explanation")

        ayurveda = ayurveda_lookup.lookup(top["disease"])
        ayurveda_guidance = None
        if ayurveda:
            ayurveda_guidance = AyurvedicGuidance(
                dosha_imbalance=ayurveda["dosha_imbalance"],
                diet=ayurveda["diet"],
                lifestyle=ayurveda["lifestyle"],
                general_herbs_note=ayurveda["general_herbs_note"],
                caution=ayurveda["caution"],
            )
        trace.mark("Ayurvedic lookup", "ok" if ayurveda else "skipped",
                   ayurveda["dosha_imbalance"] if ayurveda else "No entry for this condition")

        disease_severity = severity_svc.lookup(top["disease"])
        triage_result = followup_svc.triage(disease_severity, slots)
        trace.mark("Triage", "ok", triage_result["label"])

        clinical_context = {
            followup_svc.SLOT_LABELS.get(k, k): v for k, v in slots.items()
        }

        conditions = list(getattr(analysis, "conditions", None) or [])
        if conditions:
            session.mentioned_conditions = conditions

        prediction = Prediction(
            top_condition=top["disease"],
            confidence=top["probability"],
            severity=disease_severity,
            alternatives=alternatives,
            reasoning_summary=reasoning,
            evidence_sources=evidence_sources,
            evidence=evidence,
            explanation=explanation,
            ayurvedic_guidance=ayurveda_guidance,
            triage=Triage(**triage_result),
            clinical_context=clinical_context,
            denied_symptoms=sorted(denied),
            mentioned_conditions=conditions,
            watch_for=WATCH_FOR.get(disease_severity, WATCH_FOR["medium"]),
            abstained=abstention["abstain"],
            abstain_reason=abstention["reason"] or None,
        )

        session.state = "RESULT"
        session.accumulated_symptoms = accumulated
        session.clinical_slots = slots
        session.pending_topic = None
        session.denied_symptoms = denied
        # Stored so a follow-up question ("what should I eat?") can be
        # answered from this exact assessment instead of re-running it.
        session.last_prediction = prediction.model_dump()
        db.commit()

        context_line = ""
        if clinical_context:
            joined = "; ".join(f"{k.lower()}: {v}" for k, v in clinical_context.items())
            context_line = f"Taking into account what you told me ({joined}), "

        denied_line = ""
        if denied:
            denied_line = (
                "You also ruled out "
                + ", ".join(sorted(s.replace("_", " ") for s in denied))
                + ", which the model takes into account.\n\n"
            )

        reply = (
            self._distress_prefix(analysis)
            + f"Thanks for answering those. {context_line}the most likely condition is "
            f"**{top['disease']}** at {top['probability']:.0%} confidence. "
            f"{'This is not a definitive diagnosis. ' if top['probability'] < 0.7 else ''}"
            f"{reasoning}\n\n"
            f"{denied_line}"
            f"**Recommended next step:** {triage_result['label']}. {triage_result['detail']}\n\n"
            f"**Disclaimer:** This is an educational, informational tool and not a medical diagnosis. "
            f"Please consult a licensed healthcare professional."
        )

        return ChatResponse(
            reply_text=reply,
            state="result",
            extracted_symptoms=extracted,
            denied_symptoms=sorted(denied),
            intent=detected_intent,
            nlp=self._nlp_trace(analysis),
            turn_count=turn_count,
            followup_asked=asked,
            followup_total=max(total_planned, asked),
            suggested_replies=[
                "What should I eat?",
                "When should I see a doctor?",
                "Tell me about the other possibilities",
            ],
            pipeline=trace.stages,
            prediction=prediction,
            profile_concerns=concerns or [],
        )
