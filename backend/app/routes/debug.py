from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ..services import predictor, retriever, ayurveda_lookup

router = APIRouter(tags=["debug"])


class PredictRequest(BaseModel):
    symptoms: list[str]
    message: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: list[dict]
    symptom_flags: dict[str, bool]
    retrieved_sources: list[str]
    ayurveda: Optional[dict]


@router.post("/debug/predict", response_model=PredictResponse)
async def debug_predict(request: PredictRequest):
    feature_order = predictor.get_feature_order()
    symptom_flags = {}
    for s in feature_order:
        symptom_flags[s] = False
    for symptom in request.symptoms:
        canonical = symptom.lower().strip().replace(" ", "_")
        if canonical in feature_order:
            symptom_flags[canonical] = True

    predictions = predictor.predict(symptom_flags)

    retrieved_sources = []
    ayurveda = None
    if predictions:
        top = predictions[0]["disease"]
        query = f"{top} {' '.join(request.symptoms)}"
        chunks = retriever.retrieve(top, query, k=3)
        retrieved_sources = list(set(c["source_file"] for c in chunks))
        ayurveda = ayurveda_lookup.lookup(top)

    return PredictResponse(
        predictions=predictions,
        symptom_flags=symptom_flags,
        retrieved_sources=retrieved_sources,
        ayurveda=ayurveda,
    )