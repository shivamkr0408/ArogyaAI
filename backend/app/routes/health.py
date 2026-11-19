from fastapi import APIRouter

from ..schemas import HealthResponse
from ..services import predictor, retriever

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded(),
        vector_store_ready=retriever.is_ready(),
    )