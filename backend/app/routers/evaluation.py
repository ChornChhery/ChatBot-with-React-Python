from fastapi import APIRouter
from app.schemas.evaluation import EvaluationRequest
from app.services.evaluation_service import EvaluationService

router = APIRouter()
evaluation_service = EvaluationService()

@router.post("")
async def run_evaluation(request: EvaluationRequest):
    return await evaluation_service.evaluate(
        request.question,
        request.document_id,
        request.top_k
    )