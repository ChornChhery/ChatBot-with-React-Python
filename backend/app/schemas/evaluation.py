from pydantic import BaseModel
from typing import Optional, List

class EvaluationRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = 5

class EvaluationResult(BaseModel):
    question: str
    auto_generated_reference: str
    generated_answer: str
    source_documents: List[str]
    bleu_score: float
    gleu_score: float
    f1_score: float
    llm_judge_score: float
    llm_judge_explanation: str
    overall_score: float