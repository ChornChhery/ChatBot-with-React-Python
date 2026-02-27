import httpx
import json
import re
from app.services.embedding_service import embedding_service
from app.services.hybrid_search import hybrid_search_service
from app.evaluators.bleu import BLEUEvaluator
from app.evaluators.gleu import GLEUEvaluator
from app.evaluators.f1 import F1Evaluator
from app.core.config import settings

bleu = BLEUEvaluator()
gleu = GLEUEvaluator()
f1 = F1Evaluator()


class EvaluationService:
    async def _generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120) as client:
            response = await client.post("/api/chat", json={
                "model": settings.chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            })
            return response.json()["message"]["content"]

    async def evaluate(self, question: str, document_id: str = None, top_k: int = 5):
        query_vector = await embedding_service.embed(question)
        chunks = await hybrid_search_service.search(
            query_vector, question, top_k=top_k, document_id=document_id
        )
        context = "\n\n".join([c["content"] for c in chunks])
        source_docs = list(set(c["document_id"] for c in chunks))

        reference = await self._generate(
            f"Based on this context, write an ideal answer to: {question}\n\nContext:\n{context}"
        )
        rag_answer = await self._generate(
            f"Answer this question using only the context provided.\n\nQuestion: {question}\n\nContext:\n{context}"
        )

        judge_prompt = (
            f"Score this answer from 0-10 for quality and accuracy.\n\n"
            f"Question: {question}\nReference: {reference}\nAnswer: {rag_answer}\n\n"
            f'Respond ONLY in JSON with no extra text, no markdown, no code fences: {{"score": 8, "explanation": "..."}}'
        )
        judge_raw = await self._generate(judge_prompt)

        try:
            # FIX: Strip markdown code fences before parsing.
            # LLMs like llama3.2:3b often wrap JSON in ```json ... ``` which breaks json.loads()
            cleaned = re.sub(r"```(?:json)?|```", "", judge_raw).strip()
            judge_data = json.loads(cleaned)
            judge_score = float(judge_data.get("score", 5)) / 10
            judge_explanation = judge_data.get("explanation", "")
        except (json.JSONDecodeError, ValueError, KeyError):
            # Last resort: try to extract a number from the raw response
            numbers = re.findall(r'\b([0-9]|10)\b', judge_raw)
            judge_score = float(numbers[0]) / 10 if numbers else 0.5
            judge_explanation = judge_raw

        bleu_score = bleu.score(reference, rag_answer)
        gleu_score = gleu.score(reference, rag_answer)
        f1_score = f1.score(reference, rag_answer)
        overall = round((bleu_score + gleu_score + f1_score + judge_score) / 4, 4)

        return {
            "question": question,
            "auto_generated_reference": reference,
            "generated_answer": rag_answer,
            "source_documents": source_docs,
            "bleu_score": bleu_score,
            "gleu_score": gleu_score,
            "f1_score": f1_score,
            "llm_judge_score": judge_score,
            "llm_judge_explanation": judge_explanation,
            "overall_score": overall
        }