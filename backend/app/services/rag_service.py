from typing import List, AsyncGenerator
import httpx
from app.services.embedding_service import embedding_service
from app.services.hybrid_search import hybrid_search_service
from app.core.config import settings

class RagService:
    def __init__(self):
        self._last_sources = []

    async def stream_answer(self, question: str, history: list = [], document_id: str = None) -> AsyncGenerator[str, None]:
        query_vector = await embedding_service.embed(question)
        chunks = await hybrid_search_service.search(query_vector, question, top_k=5, document_id=document_id)
        self._last_sources = chunks

        context = "\n\n".join([f"[Source {i+1}]: {c['content']}" for i, c in enumerate(chunks)])
        messages = [{"role": "system", "content": f"You are a helpful assistant. Use the following context to answer:\n\n{context}"}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=120) as client:
            async with client.stream("POST", "/api/chat", json={
                "model": settings.chat_model,
                "messages": messages,
                "stream": True
            }) as response:
                import json
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if token := data.get("message", {}).get("content", ""):
                            yield token
                        if data.get("done"):
                            break

    def get_last_sources(self):
        return [
            {
                "chunk_id": c["id"],
                "document_id": c["document_id"],
                "content": c["content"][:200],
                "score": round(c["score"], 4)
            }
            for c in self._last_sources
        ]