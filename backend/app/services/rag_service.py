from typing import List, AsyncGenerator, Optional
import httpx
from app.services.embedding_service import embedding_service
from app.services.hybrid_search import hybrid_search_service
from app.core.config import settings

# Minimum score for RAG chunks to be considered useful enough to include in the answer.
# If no chunks meet this threshold, the LLM answers from its own knowledge only.
RAG_SCORE_THRESHOLD = 0.65


class RagService:
    def __init__(self):
        self._last_sources = []
        self._rag_used = False

    # history=None instead of history=[] to avoid mutable default argument bug
    async def stream_answer(
        self,
        question: str,
        history: Optional[List] = None,
        document_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        if history is None:
            history = []

        print(f"\n[RAG] ── New Question ──────────────────────────")
        print(f"[RAG] Question   : {question}")
        print(f"[RAG] Doc filter : {document_id or 'All documents'}")

        query_vector = await embedding_service.embed(question)
        chunks = await hybrid_search_service.search(
            query_vector, question, top_k=5, document_id=document_id
        )

        print(f"[RAG] Chunks found (before threshold): {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"[RAG]   #{i+1} score={c['score']:.4f} | {c['content'][:80].strip()!r}")

        # Only use RAG context if best chunk score is above threshold.
        # If chunks are weak/irrelevant, answer with LLM knowledge only.
        high_quality_chunks = [c for c in chunks if c["score"] >= RAG_SCORE_THRESHOLD]
        self._last_sources = high_quality_chunks
        self._rag_used = len(high_quality_chunks) > 0

        if self._rag_used:
            print(f"[RAG] Mode       : LLM + RAG ({len(high_quality_chunks)} chunks passed threshold {RAG_SCORE_THRESHOLD})")
            context = "\n\n".join(
                [f"[Source {i+1}]: {c['content']}" for i, c in enumerate(high_quality_chunks)]
            )
            system_prompt = (
                "You are a helpful assistant. Use the following context from the user's documents "
                "to answer the question. If the context does not contain enough information, "
                "say so honestly.\n\n"
                f"Context:\n{context}"
            )
        else:
            # No relevant chunks found — answer from LLM knowledge only
            print(f"[RAG] Mode       : LLM only (no chunks above threshold {RAG_SCORE_THRESHOLD})")
            system_prompt = (
                "You are a helpful assistant. Answer the question using your own knowledge. "
                "No relevant document context was found for this query."
            )

        messages = [{"role": "system", "content": system_prompt}]

        # Filter out any empty-content messages from interrupted streams
        for msg in history:
            if msg.get("content", "").strip():
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

    def get_rag_used(self) -> bool:
        return self._rag_used