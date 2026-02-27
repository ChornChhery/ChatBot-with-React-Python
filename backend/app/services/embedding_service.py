import json
import httpx
from typing import List
from app.core.config import settings

class EmbeddingService:
    async def embed(self, text: str) -> List[float]:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60) as client:
            response = await client.post("/api/embeddings", json={
                "model": settings.embed_model,
                "prompt": text
            })
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
        return results

embedding_service = EmbeddingService()