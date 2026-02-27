import numpy as np
from typing import List, Dict
from app.services.embedding_cache import embedding_cache
from app.services.bm25_service import bm25_service
from app.core.config import settings

class HybridSearchService:
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def search(self, query_vector: List[float], query_text: str, top_k: int = 5, document_id: str = None) -> List[Dict]:
        cached = embedding_cache.get_all()
        if document_id:
            cached = [c for c in cached if c["document_id"] == document_id]
        if not cached:
            return []

        qv = np.array(query_vector, dtype=np.float32)
        contents = [c["content"] for c in cached]
        bm25_scores = bm25_service.score(query_text, contents)
        vw = settings.vector_weight

        results = []
        for i, chunk in enumerate(cached):
            vec_score = self.cosine_similarity(qv, chunk["vector"])
            final_score = vw * vec_score + (1 - vw) * bm25_scores[i]
            if final_score >= settings.min_similarity_threshold:
                results.append({**chunk, "score": final_score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

hybrid_search_service = HybridSearchService()