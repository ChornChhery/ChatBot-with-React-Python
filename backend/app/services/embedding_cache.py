import json
import numpy as np
from threading import Lock
from typing import Dict, List

class EmbeddingCacheService:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._lock = Lock()
        self._loaded = False

    def warm_up(self, db):
        from app.models.document_chunk import DocumentChunk
        from app.models.document import Document
        chunks = db.query(DocumentChunk).join(Document)\
            .filter(Document.status == "Ready").all()
        with self._lock:
            for chunk in chunks:
                if chunk.embedding_json:
                    self._cache[chunk.id] = {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "vector": np.array(json.loads(chunk.embedding_json), dtype=np.float32)
                    }
            self._loaded = True
        print(f"[Cache] Loaded {len(self._cache)} chunks into memory")

    def get_all(self) -> List[dict]:
        return list(self._cache.values())

    def add_chunks(self, chunks: List[dict]):
        with self._lock:
            for chunk in chunks:
                self._cache[chunk["id"]] = chunk

    def remove_document(self, document_id: str):
        with self._lock:
            keys = [k for k, v in self._cache.items()
                    if v["document_id"] == document_id]
            for k in keys:
                del self._cache[k]

    @property
    def stats(self):
        total = len(self._cache)
        docs = len(set(v["document_id"] for v in self._cache.values())) if self._cache else 0
        mem_mb = round(total * 4 * 1024 / (1024**2), 2)
        return {
            "totalChunks": total,
            "totalDocuments": docs,
            "isLoaded": self._loaded,
            "estimatedMemoryMb": mem_mb
        }

embedding_cache = EmbeddingCacheService()