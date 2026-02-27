import json
import uuid
from typing import List
from sqlalchemy.orm import Session
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.core.enums import DocumentStatus, ChunkingStrategy
from app.services.embedding_service import embedding_service
from app.services.embedding_cache import embedding_cache
from app.chunking.fixed_size import FixedSizeChunkingStrategy
from app.chunking.content_aware import ContentAwareChunkingStrategy
from app.chunking.semantic import SemanticChunkingStrategy
import numpy as np

STRATEGY_MAP = {
    ChunkingStrategy.FIXED_SIZE: FixedSizeChunkingStrategy,
    ChunkingStrategy.CONTENT_AWARE: ContentAwareChunkingStrategy,
    ChunkingStrategy.SEMANTIC: SemanticChunkingStrategy,
}

STRATEGY_NAME_MAP = {
    ChunkingStrategy.FIXED_SIZE: "FixedSize",
    ChunkingStrategy.CONTENT_AWARE: "ContentAware",
    ChunkingStrategy.SEMANTIC: "Semantic",
}

class DocumentService:
    async def process_document(self, db: Session, document_id: str, text: str, strategy: ChunkingStrategy):
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return
        try:
            doc.status = DocumentStatus.PROCESSING
            db.commit()

            chunker = STRATEGY_MAP[strategy]()
            chunks = chunker.chunk(text)
            method_name = STRATEGY_NAME_MAP[strategy]

            new_cache_chunks = []
            for i, chunk_text in enumerate(chunks):
                vector = await embedding_service.embed(chunk_text)
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk_text,
                    chunk_index=i,
                    chunking_method=method_name,
                    embedding_json=json.dumps(vector)
                )
                db.add(chunk)
                new_cache_chunks.append({
                    "id": chunk.id,
                    "document_id": document_id,
                    "content": chunk_text,
                    "vector": np.array(vector, dtype=np.float32)
                })

            doc.status = DocumentStatus.READY
            db.commit()
            embedding_cache.add_chunks(new_cache_chunks)

        except Exception as e:
            doc.status = DocumentStatus.FAILED
            db.commit()
            print(f"[DocumentService] Error processing {document_id}: {e}")

    def delete_document(self, db: Session, document_id: str):
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            embedding_cache.remove_document(document_id)