from typing import List
from app.chunking.base import BaseChunkingStrategy

class FixedSizeChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.overlap
        return [c for c in chunks if c]