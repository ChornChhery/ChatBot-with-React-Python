from typing import List
from app.chunking.base import BaseChunkingStrategy
import re

class SemanticChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, min_size: int = 150, max_size: int = 1200, threshold: float = 0.3):
        self.min_size = min_size
        self.max_size = max_size
        self.threshold = threshold

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r'(?<=[.!?])\s+|(?<=។)\s*|(?<=[ๆฯ])\s+|\n', text)

    def _overlap(self, a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / min(len(sa), len(sb))

    def chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self._split_sentences(text) if s.strip()]
        if not sentences:
            return []
        chunks, current = [], sentences[0]
        for sent in sentences[1:]:
            if (len(current) + len(sent) <= self.max_size and
                    self._overlap(current, sent) >= self.threshold):
                current += " " + sent
            else:
                if len(current) >= self.min_size:
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())
        return chunks