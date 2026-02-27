from typing import List
from app.chunking.base import BaseChunkingStrategy
import re

class ContentAwareChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, min_size: int = 100, max_size: int = 1000):
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, text: str) -> List[str]:
        # Split on markdown headings, paragraphs, or Khmer sentence terminator
        paragraphs = re.split(r'\n#{1,6} |\n\n|(?<=។)', text)
        chunks, current = [], ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= self.max_size:
                current += " " + para
            else:
                if len(current) >= self.min_size:
                    chunks.append(current.strip())
                current = para
        if current.strip():
            chunks.append(current.strip())
        return chunks