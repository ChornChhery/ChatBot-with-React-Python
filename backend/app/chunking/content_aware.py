from typing import List
from app.chunking.base import BaseChunkingStrategy
import re


def _detect_language(text: str) -> str:
    """Detect dominant language in text by character frequency."""
    thai_count = len(re.findall(r'[\u0E00-\u0E7F]', text))
    khmer_count = len(re.findall(r'[\u1780-\u17FF]', text))
    if thai_count > khmer_count and thai_count > 10:
        return 'thai'
    if khmer_count > thai_count and khmer_count > 10:
        return 'khmer'
    return 'english'


def _split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraph-level segments depending on language.

    English/Markdown : split on headings (# ## etc) and blank lines
    Khmer            : split on ។ (khan) which is the Khmer full stop,
                       and also on blank lines
    Thai             : Thai has NO sentence-ending punctuation character.
                       Split on Thai-specific line break patterns and
                       blank lines. Optionally use pythainlp if available
                       for sentence segmentation.
    """
    lang = _detect_language(text)

    if lang == 'khmer':
        # ។ = Khmer full stop (U+17D4), ៕ = Khmer section separator (U+17D5)
        # Split after these characters, and also on blank lines
        parts = re.split(r'(?<=[។៕])\s*|\n\n+', text)

    elif lang == 'thai':
        # Thai has no standard sentence-ending character.
        # Try pythainlp sentence tokenizer first (best quality).
        # Fall back to splitting on blank lines + Thai-style line breaks.
        try:
            from pythainlp.tokenize import sent_tokenize
            parts = sent_tokenize(text, engine='crfcut')
        except ImportError:
            # Without pythainlp: split on blank lines and newlines.
            # Thai paragraphs are usually separated by \n or \n\n in practice.
            parts = re.split(r'\n\n+|\n(?=\s)', text)

    else:
        # English / Markdown
        parts = re.split(r'\n#{1,6} |\n\n+', text)

    return [p.strip() for p in parts if p and p.strip()]


class ContentAwareChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, min_size: int = 100, max_size: int = 1000):
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, text: str) -> List[str]:
        paragraphs = _split_paragraphs(text)
        chunks, current = [], ""

        for para in paragraphs:
            if len(current) + len(para) <= self.max_size:
                current += (" " if current else "") + para
            else:
                if len(current) >= self.min_size:
                    chunks.append(current.strip())
                current = para

        if current.strip():
            chunks.append(current.strip())

        return chunks