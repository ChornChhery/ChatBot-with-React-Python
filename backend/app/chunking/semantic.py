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


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences depending on detected language.

    English : standard punctuation [.!?] followed by whitespace
    Khmer   : ។ (full stop U+17D4), ៕ (section U+17D5), ៖ (colon U+17D6)
              are the real Khmer sentence/clause terminators
    Thai    : Thai has no sentence-ending character.
              Use pythainlp sent_tokenize if available (best quality).
              Fall back to newline-based splitting otherwise.
              NOTE: ๆ is a repetition mark (like "etc."), NOT a sentence end.
                    ฯ is an abbreviation mark, NOT a sentence end.
                    Both were wrong in the original code.
    """
    lang = _detect_language(text)

    if lang == 'khmer':
        # Split after Khmer sentence terminators, keep the terminator attached
        sentences = re.split(r'(?<=[។៕៖])\s*', text)

    elif lang == 'thai':
        try:
            from pythainlp.tokenize import sent_tokenize
            sentences = sent_tokenize(text, engine='crfcut')
        except ImportError:
            # Without pythainlp: split on newlines only.
            # This is the safest fallback since Thai has no punctuation-based
            # sentence boundary that can be reliably detected with regex.
            sentences = re.split(r'\n+', text)

    else:
        # English: split after [.!?] followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)

    return [s.strip() for s in sentences if s and s.strip()]


def _word_overlap(a: str, b: str) -> float:
    """
    Compute word-level overlap ratio between two text segments.
    For Thai/Khmer, uses character-level overlap since word boundaries
    are not space-separated.
    """
    lang = _detect_language(a + b)

    if lang in ('thai', 'khmer'):
        # Character-level overlap for scripts without spaces
        # Use trigrams for a more meaningful comparison
        def trigrams(s):
            return set(s[i:i+3] for i in range(len(s) - 2))
        sa, sb = trigrams(a), trigrams(b)
    else:
        sa = set(a.lower().split())
        sb = set(b.lower().split())

    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


class SemanticChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, min_size: int = 150, max_size: int = 1200, threshold: float = 0.3):
        self.min_size = min_size
        self.max_size = max_size
        self.threshold = threshold

    def chunk(self, text: str) -> List[str]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks, current = [], sentences[0]

        for sent in sentences[1:]:
            if (len(current) + len(sent) <= self.max_size and
                    _word_overlap(current, sent) >= self.threshold):
                current += " " + sent
            else:
                if len(current) >= self.min_size:
                    chunks.append(current.strip())
                current = sent

        if current.strip():
            chunks.append(current.strip())

        return chunks