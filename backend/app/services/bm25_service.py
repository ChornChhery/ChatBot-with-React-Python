import re
import math
from typing import List


def _detect_language(text: str) -> str:
    """Detect dominant language in text by character frequency."""
    thai_count = len(re.findall(r'[\u0E00-\u0E7F]', text))
    khmer_count = len(re.findall(r'[\u1780-\u17FF]', text))
    if thai_count > khmer_count and thai_count > 10:
        return 'thai'
    if khmer_count > thai_count and khmer_count > 10:
        return 'khmer'
    return 'english'


class BM25Service:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        lang = _detect_language(text)

        if lang == 'thai':
            # Best quality: use pythainlp word tokenizer (newmm engine)
            # Falls back to trigrams if pythainlp is not installed.
            try:
                from pythainlp.tokenize import word_tokenize
                tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
                return [t for t in tokens if t and t.strip()]
            except ImportError:
                n = 3
                return [text[i:i+n] for i in range(len(text) - n + 1)]

        elif lang == 'khmer':
            # Khmer has no spaces between words — trigrams are the standard
            # approach without a dedicated Khmer NLP library
            n = 3
            return [text[i:i+n] for i in range(len(text) - n + 1)]

        else:
            # English: word tokenization with stopword removal
            stop_words = {
                'the', 'a', 'an', 'is', 'in', 'on', 'at', 'to',
                'for', 'of', 'and', 'or', 'it', 'its', 'be', 'was',
                'are', 'were', 'that', 'this', 'with', 'as', 'by'
            }
            return [w for w in re.findall(r'\w+', text.lower()) if w not in stop_words]

    def score(self, query: str, documents: List[str]) -> List[float]:
        tokenized_docs = [self._tokenize(d) for d in documents]
        avg_len = sum(len(d) for d in tokenized_docs) / len(tokenized_docs) if tokenized_docs else 1
        query_tokens = self._tokenize(query)
        N = len(documents)
        scores = []

        for doc_tokens in tokenized_docs:
            doc_len = len(doc_tokens)
            score = 0.0
            tf_map = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            for qt in query_tokens:
                tf = tf_map.get(qt, 0)
                df = sum(1 for d in tokenized_docs if qt in d)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
                )
            scores.append(score)

        max_score = max(scores) if scores else 1
        return [s / max_score if max_score > 0 else 0.0 for s in scores]


bm25_service = BM25Service()