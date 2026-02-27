import re
from typing import List

class BM25Service:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _detect_language(self, text: str) -> str:
        if re.search(r'[\u0E00-\u0E7F]', text):
            return 'thai'
        if re.search(r'[\u1780-\u17FF]', text):
            return 'khmer'
        return 'english'

    def _tokenize(self, text: str) -> List[str]:
        lang = self._detect_language(text)
        if lang in ('thai', 'khmer'):
            n = 3
            return [text[i:i+n] for i in range(len(text)-n+1)]
        stop_words = {'the','a','an','is','in','on','at','to','for','of','and','or'}
        return [w for w in re.findall(r'\w+', text.lower()) if w not in stop_words]

    def score(self, query: str, documents: List[str]) -> List[float]:
        tokenized_docs = [self._tokenize(d) for d in documents]
        avg_len = sum(len(d) for d in tokenized_docs) / len(tokenized_docs) if tokenized_docs else 1
        query_tokens = self._tokenize(query)
        scores = []

        import math
        N = len(documents)
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
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len))
            scores.append(score)

        max_score = max(scores) if scores else 1
        return [s / max_score if max_score > 0 else 0.0 for s in scores]

bm25_service = BM25Service()