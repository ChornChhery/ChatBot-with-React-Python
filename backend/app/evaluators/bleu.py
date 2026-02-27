from collections import Counter
import math

class BLEUEvaluator:
    def score(self, reference: str, hypothesis: str, max_n: int = 4) -> float:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
            hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
            match = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            precisions.append(match / total if total > 0 else 0.0)

        if all(p == 0 for p in precisions):
            return 0.0

        log_avg = sum(math.log(p) for p in precisions if p > 0) / max_n
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if hyp_tokens else 0.0
        return round(bp * math.exp(log_avg), 4)