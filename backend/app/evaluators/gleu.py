from collections import Counter

class GLEUEvaluator:
    def score(self, reference: str, hypothesis: str, max_n: int = 4) -> float:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0

        total_match = total_ref = total_hyp = 0
        for n in range(1, max_n + 1):
            ref_ng = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
            hyp_ng = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
            total_match += sum((ref_ng & hyp_ng).values())
            total_ref += sum(ref_ng.values())
            total_hyp += sum(hyp_ng.values())

        if total_ref == 0 or total_hyp == 0:
            return 0.0
        precision = total_match / total_hyp
        recall = total_match / total_ref
        return round((precision + recall) / 2, 4)