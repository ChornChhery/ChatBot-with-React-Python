class F1Evaluator:
    def score(self, reference: str, hypothesis: str) -> float:
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if not ref_tokens or not hyp_tokens:
            return 0.0
        common = ref_tokens & hyp_tokens
        if not common:
            return 0.0
        precision = len(common) / len(hyp_tokens)
        recall = len(common) / len(ref_tokens)
        return round(2 * precision * recall / (precision + recall), 4)