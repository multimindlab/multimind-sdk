"""
Pure-Python evaluation metrics (BLEU, ROUGE, NDCG, MRR) and score parsing helpers.
"""

import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

_TOKEN_RE = re.compile(r"\w+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LIST_RE = re.compile(r"\[[^\]]*\]")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenization."""
    return _TOKEN_RE.findall(text.lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """
    Sentence BLEU with modified n-gram precision (1..max_n), brevity penalty,
    and add-one smoothing on n-grams of order >= 2.
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = _ngram_counts(cand_tokens, n)
        ref_ngrams = _ngram_counts(ref_tokens, n)
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        if n == 1:
            if overlap == 0:
                return 0.0
            precision = overlap / total
        else:
            precision = (overlap + 1) / (total + 1)
        log_precisions.append(math.log(precision))

    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    if len(cand_tokens) >= len(ref_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(cand_tokens))
    return brevity_penalty * geo_mean


def _prf(overlap: float, cand_total: float, ref_total: float) -> Dict[str, float]:
    precision = overlap / cand_total if cand_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(candidate: str, reference: str, n: int = 1) -> Dict[str, float]:
    """ROUGE-N precision/recall/F1 from n-gram overlap."""
    cand_ngrams = _ngram_counts(tokenize(candidate), n)
    ref_ngrams = _ngram_counts(tokenize(reference), n)
    overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
    return _prf(overlap, sum(cand_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """ROUGE-L precision/recall/F1 based on longest common subsequence."""
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    lcs = _lcs_length(cand_tokens, ref_tokens)
    return _prf(lcs, len(cand_tokens), len(ref_tokens))


def rouge_scores(candidate: str, reference: str) -> Dict[str, float]:
    """F1 scores for ROUGE-1, ROUGE-2 and ROUGE-L."""
    return {
        "rouge1": rouge_n(candidate, reference, 1)["f1"],
        "rouge2": rouge_n(candidate, reference, 2)["f1"],
        "rougeL": rouge_l(candidate, reference)["f1"],
    }


def _dcg(relevances: Sequence[float], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_score(
    relevances: Sequence[float],
    k: Optional[int] = None,
    ideal_relevances: Optional[Sequence[float]] = None,
) -> float:
    """
    NDCG@k over graded (or binary) relevance in retrieved order.
    The ideal ranking defaults to the sorted input relevances.
    """
    if not relevances:
        return 0.0
    k = k or len(relevances)
    ideal = (
        sorted(ideal_relevances, reverse=True)
        if ideal_relevances is not None
        else sorted(relevances, reverse=True)
    )
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return _dcg(relevances, k) / idcg


def reciprocal_rank(relevances: Sequence[float]) -> float:
    """Reciprocal rank of the first relevant (> 0) item."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(relevance_lists: Sequence[Sequence[float]]) -> float:
    """MRR averaged over multiple queries."""
    if not relevance_lists:
        return 0.0
    return sum(reciprocal_rank(rels) for rels in relevance_lists) / len(relevance_lists)


def parse_judge_score(text: str) -> float:
    """Parse a 0-1 score from an LLM judge response, clamped to [0, 1]."""
    match = _NUMBER_RE.search(str(text))
    if not match:
        raise ValueError(f"Could not parse a numeric score from model output: {text!r}")
    return max(0.0, min(1.0, float(match.group())))


def parse_index_list(text: str) -> List[int]:
    """Parse a JSON-style list of integer indices from an LLM response."""
    match = _LIST_RE.search(str(text))
    if not match:
        raise ValueError(f"Could not parse an index list from model output: {text!r}")
    try:
        values = json.loads(match.group())
    except json.JSONDecodeError:
        values = [int(v) for v in _NUMBER_RE.findall(match.group())]
    return [int(v) for v in values]
