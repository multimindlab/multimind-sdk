"""Known-answer tests for the evaluation metrics in multimind.evaluation.metrics."""

import math

import pytest

# AdvancedEvaluator needs the heavy eval stack; metrics.py itself is stdlib-only
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("sklearn")

from multimind.evaluation.advanced_evaluation import AdvancedEvaluator  # noqa: E402
from multimind.evaluation.metrics import (  # noqa: E402
    mean_reciprocal_rank,
    ndcg_score,
    parse_index_list,
    parse_judge_score,
    reciprocal_rank,
    rouge_l,
    rouge_n,
    rouge_scores,
    sentence_bleu,
)


def _has_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


class TestBleu:
    def test_identical(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert sentence_bleu(text, text) == pytest.approx(1.0)

    def test_disjoint(self):
        assert sentence_bleu("apples bananas cherries", "dogs cats birds") == 0.0

    def test_partial_between_zero_and_one(self):
        score = sentence_bleu("the cat sat on the mat", "the cat is sitting on the mat")
        assert 0.0 < score < 1.0

    def test_empty_candidate(self):
        assert sentence_bleu("", "some reference text") == 0.0

    def test_brevity_penalty(self):
        # Short candidate matching a prefix must score below a full match
        short = sentence_bleu("the quick brown", "the quick brown fox jumps")
        full = sentence_bleu("the quick brown fox jumps", "the quick brown fox jumps")
        assert short < full


class TestRouge:
    def test_identical(self):
        text = "the quick brown fox"
        scores = rouge_scores(text, text)
        assert scores["rouge1"] == pytest.approx(1.0)
        assert scores["rouge2"] == pytest.approx(1.0)
        assert scores["rougeL"] == pytest.approx(1.0)

    def test_disjoint(self):
        scores = rouge_scores("apples bananas", "dogs cats")
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_rouge1_known_value(self):
        # candidate "the cat" vs reference "the cat sat": P=1.0, R=2/3, F1=0.8
        result = rouge_n("the cat", "the cat sat", 1)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["f1"] == pytest.approx(0.8)

    def test_rouge_l_known_value(self):
        # LCS("a b c d", "a c b d") = 3, so P = R = F1 = 0.75
        result = rouge_l("a b c d", "a c b d")
        assert result["f1"] == pytest.approx(0.75)


class TestNdcg:
    def test_perfect_ranking(self):
        assert ndcg_score([3, 2, 1]) == pytest.approx(1.0)

    def test_known_binary_value(self):
        # Relevant item at rank 2: DCG = 1/log2(3), IDCG = 1
        assert ndcg_score([0, 1]) == pytest.approx(1 / math.log2(3))

    def test_empty(self):
        assert ndcg_score([]) == 0.0

    def test_no_relevant(self):
        assert ndcg_score([0, 0, 0]) == 0.0

    def test_explicit_ideal(self):
        # One of two relevant docs retrieved at rank 1: IDCG = 1 + 1/log2(3)
        expected = 1.0 / (1.0 + 1 / math.log2(3))
        assert ndcg_score([1, 0], ideal_relevances=[1, 1]) == pytest.approx(expected)


class TestMrr:
    def test_relevant_at_rank_two(self):
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5)

    def test_relevant_at_rank_one(self):
        assert reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert reciprocal_rank([0, 0, 0]) == 0.0

    def test_mean_over_queries(self):
        assert mean_reciprocal_rank([[0, 1], [1, 0]]) == pytest.approx(0.75)

    def test_mean_empty(self):
        assert mean_reciprocal_rank([]) == 0.0


class TestJudgeParsing:
    def test_plain_number(self):
        assert parse_judge_score("0.8") == pytest.approx(0.8)

    def test_number_in_text(self):
        assert parse_judge_score("Score: 0.75 because ...") == pytest.approx(0.75)

    def test_clamped(self):
        assert parse_judge_score("5") == 1.0

    def test_no_number_raises(self):
        with pytest.raises(ValueError):
            parse_judge_score("no score here")

    def test_index_list(self):
        assert parse_index_list("The relevant documents are [0, 2].") == [0, 2]

    def test_index_list_no_list_raises(self):
        with pytest.raises(ValueError):
            parse_index_list("none of them")


class TestEvaluatorMetricMethods:
    """Exercise AdvancedEvaluator metric methods without instantiating it."""

    async def test_ndcg_binary_relevance(self):
        retrieved = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        relevant = [{"id": "b"}]
        score = await AdvancedEvaluator._calculate_ndcg(object(), retrieved, relevant)
        assert score == pytest.approx(1 / math.log2(3))

    async def test_mrr_relevant_at_rank_two(self):
        retrieved = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        relevant = [{"id": "b"}]
        score = await AdvancedEvaluator._calculate_mrr(object(), retrieved, relevant)
        assert score == pytest.approx(0.5)

    async def test_bleu_identical(self):
        text = "the quick brown fox jumps over the lazy dog"
        score = await AdvancedEvaluator._calculate_bleu(object(), text, text)
        assert score == pytest.approx(1.0)

    async def test_rouge_identical(self):
        text = "the quick brown fox"
        scores = await AdvancedEvaluator._calculate_rouge(object(), text, text)
        assert scores["rouge1"] == pytest.approx(1.0)

    @pytest.mark.skipif(_has_module("nltk"), reason="nltk installed; METEOR delegates to it")
    async def test_meteor_fails_honest_without_nltk(self):
        with pytest.raises(NotImplementedError, match="nltk"):
            await AdvancedEvaluator._calculate_meteor(object(), "a b", "a b")

    @pytest.mark.skipif(
        _has_module("bert_score"), reason="bert-score installed; BERTScore delegates to it"
    )
    async def test_bertscore_fails_honest_without_bert_score(self):
        with pytest.raises(NotImplementedError, match="bert-score"):
            await AdvancedEvaluator._calculate_bertscore(object(), "a b", "a b")
