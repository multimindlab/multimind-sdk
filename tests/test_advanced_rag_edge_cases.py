"""Edge-case tests for the Advanced RAG "Beta" feature area.

Covers ``multimind.retrieval.retriever.Retriever``/``RetrievalResult`` (metadata
filtering / malformed-input handling), ``multimind.retrieval.enhanced_retrieval
.EnhancedRetriever`` (fusion weights / feedback / empty results), and
``multimind.retrieval.retrieval.HybridRetriever`` (empty-corpus / oversized-k
behavior). All of these are pure-Python + numpy/sklearn/networkx (already
installed in this venv) — no torch needed.
"""

import math
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("numpy")
pytest.importorskip("networkx")

from multimind.core.exceptions import RetrievalError
from multimind.retrieval.enhanced_retrieval import EnhancedRetriever, RetrievalType
from multimind.retrieval.retriever import RetrievalConfig, RetrievalResult, Retriever


class _StubSearchResult:
    """Duck-types the vector store's search-result shape used by Retriever.retrieve."""

    def __init__(self, score, content="doc", metadata=None, doc_id="id"):
        self.score = score
        self._content = content
        self.metadata = metadata
        self.id = doc_id

    def get_content(self):
        return self._content


def _make_retriever(search_return, top_k=5, similarity_threshold=0.7):
    config = RetrievalConfig(
        vector_store=AsyncMock(),
        document_processor=AsyncMock(),
        embedding_generator=AsyncMock(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    config.embedding_generator.generate_embedding = AsyncMock(return_value=[0.1, 0.2])
    config.vector_store.search = AsyncMock(return_value=search_return)
    return Retriever(config)


# --- RetrievalResult: malformed-input validation ----------------------------


def test_retrieval_result_rejects_non_string_content():
    with pytest.raises(ValueError, match="content must be a string"):
        RetrievalResult(content=123, score=0.5, metadata={})


def test_retrieval_result_rejects_non_numeric_score():
    with pytest.raises(ValueError, match="score must be a number"):
        RetrievalResult(content="text", score="high", metadata={})


def test_retrieval_result_rejects_non_dict_metadata():
    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        RetrievalResult(content="text", score=0.5, metadata=["not", "a", "dict"])


# --- Retriever.retrieve: empty corpus / oversized k / threshold boundaries -


@pytest.mark.asyncio
async def test_retriever_retrieve_empty_corpus_returns_empty_list():
    retriever = _make_retriever(search_return=[])
    results = await retriever.retrieve("query")
    assert results == []


@pytest.mark.asyncio
async def test_retriever_retrieve_k_larger_than_corpus_returns_all_available():
    """top_k requested (100) exceeds the corpus size (2) — should just return
    what's available, not error."""
    stub_results = [
        _StubSearchResult(score=0.9, content="a", metadata={}, doc_id="a"),
        _StubSearchResult(score=0.8, content="b", metadata={}, doc_id="b"),
    ]
    retriever = _make_retriever(search_return=stub_results, top_k=100)
    results = await retriever.retrieve("query", top_k=100)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_retriever_retrieve_similarity_threshold_filters_out_all_matches():
    """Every candidate scores below the similarity threshold -> zero results,
    not an error."""
    stub_results = [_StubSearchResult(score=0.1, content="a", metadata={})]
    retriever = _make_retriever(search_return=stub_results, similarity_threshold=0.9)
    results = await retriever.retrieve("query")
    assert results == []


@pytest.mark.asyncio
async def test_retriever_retrieve_similarity_threshold_zero_matches_everything():
    stub_results = [
        _StubSearchResult(score=0.01, content="a", metadata={}),
        _StubSearchResult(score=0.99, content="b", metadata={}),
    ]
    retriever = _make_retriever(search_return=stub_results, similarity_threshold=0.0)
    results = await retriever.retrieve("query")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_retriever_retrieve_malformed_non_dict_metadata_is_coerced():
    """A backend returning malformed (non-dict) metadata must not crash
    retrieval — Retriever.retrieve() coerces it to {}."""
    stub_results = [_StubSearchResult(score=0.9, content="a", metadata="not-a-dict")]
    retriever = _make_retriever(search_return=stub_results)
    results = await retriever.retrieve("query")
    assert len(results) == 1
    assert results[0].metadata == {}
    assert results[0].source is None


@pytest.mark.asyncio
async def test_retriever_retrieve_wraps_backend_errors_in_retrieval_error():
    config = RetrievalConfig(
        vector_store=AsyncMock(),
        document_processor=AsyncMock(),
        embedding_generator=AsyncMock(),
    )
    config.embedding_generator.generate_embedding = AsyncMock(side_effect=RuntimeError("boom"))
    retriever = Retriever(config)
    with pytest.raises(RetrievalError):
        await retriever.retrieve("query")


# --- EnhancedRetriever: fusion weights / feedback / empty results -----------


def _make_enhanced_retriever(base_retrieve_return):
    base_retriever = AsyncMock()
    base_retriever.retrieve = AsyncMock(return_value=base_retrieve_return)
    return EnhancedRetriever(model=AsyncMock(), base_retriever=base_retriever)


@pytest.mark.asyncio
async def test_enhanced_retriever_empty_corpus_returns_empty_for_every_strategy():
    retriever = _make_enhanced_retriever([])
    for strategy in RetrievalType:
        results = await retriever.retrieve("query", retrieval_type=strategy)
        assert results == []


def test_set_fusion_weights_all_zero_does_not_crash_or_produce_nan():
    """Degenerate all-zero weights must not trigger a ZeroDivisionError or
    leave NaNs in the fusion weights (guarded by `total if total > 0 else 1.0`)."""
    retriever = _make_enhanced_retriever([])
    retriever.set_fusion_weights({"hierarchical": 0.0, "temporal": 0.0, "domain": 0.0, "multi_lingual": 0.0})
    weights = retriever.get_fusion_explanation()
    assert all(w == 0.0 for w in weights.values())
    assert not any(math.isnan(w) for w in weights.values())


def test_record_feedback_unknown_strategy_is_ignored():
    retriever = _make_enhanced_retriever([])
    before = retriever.get_fusion_explanation()
    retriever.record_feedback(strategy="does-not-exist", success=True)
    assert retriever.get_fusion_explanation() == before


def test_custom_fusion_fn_overrides_default_combination():
    retriever = _make_enhanced_retriever([])
    sentinel = [{"id": "custom", "content": "x", "score": 1.0}]
    retriever.set_custom_fusion(lambda *args, **kwargs: sentinel)

    combined = retriever._combine_results(
        hierarchical_results=[{"id": "h", "content": "y", "score": 0.5}],
        temporal_results=[],
        domain_results=[],
        multi_lingual_results=[],
    )
    assert combined is sentinel


def test_combine_results_all_strategies_empty_returns_empty_list():
    retriever = _make_enhanced_retriever([])
    combined = retriever._combine_results(
        hierarchical_results=[], temporal_results=[], domain_results=[], multi_lingual_results=[]
    )
    assert combined == []


# --- HybridRetriever: empty corpus / oversized k ----------------------------


class _DummyDenseRetriever:
    """Distinct per-document embeddings, so score normalization doesn't degenerate
    to a zero-range (all-identical) division."""

    async def embeddings(self, texts):
        return [[float(i + 1), 0.0, 1.0] for i, _ in enumerate(texts)]


@pytest.mark.asyncio
async def test_hybrid_retriever_empty_corpus_raises_clear_error():
    pytest.importorskip("sklearn")
    from multimind.retrieval.retrieval import HybridRetriever

    retriever = HybridRetriever(dense_retriever=_DummyDenseRetriever())
    with pytest.raises(ValueError, match="empty vocabulary"):
        await retriever.retrieve("query", documents=[], metadata=[], k=3, use_reranking=False)


@pytest.mark.asyncio
async def test_hybrid_retriever_k_larger_than_corpus_returns_all_available():
    """Regression test for a real bug: `await x.embeddings([query])[0]` indexed
    the coroutine before awaiting it and always raised TypeError — fixed to
    `(await x.embeddings([query]))[0]`."""
    pytest.importorskip("sklearn")
    from multimind.retrieval.retrieval import HybridRetriever

    retriever = HybridRetriever(dense_retriever=_DummyDenseRetriever())
    docs = ["hello world", "goodbye world"]
    metadata = [{}, {}]

    results = await retriever.retrieve("hello", docs, metadata, k=10, use_reranking=False)

    assert len(results) == 2
    assert {r.document for r in results} == set(docs)
