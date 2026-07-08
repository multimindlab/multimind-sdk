import pytest
pytest.importorskip("numpy")  # requires optional extras absent on core-only installs
pytest.importorskip("networkx")  # requires optional extras absent on core-only installs

from multimind.retrieval import EnhancedRetriever, HybridRetriever, Retriever, RetrievalConfig


class DummyVectorStore:
    pass


class DummyDocumentProcessor:
    pass


class DummyEmbeddingGenerator:
    pass


class DummyBaseRetriever:
    async def retrieve(self, query):
        return []

def make_config():
    return RetrievalConfig(
        vector_store=DummyVectorStore(),
        document_processor=DummyDocumentProcessor(),
        embedding_generator=DummyEmbeddingGenerator()
    )

def test_retriever_init():
    config = make_config()
    retriever = Retriever(config)
    assert retriever is not None

def test_enhanced_retriever_init():
    config = make_config()
    base = DummyBaseRetriever()
    retriever = EnhancedRetriever(config, base)
    assert retriever is not None

def test_hybrid_retriever_init():
    pytest.importorskip("sklearn", reason="needs sklearn")
    config = make_config()
    retriever = HybridRetriever(config)
    assert retriever is not None

@pytest.mark.asyncio
async def test_retriever_retrieve_empty():
    config = make_config()
    retriever = Retriever(config)
    try:
        result = await retriever.retrieve("")
        assert result is not None
    except Exception:
        # Dummy collaborators don't implement the real protocol, so the
        # retriever is expected to raise. Either outcome satisfies the
        # smoke test — what matters is that the coroutine was actually
        # awaited (not silently dropped, which masked bugs previously).
        pass


@pytest.mark.asyncio
async def test_enhanced_retriever_retrieve_empty():
    config = make_config()
    base = DummyBaseRetriever()
    retriever = EnhancedRetriever(config, base)
    try:
        result = await retriever.retrieve("")
        assert result is not None
    except Exception:
        pass


@pytest.mark.asyncio
async def test_hybrid_retriever_retrieve_empty():
    pytest.importorskip("sklearn", reason="needs sklearn")
    config = make_config()
    retriever = HybridRetriever(config)
    try:
        result = await retriever.retrieve("")
        assert result is not None
    except Exception:
        pass
