"""Tests for OpenAIEmbedder / EmbeddingGenerator with a mocked OpenAI client.

No network: ``openai.AsyncOpenAI`` is patched to return a fake client whose
``embeddings.create`` records calls and returns canned vectors.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from multimind.embeddings.embeddings import (
    EmbeddingConfig,
    EmbeddingGenerator,
    OpenAIEmbedder,
    get_embedder,
)

DIM = 4


class FakeEmbeddings:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.calls = []

    async def create(self, model, input, **kwargs):
        self.calls.append({"model": model, "input": list(input), **kwargs})
        # i-th text -> vector whose first component encodes its batch position
        data = [
            SimpleNamespace(embedding=[float(i + 1)] + [0.0] * (self.dim - 1))
            for i, _ in enumerate(input)
        ]
        return SimpleNamespace(data=data)


class FakeClient:
    def __init__(self, dim=DIM):
        self.embeddings = FakeEmbeddings(dim)


@pytest.fixture
def fake_client():
    client = FakeClient()
    with patch("openai.AsyncOpenAI", return_value=client):
        yield client


async def test_embed_single_batch_returns_one_vector_per_text(fake_client):
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    out = await embedder.embed(["a", "b", "c"])

    assert len(out) == 3
    assert all(len(v) == DIM for v in out)
    assert len(fake_client.embeddings.calls) == 1
    call = fake_client.embeddings.calls[0]
    assert call["model"] == "text-embedding-3-small"
    assert call["input"] == ["a", "b", "c"]


async def test_embed_splits_into_batches_and_preserves_order(fake_client):
    embedder = OpenAIEmbedder(batch_size=2)
    texts = ["t1", "t2", "t3", "t4", "t5"]
    out = await embedder.embed(texts)

    assert len(out) == 5
    inputs = [c["input"] for c in fake_client.embeddings.calls]
    assert inputs == [["t1", "t2"], ["t3", "t4"], ["t5"]]
    # position within each batch is encoded in the first component
    assert [v[0] for v in out] == [1.0, 2.0, 1.0, 2.0, 1.0]


async def test_embed_forwards_init_and_call_kwargs(fake_client):
    embedder = OpenAIEmbedder(dimensions=256)
    await embedder.embed(["x"], user="tester")
    call = fake_client.embeddings.calls[0]
    assert call["dimensions"] == 256
    assert call["user"] == "tester"


async def test_embed_call_kwargs_override_init_kwargs(fake_client):
    embedder = OpenAIEmbedder(dimensions=256)
    await embedder.embed(["x"], dimensions=64)
    assert fake_client.embeddings.calls[0]["dimensions"] == 64


async def test_embed_empty_list_makes_no_calls(fake_client):
    embedder = OpenAIEmbedder()
    assert await embedder.embed([]) == []
    assert fake_client.embeddings.calls == []


async def test_embeddings_str_returns_flat_vector_and_list_returns_nested(fake_client):
    embedder = OpenAIEmbedder()
    single = await embedder.embeddings("hello")
    many = await embedder.embeddings(["a", "b"])

    assert isinstance(single[0], float) and len(single) == DIM
    assert len(many) == 2 and len(many[0]) == DIM


async def test_embedding_dimension_follows_client(monkeypatch):
    client = FakeClient(dim=1536)
    with patch("openai.AsyncOpenAI", return_value=client):
        embedder = OpenAIEmbedder()
    out = await embedder.embeddings("x")
    assert len(out) == 1536


async def test_generation_methods_are_not_supported(fake_client):
    embedder = OpenAIEmbedder()
    with pytest.raises(NotImplementedError):
        await embedder.generate("x")
    with pytest.raises(NotImplementedError):
        await embedder.chat([{"role": "user", "content": "x"}])
    with pytest.raises(NotImplementedError):
        async for _ in embedder.generate_stream("x"):
            pass
    with pytest.raises(NotImplementedError):
        async for _ in embedder.chat_stream([{"role": "user", "content": "x"}]):
            pass
    assert await embedder.get_quality() is None


def test_missing_api_key_raises(monkeypatch):
    import openai

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(openai.OpenAIError, match="api_key"):
        OpenAIEmbedder()


def test_cache_flag_controls_cache_dict(fake_client):
    assert OpenAIEmbedder(cache_enabled=True).cache == {}
    assert OpenAIEmbedder(cache_enabled=False).cache is None


# --- EmbeddingGenerator -------------------------------------------------------


def test_generator_picks_openai_embedder_and_passes_config(fake_client):
    config = EmbeddingConfig(
        model_name="openai/text-embedding-3-small",
        batch_size=7,
        cache_enabled=False,
        custom_params={"dimensions": 128},
    )
    gen = EmbeddingGenerator(config)

    assert isinstance(gen.embedder, OpenAIEmbedder)
    assert gen.embedder.model == "openai/text-embedding-3-small"
    assert gen.embedder.batch_size == 7
    assert gen.embedder.cache is None
    assert gen.embedder.kwargs == {"dimensions": 128}
    stats = gen.get_stats()
    assert stats["embedder_type"] == "OpenAIEmbedder"
    assert stats["config"]["model_name"] == "openai/text-embedding-3-small"


async def test_generator_normalizes_to_unit_vectors(fake_client):
    gen = EmbeddingGenerator(EmbeddingConfig(model_name="openai-ada", normalize=True))
    vectors = await gen.generate(["a", "b"])

    assert len(vectors) == 2
    for v in vectors:
        assert np.linalg.norm(v) == pytest.approx(1.0)


async def test_generator_normalize_false_keeps_raw_vectors(fake_client):
    gen = EmbeddingGenerator(EmbeddingConfig(model_name="openai-ada", normalize=False))
    vectors = await gen.generate(["a", "b"])
    assert [v[0] for v in vectors] == [1.0, 2.0]


async def test_generator_normalize_leaves_zero_vector_untouched(fake_client):
    gen = EmbeddingGenerator(EmbeddingConfig(model_name="openai-ada"))
    assert gen._normalize_embeddings([[0.0, 0.0]]) == [[0.0, 0.0]]


async def test_generator_generate_embedding_single_text(fake_client):
    gen = EmbeddingGenerator(EmbeddingConfig(model_name="openai-ada"))
    await gen.initialize()
    v = await gen.generate_embedding("hello")
    assert len(v) == DIM
    assert isinstance(v[0], float)


# --- factory ------------------------------------------------------------------


def test_get_embedder_openai_and_unsupported(fake_client):
    assert isinstance(get_embedder("openai", model="m"), OpenAIEmbedder)
    with pytest.raises(ValueError, match="Unsupported embedder type"):
        get_embedder("nope")


def test_huggingface_embedder_requires_torch():
    pytest.importorskip("torch", reason="HuggingFaceEmbedder needs torch/transformers")
    pytest.importorskip("transformers", reason="HuggingFaceEmbedder needs transformers")
    from multimind.embeddings.embeddings import HuggingFaceEmbedder

    assert HuggingFaceEmbedder is not None
