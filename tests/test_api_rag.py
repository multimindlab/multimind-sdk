"""API tests for multimind.gateway.rag_api using TestClient with a mocked RAG backend."""

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
# rag_api.py imports multimind.rag, which needs the [rag] extras (faiss et al.)
# on top of [gateway].
pytest.importorskip("faiss", reason="requires multimind-sdk[gateway,rag]")
from fastapi.testclient import TestClient

import multimind
import multimind.gateway.rag_api as rag_api


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}


class FakeBackend:
    def __init__(self, count):
        self.metadata = list(range(count))


class FakeVectorStore:
    def __init__(self, count):
        self._backend = FakeBackend(count)

    def _get_backend(self):
        return self._backend


class FakeRAG:
    def __init__(self):
        self.docs = []
        self.vector_store = FakeVectorStore(0)

    async def add_documents(self, documents, process=True):
        self.docs.extend(documents)
        self.vector_store = FakeVectorStore(len(self.docs))

    async def retrieve(self, query, k=3, filter_criteria=None):
        return [FakeDoc("relevant text", {"score": 0.9})]

    async def clear(self):
        self.docs = []


class FakeModel:
    model_name = "fake-model"

    async def generate(self, prompt, **kwargs):
        return "generated answer"


@pytest.fixture()
def bare_client(monkeypatch):
    # No RAG backend, no auth configured.
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setattr(rag_api, "rag_instance", None)
    return TestClient(rag_api.app)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setattr(rag_api, "rag_instance", FakeRAG())
    monkeypatch.setattr(rag_api, "current_model", FakeModel())
    return TestClient(rag_api.app)


def test_health_does_not_initialize_backend(bare_client):
    r = bare_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["version"] == multimind.__version__
    assert body["rag_initialized"] is False


def test_ready_503_until_initialized(bare_client):
    assert bare_client.get("/ready").status_code == 503


def test_ready_when_initialized(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_docs_and_openapi(bare_client):
    assert bare_client.get("/docs").status_code == 200
    r = bare_client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["version"] == multimind.__version__
    assert "/query" in schema["paths"]


def test_token_503_when_jwt_not_configured(bare_client):
    r = bare_client.post("/token", data={"username": "u", "password": "p"})
    assert r.status_code == 503


def test_add_documents_happy_path(client):
    r = client.post(
        "/documents",
        json={"documents": [{"text": "hello world", "metadata": {"source": "test"}}]},
    )
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_add_documents_validation_error(client):
    r = client.post("/documents", json={})
    assert r.status_code == 422


def test_query_happy_path(client):
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["documents"][0]["text"] == "relevant text"


def test_query_validation_error(client):
    r = client.post("/query", json={"top_k": 3})
    assert r.status_code == 422


def test_generate_happy_path(client):
    r = client.post("/generate", json={"query": "hello"})
    assert r.status_code == 200
    assert r.json()["text"] == "generated answer"


def test_generate_validation_error(client):
    r = client.post("/generate", json={})
    assert r.status_code == 422


def test_document_count(client):
    client.post("/documents", json={"documents": [{"text": "abc"}]})
    r = client.get("/documents/count")
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_clear_documents(client):
    r = client.delete("/documents")
    assert r.status_code == 200
