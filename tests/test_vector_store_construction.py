"""Construction / ``initialize()`` contract for every exported vector-store backend.

For each backend in the registry this checks that:

* the class loads and is concrete (no leftover abstract methods);
* it can be constructed with a minimal config without touching any client lib;
* ``initialize()`` either succeeds (local backends whose dependency is
  installed), raises ``ImportError`` naming the pip package (remote backend,
  client lib missing), or raises ``NotImplementedError`` (experimental stub).
  ``AttributeError`` / ``TypeError`` from ``initialize()`` is a bug.

Remote backends whose client lib *is* installed are skipped at the
``initialize()`` step: that call would open a real connection, which belongs
in ``tests/integration/test_vector_stores_live.py``.
"""

from __future__ import annotations

import importlib.util
import inspect

import pytest

from multimind.vector_store import (
    _BACKEND_MODULES,
    _EXPERIMENTAL_BACKENDS,
    _EXPERIMENTAL_MODULES,
    get_backend_class,
)
from multimind.vector_store.base import VectorStoreBackend, VectorStoreConfig, VectorStoreType

ALL_BACKENDS = sorted(_BACKEND_MODULES) + sorted(_EXPERIMENTAL_MODULES)

# Backends that run fully in-process; initialize() must succeed when the dep is installed.
LOCAL_BACKENDS = {
    "FAISSBackend",
    "AnnoyBackend",
    "SklearnBackend",
    "SQLiteVSSBackend",
    "ChromaBackend",
    "LanceDBBackend",
}

# Importable module name of each backend's client library (None: stdlib only).
CLIENT_MODULE = {
    "AlibabaCloudOpenSearchBackend": "opensearchpy",
    "AnalyticDBBackend": "psycopg2",
    "AnnoyBackend": "annoy",
    "AstraDBBackend": "astrapy",
    "AtlasBackend": "pymongo",
    "AzureCosmosDBBackend": "azure.cosmos",
    "AzureSearchBackend": "azure.search.documents",
    "CassandraBackend": "cassandra",
    "ChromaBackend": "chromadb",
    "ElasticsearchBackend": "elasticsearch",
    "FAISSBackend": "faiss",
    "HippoBackend": "hippo_api",
    "HologresBackend": "hologres_vector",
    "LLMRailsBackend": "llmrails",
    "LanceDBBackend": "lancedb",
    "MarqoBackend": "marqo",
    "MatchingEngineBackend": "google.cloud.aiplatform",
    "MeiliSearchBackend": "meilisearch",
    "MilvusBackend": "pymilvus",
    "MomentoVectorIndexBackend": "momento",
    "MongoDBAtlasBackend": "pymongo",
    "MyScaleBackend": "clickhouse_connect",
    "Neo4jVectorBackend": "neo4j",
    "NucliaDBBackend": "nucliadb_sdk",
    "OpenSearchVectorBackend": "opensearchpy",
    "PGEmbeddingBackend": "psycopg2",
    "PGVectoRSBackend": "psycopg2",
    "PGVectorBackend": "psycopg2",
    "PineconeBackend": "pinecone",
    "QdrantBackend": "qdrant_client",
    "RocksetDBBackend": "rockset",
    "SQLiteVSSBackend": None,
    "SingleStoreDBBackend": "singlestoredb",
    "SklearnBackend": "sklearn",
    "StarRocksBackend": "pymysql",
    "SupabaseVectorStore": "supabase",
    "TairVectorStore": "tair",
    "TencentVectorDBVectorStore": "tcvectordb",
    "TigrisVectorStore": "tigrisdb",
    "TileDBVectorStore": "tiledb",
    "TimescaleVectorStore": "timescale_vector",
    "TypesenseVectorStore": "typesense",
    "USearchVectorStore": "usearch",
    "ValdVectorStore": "vald",
    "VectaraVectorStore": "vectara",
    "WeaviateVectorStore": "weaviate",
    "XataVectorStore": "xata",
    "ZepVectorStore": "zep_python",
    "ZillizVectorStore": "pymilvus",
}

# Backends whose module still imports its client at top level, so the module
# itself cannot load without the dependency.
TOP_LEVEL_IMPORT = {"FAISSBackend", "AnnoyBackend", "SklearnBackend", "ChromaBackend"}


def _has_module(name: str | None) -> bool:
    if name is None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _chroma_settings(tmp_path) -> dict:
    return {
        "is_persistent": True,
        "persist_directory": str(tmp_path / "chroma"),
        "anonymized_telemetry": False,
    }


def _build(name: str, cls, tmp_path) -> VectorStoreBackend:
    """Construct ``cls`` with the smallest config its ``__init__`` accepts.

    Credentials are dummies: nothing may connect at construction time.
    """
    if name == "FAISSBackend":
        return cls(VectorStoreConfig({"dimension": 4}, VectorStoreType.FAISS))
    kwargs = {
        "AlibabaCloudOpenSearchBackend": {"api_key": "k", "endpoint": "example.invalid"},
        "AnalyticDBBackend": {"host": "h", "user": "u", "password": "p", "database": "d"},
        "AnnoyBackend": {"vector_dim": 4},
        "AstraDBBackend": {"token": "t", "api_endpoint": "https://example.invalid"},
        "AtlasBackend": {"uri": "mongodb://example.invalid"},
        "AwaDBBackend": {"api_key": "k", "endpoint": "e"},
        "AzureCosmosDBBackend": {"endpoint": "https://example.invalid", "key": "k"},
        "AzureSearchBackend": {"endpoint": "https://example.invalid", "api_key": "k"},
        "BagelDBBackend": {"api_key": "k", "endpoint": "e"},
        "BaiduCloudVectorSearchBackend": {"api_key": "k", "secret_key": "s", "endpoint": "e"},
        "ChromaBackend": {
            "collection_name": "construction_test",
            "chroma_settings": _chroma_settings(tmp_path),
        },
        "ClickHouseBackend": {"host": "h"},
        "DatabricksVectorSearchBackend": {"api_token": "t", "workspace_url": "u"},
        "ElasticVectorSearchBackend": {"api_key": "k", "endpoint": "e"},
        "HippoBackend": {"api_key": "k", "endpoint": "e"},
        "HologresBackend": {
            "host": "h",
            "port": 80,
            "dbname": "d",
            "user": "u",
            "password": "p",
            "vector_dim": 4,
        },
        "LLMRailsBackend": {"api_key": "k", "datastore_id": "d"},
        "LanceDBBackend": {"uri": str(tmp_path / "lancedb"), "vector_dim": 4},
        "MatchingEngineBackend": {"project": "p", "index_id": "i", "endpoint_id": "e"},
        "MomentoVectorIndexBackend": {"auth_token": "t"},
        "MongoDBAtlasBackend": {"uri": "mongodb://example.invalid"},
        "NucliaDBBackend": {"key": "k", "kb": "kb"},
        "PineconeBackend": {"api_key": "k", "environment": "env"},
        "RocksetDBBackend": {"api_key": "k"},
        "SQLiteVSSBackend": {"db_path": str(tmp_path / "vectors.db"), "dim": 4},
        "SklearnBackend": {"dim": 4},
        "SupabaseVectorStore": {"url": "https://example.invalid", "api_key": "k"},
        "TencentVectorDBVectorStore": {"endpoint": "e", "username": "u", "password": "p"},
        "TigrisVectorStore": {"project": "p"},
        "TileDBVectorStore": {"array_uri": str(tmp_path / "tiledb")},
        "TypesenseVectorStore": {"api_key": "k"},
        "USearchVectorStore": {"index_path": str(tmp_path / "usearch.idx")},
        "VectaraVectorStore": {"customer_id": "c", "api_key": "k", "corpus_id": "1"},
        "XataVectorStore": {"db_url": "https://example.invalid", "api_key": "k"},
        "ZepVectorStore": {"api_url": "https://example.invalid"},
    }.get(name, {})
    return cls(**kwargs)


@pytest.fixture(autouse=True)
def _enable_experimental(monkeypatch):
    monkeypatch.setenv("MULTIMIND_EXPERIMENTAL_VECTOR_STORES", "1")


def _load(name: str):
    if name in TOP_LEVEL_IMPORT:
        pytest.importorskip(CLIENT_MODULE[name])
    cls = get_backend_class(name)
    assert cls is not None, f"{name} failed to load"
    return cls


def test_registry_covers_every_backend_module():
    assert len(ALL_BACKENDS) == 60
    assert set(_EXPERIMENTAL_MODULES) == set(_EXPERIMENTAL_BACKENDS)
    assert not set(_BACKEND_MODULES) & set(_EXPERIMENTAL_MODULES)


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_backend_loads_and_is_concrete(name):
    cls = _load(name)
    assert issubclass(cls, VectorStoreBackend)
    assert not inspect.isabstract(cls), sorted(cls.__abstractmethods__)
    assert "initialize" in cls.__dict__, f"{name} does not override initialize()"


@pytest.mark.parametrize("name", ALL_BACKENDS)
async def test_backend_constructs_and_initializes_cleanly(name, tmp_path):
    cls = _load(name)
    backend = _build(name, cls, tmp_path)
    assert isinstance(backend, VectorStoreBackend)

    if name in _EXPERIMENTAL_BACKENDS:
        # Stubs: initialize() raises NotImplementedError (AwaDB predates this and
        # is a no-op); every data operation must raise NotImplementedError.
        try:
            await backend.initialize()
        except NotImplementedError:
            pass
        with pytest.raises(NotImplementedError):
            await backend.add_vectors([[0.0] * 4], [{}], [{"content": "x"}], ids=["a"])
        return

    dep_installed = _has_module(CLIENT_MODULE[name])
    if name in LOCAL_BACKENDS and dep_installed:
        await backend.initialize()
        await backend.initialize()  # idempotent
        return
    if dep_installed:
        pytest.skip(f"{name}: client lib installed; initialize() would connect (integration)")
    with pytest.raises(ImportError, match="pip install"):
        await backend.initialize()


async def test_chroma_in_process_round_trip(tmp_path):
    pytest.importorskip("chromadb")
    from multimind.vector_store.chroma import ChromaBackend

    backend = ChromaBackend(
        collection_name="roundtrip_test", chroma_settings=_chroma_settings(tmp_path)
    )
    await backend.initialize()
    await backend.add_vectors(
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
        documents=[{"content": "doc-a"}, {"content": "doc-b"}, {"content": "doc-c"}],
        ids=["x", "y", "z"],
    )

    results = await backend.search([1, 0, 0, 0], k=2)
    assert [r.id for r in results][0] == "x"
    assert results[0].document["content"] == "doc-a"
    assert results[0].metadata == {"tag": "a"}
    assert len(results) == 2

    filtered = await backend.search([1, 0, 0, 0], k=3, filter_criteria={"tag": "c"})
    assert [r.id for r in filtered] == ["z"]

    await backend.delete_vectors(["x"])
    after = await backend.search([1, 0, 0, 0], k=3)
    assert "x" not in [r.id for r in after]
    assert len(after) == 2

    # Data survives a fresh backend on the same persist directory.
    reopened = ChromaBackend(
        collection_name="roundtrip_test", chroma_settings=_chroma_settings(tmp_path)
    )
    reopened_results = await reopened.search([0, 1, 0, 0], k=1)
    assert [r.id for r in reopened_results] == ["y"]


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", ["faiss", "chroma"])
async def test_unified_vector_store_round_trip(tmp_path, store_type):
    """VectorStore must construct both config-style and kwargs-style backends."""
    from multimind.vector_store.vector_store import VectorStore

    if store_type == "faiss":
        pytest.importorskip("faiss")
        config = VectorStoreConfig({"dimension": 3}, VectorStoreType.FAISS)
    else:
        pytest.importorskip("chromadb")
        config = VectorStoreConfig(
            {
                "collection_name": "unified_test",
                "dimension": 3,
                "persist_directory": str(tmp_path / "chroma"),
                "chroma_settings": {"anonymized_telemetry": False},
            },
            VectorStoreType.CHROMA,
        )

    store = VectorStore(config)
    await store.initialize()
    await store.add_vectors(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [{"tag": "a"}, {"tag": "b"}],
        [{"text": "alpha"}, {"text": "beta"}],
        ids=["a", "b"],
    )
    results = await store.search([1.0, 0.0, 0.0], k=1)
    assert results and results[0].id == "a"
    await store.delete_vectors(["a"])
    assert all(r.id != "a" for r in await store.search([1.0, 0.0, 0.0], k=2))


def test_unified_vector_store_requires_store_type():
    from multimind.vector_store.vector_store import VectorStore

    with pytest.raises(ValueError, match="store_type"):
        VectorStore(VectorStoreConfig({"dimension": 3}))._get_backend()
