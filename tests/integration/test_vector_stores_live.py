"""Integration round-trip tests for vector-store backends.

Two groups, both under ``@pytest.mark.integration`` (excluded from the default
suite; run explicitly with ``pytest -m integration``):

1. Local backends that need no external service: FAISS, sklearn, Annoy,
   SQLiteVSS. Each is ``importorskip``'d so a minimal install still collects
   cleanly; when the optional dependency is present these exercise real
   add/search/delete round-trips against tiny in-memory vectors.

2. Live-service backends (Chroma, Qdrant, Pinecone, Weaviate, Milvus) that
   need a reachable server/API key. Each test reads the backend's connection
   env vars and skips cleanly when they are not set, so CI can enable them
   later simply by providing credentials — no code changes required.

Everything here talks to real backend code (no mocking of the vector-store
layer itself); only the presence of optional deps / live services is gated.
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytestmark = pytest.mark.integration


def _require_env(*names: str) -> None:
    """Skip the current test unless at least one of ``names`` is set."""
    if not any(os.environ.get(n) for n in names):
        pytest.skip(
            f"None of {names} set; export to run this live-service integration test"
        )


# Note on faiss/sqlite-vss coexistence: sqlite-vss vendors its own private
# libfaiss inside its vss0/vector0 extensions. If the `faiss` PyPI package's
# libfaiss loads into the process *first*, the two collide (native symbol
# conflict) and every insert fails with a nonsensical "add_with_ids not
# implemented" error. `tests/conftest.py` pre-loads sqlite-vss's extension
# pair into a throwaway connection before it imports `faiss` for its
# HAS_FAISS flag, which fixes the load order for the whole session.


# --------------------------------------------------------------------------
# Local backends (no external service required)
# --------------------------------------------------------------------------


class TestFAISSBackend:
    def _backend(self):
        faiss = pytest.importorskip("faiss")
        del faiss
        from multimind.vector_store.base import VectorStoreConfig, VectorStoreType
        from multimind.vector_store.faiss import FAISSBackend

        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS, connection_params={"dimension": 4}
        )
        return FAISSBackend(config)

    async def test_add_search_round_trip(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
            documents=[{"content": "doc-a"}, {"content": "doc-b"}, {"content": "doc-c"}],
            ids=["x", "y", "z"],
        )
        results = await backend.search([1, 0, 0, 0], k=2)
        assert [r.id for r in results][:1] == ["x"]
        assert len(results) == 2

    async def test_delete_removes_vector(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadatas=[{}, {}],
            documents=[{"content": "a"}, {"content": "b"}],
            ids=["x", "y"],
        )
        await backend.delete_vectors(["y"])
        results = await backend.search([0, 1, 0, 0], k=5)
        assert "y" not in {r.id for r in results}
        assert "x" in {r.id for r in results}

    async def test_clear_empties_the_index(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["x"]
        )
        await backend.clear()
        assert await backend.search([1, 0, 0, 0], k=5) == []

    async def test_persist_and_load_round_trip(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadatas=[{"tag": "a"}, {"tag": "b"}],
            documents=[{"content": "a"}, {"content": "b"}],
            ids=["x", "y"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.persist(tmpdir)
            from multimind.vector_store.faiss import FAISSBackend

            reloaded = await FAISSBackend.load(tmpdir, backend.config)
            results = await reloaded.search([1, 0, 0, 0], k=2)
            assert {r.id for r in results} == {"x", "y"}


class TestSklearnBackend:
    def _backend(self):
        pytest.importorskip("sklearn")
        from multimind.vector_store.sklearn import SklearnBackend

        return SklearnBackend(dim=4)

    async def test_add_search_delete_round_trip(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            metadatas=[{}, {}, {}],
            documents=[{"content": "a"}, {"content": "b"}, {"content": "c"}],
            ids=["x", "y", "z"],
        )
        results = await backend.search([1, 0, 0, 0], k=1)
        assert results[0].id == "x"
        assert results[0].vector is not None

        await backend.delete_vectors(["x"])
        results_after = await backend.search([1, 0, 0, 0], k=3)
        assert "x" not in {r.id for r in results_after}
        assert {r.id for r in results_after} == {"y", "z"}

    async def test_clear_empties_the_store(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["x"]
        )
        await backend.clear()
        assert await backend.search([1, 0, 0, 0], k=5) == []

    async def test_search_k_larger_than_dataset_does_not_raise(self):
        """sklearn's kneighbors raises if k exceeds fitted samples; backend clamps it."""
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["x"]
        )
        results = await backend.search([1, 0, 0, 0], k=50)
        assert [r.id for r in results] == ["x"]


class TestAnnoyBackend:
    def _backend(self):
        pytest.importorskip("annoy")
        from multimind.vector_store.annoy import AnnoyBackend

        return AnnoyBackend(vector_dim=4, n_trees=10)

    async def test_add_and_search_finds_nearest(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadatas=[{}, {}],
            documents=[{"content": "a"}, {"content": "b"}],
            ids=["x", "y"],
        )
        # Annoy is an approximate index; on tiny datasets it may legitimately
        # return fewer than k candidates, so only the top match is asserted.
        results = await backend.search([1, 0, 0, 0], k=2)
        assert len(results) >= 1
        assert results[0].id == "x"

    async def test_delete_then_add_rebuilds_index(self):
        """Delete+re-add bookkeeping is exact even though Annoy's own search
        is approximate (and, per Annoy's own docs, unreliable on datasets
        this tiny/symmetric — so search recall itself isn't asserted here;
        see test_add_and_search_finds_nearest for a search-correctness case).
        """
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadatas=[{}, {}],
            documents=[{"content": "a"}, {"content": "b"}],
            ids=["x", "y"],
        )
        await backend.delete_vectors(["y"])
        assert "y" not in backend.id_map

        await backend.add_vectors(
            vectors=[[0, 0, 0, 1]], metadatas=[{}], documents=[{"content": "w"}], ids=["w"]
        )
        # Bookkeeping stays exact and dense (no stale or gapped indices).
        assert set(backend.id_map) == {"x", "w"}
        assert sorted(backend.id_map.values()) == [0, 1]
        assert backend.documents["w"] == {"content": "w"}
        assert "y" not in backend.metadata and "y" not in backend.documents

    async def test_clear_empties_the_store(self):
        backend = self._backend()
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["x"]
        )
        await backend.clear()
        assert backend.id_map == {}


class TestSQLiteVSSBackend:
    def _backend(self, db_path: str):
        sqlite_vss = pytest.importorskip("sqlite_vss")
        from multimind.vector_store.sqlitevss import SQLiteVSSBackend

        return SQLiteVSSBackend(
            db_path=db_path, dim=4, vss_extension_path=sqlite_vss.vss_loadable_path()
        )

    async def test_add_search_delete_round_trip(self, tmp_path):
        backend = self._backend(str(tmp_path / "vectors.db"))
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            metadatas=[{}, {}, {}],
            documents=["a", "b", "c"],
            ids=["x", "y", "z"],
        )
        results = await backend.search([1, 0, 0, 0], k=2)
        assert {r.id for r in results} == {"x", "y"}

        await backend.delete_vectors(["y"])
        results_after = await backend.search([0, 1, 0, 0], k=3)
        assert "y" not in {r.id for r in results_after}
        assert {r.id for r in results_after} == {"x", "z"}

    async def test_search_on_empty_table_returns_empty_not_crash(self, tmp_path):
        """Regression: querying vss0 with zero rows previously aborted the process."""
        backend = self._backend(str(tmp_path / "empty.db"))
        await backend.initialize()
        assert await backend.search([1, 0, 0, 0], k=3) == []

    async def test_clear_then_search_returns_empty(self, tmp_path):
        backend = self._backend(str(tmp_path / "cleared.db"))
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=["a"], ids=["x"]
        )
        await backend.clear()
        assert await backend.search([1, 0, 0, 0], k=3) == []


# --------------------------------------------------------------------------
# Live-service backends: gated on connection env vars, skip cleanly when unset
# --------------------------------------------------------------------------


class TestChromaLive:
    async def test_add_search_round_trip(self):
        _require_env("CHROMA_HOST", "MULTIMIND_TEST_CHROMA")
        pytest.importorskip("chromadb")
        from multimind.vector_store.chroma import ChromaBackend

        backend = ChromaBackend(collection_name="multimind_integration_test")
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]],
            metadatas=[{}],
            documents=[{"content": "a"}],
            ids=["live-x"],
        )
        results = await backend.search([1, 0, 0, 0], k=1)
        assert results
        await backend.delete_vectors(["live-x"])


class TestQdrantLive:
    async def test_add_search_round_trip(self):
        _require_env("QDRANT_HOST", "QDRANT_URL")
        pytest.importorskip("qdrant_client")
        from multimind.vector_store.qdrant import QdrantBackend

        backend = QdrantBackend(collection="multimind_integration_test", dim=4)
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["live-x"]
        )
        results = await backend.search([1, 0, 0, 0], k=1)
        assert results


class TestPineconeLive:
    async def test_add_search_round_trip(self):
        _require_env("PINECONE_API_KEY")
        pytest.importorskip("pinecone")
        from multimind.vector_store.pinecone import PineconeBackend

        backend = PineconeBackend(index_name="multimind-integration-test", dimension=4)
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["live-x"]
        )
        results = await backend.search([1, 0, 0, 0], k=1)
        assert results


class TestWeaviateLive:
    async def test_add_search_round_trip(self):
        _require_env("WEAVIATE_HOST", "WEAVIATE_URL")
        pytest.importorskip("weaviate")
        from multimind.vector_store.weaviate import WeaviateVectorStore

        backend = WeaviateVectorStore(class_name="MultimindIntegrationTest", dim=4)
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["live-x"]
        )
        results = await backend.search([1, 0, 0, 0], k=1)
        assert results


class TestMilvusLive:
    async def test_add_search_round_trip(self):
        _require_env("MILVUS_HOST", "MILVUS_URI")
        pytest.importorskip("pymilvus")
        from multimind.vector_store.milvus import MilvusBackend

        backend = MilvusBackend(collection_name="multimind_integration_test", dim=4)
        await backend.initialize()
        await backend.add_vectors(
            vectors=[[1, 0, 0, 0]], metadatas=[{}], documents=[{"content": "a"}], ids=["live-x"]
        )
        results = await backend.search([1, 0, 0, 0], k=1)
        assert results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
