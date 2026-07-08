# Vector-store integration tests

`test_vector_stores_live.py` is excluded from the default test run (marked
`@pytest.mark.integration`; the default `pytest` invocation runs the unit
suite only). Run it explicitly with:

```
pytest -m integration tests/integration/
```

## Local backends (no external service)

FAISS, sklearn, Annoy, and SQLiteVSS are exercised with real add/search/delete
round-trips against tiny in-memory vectors. Each is `pytest.importorskip`'d,
so the tests collect cleanly on a minimal install and only run when the
optional dependency is present:

```
pip install faiss-cpu scikit-learn annoy sqlite-vss
```

Notes on known, real limitations hit while writing these tests (not fixed
here because they are either third-party library behavior or out of this
task's scope):

- **faiss / sqlite-vss load order**: `sqlite-vss` vendors its own private
  build of libfaiss inside its `vss0`/`vector0` extensions. If the `faiss`
  PyPI package's libfaiss loads into the process *first*, the two collide
  (a native symbol conflict) and every insert fails with a nonsensical
  `"add_with_ids not implemented for this type of index"` error.
  `tests/conftest.py` pre-loads sqlite-vss's extension pair into a throwaway
  connection before importing `faiss` for its `HAS_FAISS` flag, which fixes
  the load order for the whole test session. If you see that error outside
  this test session, it means something else imported `faiss` first.
- **Annoy on tiny/symmetric datasets**: Annoy is an *approximate* nearest
  neighbor index (random-projection trees). On very small datasets (2-3
  items) of perfectly orthogonal, axis-aligned vectors it can fail to find
  even an exact match — this was observed directly against the raw `annoy`
  library, independent of this SDK's wrapper code. Tests here assert
  bookkeeping correctness (ids added/removed, no stale indices) rather than
  exact search recall in adversarial tiny-N cases; `vss0`'s query syntax for
  the installed sqlite-vss build is `vss_search(vector, ?)` with the `LIMIT`
  inside the vss0 subquery (not `column MATCH ? ... LIMIT ?` at the outer
  query), and it aborts the whole process (an uncaught C++ exception, not a
  catchable Python error) if queried while its table has zero rows — the
  backend now checks the row count first and returns `[]` instead.

## Live-service backends (Chroma, Qdrant, Pinecone, Weaviate, Milvus)

These need a reachable server or API key. Each test reads the backend's
connection env vars and calls `pytest.skip(...)` when they are not set, so
CI can enable a given backend later just by providing credentials — no code
changes required.

| Backend  | Env vars (any one enables the test)         |
|----------|----------------------------------------------|
| Chroma   | `CHROMA_HOST`, `MULTIMIND_TEST_CHROMA`        |
| Qdrant   | `QDRANT_HOST`, `QDRANT_URL`                   |
| Pinecone | `PINECONE_API_KEY`                            |
| Weaviate | `WEAVIATE_HOST`, `WEAVIATE_URL`                |
| Milvus   | `MILVUS_HOST`, `MILVUS_URI`                   |

Example, to run only the Qdrant live test against a local instance:

```
QDRANT_HOST=localhost pytest -m integration tests/integration/test_vector_stores_live.py::TestQdrantLive
```
