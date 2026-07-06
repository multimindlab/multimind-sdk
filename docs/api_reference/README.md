# MultiMind API Reference

Reference documentation for MultiMind's REST APIs. For the Python API, start with the [quickstart](../quickstart.md) and [cookbook](../cookbook.md).

## REST APIs

- [Gateway API](gateway.md) — chat, compare, sessions, metrics, and compliance endpoints
- [RAG API](rag_api.md) — document management, semantic search, and generation endpoints ([step-by-step server setup](rag_api_server_setup.md))
- [Ensemble API](rest_api.md) — text generation, code review, image analysis, and embeddings via model ensembles

## Generated OpenAPI specs

Machine-readable specs generated from the running services live in [docs/api/](../api/):

- [openapi-gateway.json](../api/openapi-gateway.json)
- [openapi-rag.json](../api/openapi-rag.json)
- [openapi-multi-model.json](../api/openapi-multi-model.json)
- [openapi-unified.json](../api/openapi-unified.json)
- [openapi-server.json](../api/openapi-server.json)

Every running server also serves live interactive Swagger docs at `http://localhost:8000/docs` (and the raw spec at `/openapi.json`).

## Python API

- [RAG pipeline guide](../rag.md)
- [Memory systems](../memory.md)
- [CLI reference](../cli.md)
- [Configuration](../configuration.md)
