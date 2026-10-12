---
title: "MultiMind API Reference"
---

Reference documentation for MultiMind's REST APIs. For the Python API, start with the [quickstart](/quickstart) and [cookbook](/cookbook).

All HTTP servers need the `gateway` extra (`pip install "multimind-sdk[gateway]"`); the RAG routes additionally need the `rag` extra.

## REST APIs

- **MultiMind API** (`multimind backend`, default `http://127.0.0.1:8080`, Swagger at `/docs`) — one FastAPI app built by `multimind.server.platform.create_platform_app()` that serves every supported route: the gateway (chat, generate, compare, sessions, metrics, compliance), the governance dashboard `/api/*` routes, the OpenAI-compatible guard proxy (`/v1/chat/completions`, `/v1/embeddings`, `/v1/guardrails/reload`) and, when the `rag` extra is installed, the RAG routes. Set `API_KEYS` to require an `X-API-Key` header. This is the recommended way to run the HTTP API.
- [Gateway API](/api_reference/gateway) — chat, generate, compare, sessions, metrics, and compliance endpoints (`multimind.gateway.api`)
- [RAG API](/api_reference/rag_api) — document management, semantic search, and generation endpoints (`multimind.gateway.rag_api`; [step-by-step server setup](/api_reference/rag_api_server_setup))

The guard proxy on its own is started with `multimind serve` (default port 8400); see [cli.md](/cli#compliance-proxy-server).

## Generated OpenAPI specs

Machine-readable specs generated from the FastAPI apps (`app.openapi()`) live in [docs/api/](/api):

- [openapi-platform.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-platform.json) — the combined MultiMind API (`multimind backend`)
- [openapi-gateway.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-gateway.json) — `multimind.gateway.api`
- [openapi-rag.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-rag.json) — `multimind.gateway.rag_api`
- [openapi-multi-model.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-multi-model.json) — `multimind.api.multi_model_api` (`/generate`, `/chat`, `/embeddings` over `MultiModelWrapper`)
- [openapi-unified.json](https://github.com/multimindlab/multimind-sdk/blob/develop/docs/api/openapi-unified.json) — `multimind.api.unified_api` (`/v1/process` multi-modal workflows)

Every running server also serves live interactive Swagger docs at `/docs` (ReDoc at `/redoc`, raw spec at `/openapi.json`) on whatever port it is bound to (8080 for `multimind backend`, 8000 for the standalone apps).

## Python API

- [RAG pipeline guide](/rag)
- [Memory systems](/memory)
- [CLI reference](/cli)
- [Configuration](/configuration)
