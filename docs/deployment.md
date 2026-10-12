---
title: "Deployment Guide"
---

Containerized deployment of the MultiMind API.

The image is a multi-stage build on `python:3.11-slim`: the builder stage wheels the package, the runtime stage installs the wheel plus the `gateway` and `rag` extras. The process serves `create_platform_app`: one Swagger UI at `/docs` for chat, sessions, compliance checks, governance, the guard proxy, and RAG. No docs, tests, or examples are included. RAG pulls FAISS, Chroma, and sentence-transformers, so the image carries a CPU-only torch; GPU libraries are left out on purpose.

## Pull the published image

Every merge to `develop` and every release publishes the image to GitHub Container Registry for `linux/amd64` and `linux/arm64` (the `Docker image` workflow in `.github/workflows/docker.yml`). No login is needed.

```bash
docker pull ghcr.io/multimindlab/multimind-sdk:latest
docker run -d --name multimind -p 8000:8000 --env-file .env ghcr.io/multimindlab/multimind-sdk:latest
curl http://127.0.0.1:8000/health
```

| Tag | Built from |
| --- | --- |
| `latest` | the most recent release tag (`v1.0.0` and later) |
| `1.0.0`, `1.0` | that release |
| `develop` | the current `develop` branch, after the smoke test passes |
| `sha-<short>` | one specific commit on `develop` |

The workflow can also be run by hand from the Actions tab (`Docker image`, "Run workflow"); untick "Push" for a build-only run, or add an extra tag such as `rc1`. Pull requests that touch the Dockerfile or the package build the image and run the smoke test without pushing.

`docker compose up` uses the published image; run `docker compose build` to build from the checkout instead.

## Build locally

```bash
docker build -t multimind-sdk:latest .
```

Optional: bake extra dependency groups into the image (see `pyproject.toml` for available extras):

```bash
docker build --build-arg MULTIMIND_EXTRAS=gateway,rag -t multimind-sdk:rag .
```

## Run

```bash
docker run -d --name multimind -p 8000:8000 \
  -e OPENAI_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  multimind-sdk:latest
```

All provider keys are optional; set only the ones you use: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `HUGGINGFACE_API_KEY`. Other settings: `DEFAULT_MODEL` (default `openai`), `OLLAMA_API_BASE`, `OLLAMA_MODEL_NAME`, `LOG_LEVEL`, `MULTIMIND_PORT` (default `8000`).

Or pass an env file instead of individual `-e` flags:

```bash
docker run -d --name multimind -p 8000:8000 --env-file .env multimind-sdk:latest
```

## Docker Compose

The gateway reads a `.env` file at the repo root automatically (optional).

```bash
docker compose up -d               # API on :8000 and dashboard on :8501
docker compose --profile local up -d          # plus ollama (local models)
docker compose --profile vector-store up -d   # plus chroma
```

The API and the dashboard share the `multimind_governance` volume (`/data`): audit log, cost log, and `guardrails.json`. `POST /v1/guardrails/reload` applies that file to the running proxy. Set `API_KEYS` when the API should require an `X-API-Key` header. `/health`, `/ready`, and `/docs` stay open. Compose allows the dashboard origin (`http://localhost:8501`) through `MULTIMIND_CORS_ORIGINS` so the browser can call the API.

### Fully local, no API keys

```bash
DEFAULT_MODEL=ollama docker compose --profile local up -d
docker compose exec ollama ollama pull mistral
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"model": "ollama", "messages": [{"role": "user", "content": "Hello"}]}'
```

The gateway is preconfigured with `OLLAMA_API_BASE=http://ollama:11434`, so the two containers talk over the compose network. Ollama models persist in the `ollama_models` volume.

## Health checks

- `GET /health` — liveness (no auth, no provider keys needed). Used by the image `HEALTHCHECK` and the compose healthcheck.
- `GET /ready` — readiness probe.

```bash
curl http://localhost:8000/health
docker inspect --format '{{.State.Health.Status}}' multimind
```

The healthcheck uses Python's `urllib`, so no curl is required inside the image.

## Running the CLI in the container

```bash
docker run --rm -it --env-file .env multimind-sdk:latest multimind chat start -m openai
```

The CLI works torch-free: only the `convert` command (dispatched before the Click group, so
not listed in `multimind --help`) needs torch, and it fails with a clear
install hint in the lean image. For model conversion inside a container, build with the
finetune extra (large image, several GB):

```bash
docker build --build-arg MULTIMIND_EXTRAS=gateway,finetune -t multimind-sdk:cli .
docker run --rm -it --env-file .env multimind-sdk:cli multimind convert --help
```

## Production notes

- The container runs as non-root user `multimind` (uid 10001).
- Read-only root filesystem is supported. Chat sessions are written under `/app`, so mount it as a tmpfs owned by the app user:

  ```bash
  docker run -d --read-only --tmpfs /app:uid=10001,gid=10001 \
    -p 8000:8000 --env-file .env multimind-sdk:latest
  ```

- Set resource limits, e.g. `docker run --memory 1g --cpus 2 ...`, or in compose:

  ```yaml
  services:
    gateway:
      deploy:
        resources:
          limits:
            memory: 1g
            cpus: "2"
  ```

- Never bake secrets into the image; `.env` is excluded by `.dockerignore`. Inject keys at runtime via env vars or your orchestrator's secret store.
- CORS is off unless `MULTIMIND_CORS_ORIGINS` (comma-separated origins) is set; compose sets it to the dashboard origin only. `MULTIMIND_ALLOWED_ORIGINS` is accepted as a legacy alias.
- Chat sessions in compose persist in the `multimind_sessions` volume; drop the volume mount if you want stateless containers.
