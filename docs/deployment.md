# Deployment Guide

Containerized deployment of the MultiMind gateway API.

The image is a multi-stage build on `python:3.11-slim`: the builder stage wheels the package, the runtime stage installs only the wheel plus the `gateway` extra (FastAPI, uvicorn). No torch, docs, tests, or examples are included. Final size is roughly 300 MB.

## Build

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
docker compose up -d               # gateway only
docker compose --profile local up -d          # gateway + ollama (local models)
docker compose --profile vector-store up -d   # gateway + chroma
```

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
docker run --rm -it --env-file .env multimind-sdk:latest multimind chat
```

The CLI works torch-free: only the `convert` command needs torch, and it fails with a clear
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
- Restrict CORS in production with `MULTIMIND_ALLOWED_ORIGINS` (comma-separated); the default allows localhost only.
- Chat sessions in compose persist in the `multimind_sessions` volume; drop the volume mount if you want stateless containers.
