# syntax=docker/dockerfile:1

# ── Stage 1: build the wheel ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /src

RUN pip install --no-cache-dir --upgrade pip build

# Only what the sdist/wheel needs (see MANIFEST.in / pyproject.toml)
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE ./
COPY multimind/ multimind/

RUN python -m build --wheel --outdir /wheels

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="MultiMind SDK" \
      org.opencontainers.image.description="Compliance-first AI agent framework — gateway API image" \
      org.opencontainers.image.vendor="MultimindLAB" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/multimindlab/multimind-sdk" \
      org.opencontainers.image.documentation="https://github.com/multimindlab/multimind-sdk/blob/develop/docs/deployment.md"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MULTIMIND_PORT=8000

RUN groupadd --gid 10001 multimind \
    && useradd --uid 10001 --gid multimind --create-home --shell /usr/sbin/nologin multimind

# Extras installed into the image. Default is gateway-only (lean, no torch).
# Override for a CLI-capable image: --build-arg MULTIMIND_EXTRAS=gateway,finetune
ARG MULTIMIND_EXTRAS=gateway

# Core deps + selected extras only. No torch, no docs, no tests, no examples.
# numpy: multimind.gateway.api imports multimind.compliance, whose eager
# imports need numpy but nothing else from the heavy [compliance] extra.
RUN --mount=from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir "$(ls /wheels/multimind_sdk-*.whl)[${MULTIMIND_EXTRAS}]" "numpy>=1.21.0"

# Launcher: some versions of the gateway only register GET /health inside
# MultiMindAPI().configure_routes(), so call it when present (no-op otherwise).
RUN printf '%s\n' \
      'import os' \
      'import uvicorn' \
      'from multimind.gateway.api import app' \
      'try:' \
      '    from multimind.gateway.api import MultiMindAPI' \
      '    MultiMindAPI().configure_routes()' \
      'except Exception:' \
      '    pass' \
      'uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MULTIMIND_PORT", "8000")))' \
      > /usr/local/bin/multimind-serve.py

# Writable app dir (chat sessions are stored under the working directory)
WORKDIR /app
RUN chown multimind:multimind /app

USER multimind

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('MULTIMIND_PORT', '8000') + '/health', timeout=4)"

CMD ["python", "/usr/local/bin/multimind-serve.py"]
