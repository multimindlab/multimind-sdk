"""
FastAPI-based API interface for the MultiModelWrapper.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import __version__
from ..models.factory import ModelFactory
from ..models.multi_model import MultiModelWrapper

app = FastAPI(
    title="MultiMind Multi-Model API",
    description=(
        "Text generation, chat, and embeddings backed by the MultiModelWrapper, "
        "with automatic fallback across providers (OpenAI, Anthropic, Ollama, ...). "
        "Provider API keys are read from the environment at request time; "
        "optional client authentication uses the `X-API-Key` header when the "
        "`API_KEYS` environment variable is set."
    ),
    version=__version__,
    openapi_tags=[
        {"name": "generation", "description": "Text generation and chat completions"},
        {"name": "embeddings", "description": "Text embeddings"},
        {"name": "system", "description": "Health and readiness probes"},
    ],
)
logger = logging.getLogger(__name__)


def _get_cors_origins() -> List[str]:
    # CORS is off unless MULTIMIND_CORS_ORIGINS is set (comma-separated origins).
    raw = os.getenv("MULTIMIND_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


if _get_cors_origins():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _get_api_keys() -> List[str]:
    # Read at request time so the app can start without any keys configured.
    raw = os.getenv("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> bool:
    """Verify the API key from request header."""
    api_keys = _get_api_keys()
    if not api_keys:
        return True
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if api_key not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


class ErrorResponse(BaseModel):
    detail: str


ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    422: {"description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
    503: {"model": ErrorResponse, "description": "No provider models configured"},
}


# Reuse a single factory across requests to avoid re-loading env / re-allocating caches.
_MODEL_FACTORY = ModelFactory()

# Cache MultiModelWrapper instances by request parameters.
# Note: wrapper init can be expensive because it initializes provider model instances.
_WRAPPER_CACHE: Dict[Tuple[str, Tuple[str, ...], str], MultiModelWrapper] = {}
_WRAPPER_LOCKS: Dict[Tuple[str, Tuple[str, ...], str], asyncio.Lock] = {}


def _weights_key(model_weights: Optional[Dict[str, float]]) -> str:
    # Stable string key for dict weights (used for caching).
    return json.dumps(model_weights or {}, sort_keys=True, default=str)


async def _get_multi_model(
    *,
    primary_model: str,
    fallback_models: List[str],
    model_weights: Optional[Dict[str, float]],
) -> MultiModelWrapper:
    fallback_tuple = tuple(fallback_models or [])
    key = (primary_model, fallback_tuple, _weights_key(model_weights))

    if key in _WRAPPER_CACHE:
        return _WRAPPER_CACHE[key]

    lock = _WRAPPER_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        if key in _WRAPPER_CACHE:
            return _WRAPPER_CACHE[key]

        wrapper = MultiModelWrapper(
            model_factory=_MODEL_FACTORY,
            primary_model=primary_model,
            fallback_models=list(fallback_tuple),
            model_weights=model_weights,
        )
        _WRAPPER_CACHE[key] = wrapper
        return wrapper


def _require_models(multi_model: MultiModelWrapper) -> None:
    # Keys are checked at request time; a wrapper with no models means none configured.
    if not multi_model.models:
        raise HTTPException(
            status_code=503,
            detail=(
                "No provider models are configured. "
                "Set provider API keys (e.g. OPENAI_API_KEY) and retry."
            ),
        )


class GenerateRequest(BaseModel):
    prompt: str
    primary_model: str = "openai"
    fallback_models: List[str] = Field(default_factory=list)
    model_weights: Optional[Dict[str, float]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Write a haiku about the sea.",
                    "primary_model": "openai",
                    "fallback_models": ["anthropic"],
                    "temperature": 0.7,
                    "max_tokens": 128,
                }
            ]
        }
    }


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    primary_model: str = "openai"
    fallback_models: List[str] = Field(default_factory=list)
    model_weights: Optional[Dict[str, float]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [{"role": "user", "content": "Hello, who are you?"}],
                    "primary_model": "openai",
                    "temperature": 0.7,
                }
            ]
        }
    }


class EmbeddingsRequest(BaseModel):
    text: Union[str, List[str]]
    primary_model: str = "openai"
    fallback_models: List[str] = Field(default_factory=list)
    model_weights: Optional[Dict[str, float]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": ["MultiMind SDK", "vector search"], "primary_model": "openai"}]
        }
    }


class GenerateResponse(BaseModel):
    response: Any = Field(..., description="Generated text")


class ChatResponse(BaseModel):
    response: Any = Field(..., description="Assistant reply")


class EmbeddingsResponse(BaseModel):
    embeddings: Any = Field(..., description="Embedding vector(s)")


class HealthResponse(BaseModel):
    status: str
    version: str


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["generation"],
    responses=ERROR_RESPONSES,
)
async def generate(request: GenerateRequest, authenticated: bool = Depends(verify_api_key)):
    """Generate text using the multi-model wrapper."""
    multi_model = await _get_multi_model(
        primary_model=request.primary_model,
        fallback_models=request.fallback_models,
        model_weights=request.model_weights,
    )
    _require_models(multi_model)
    try:
        response = await multi_model.generate(
            prompt=request.prompt, temperature=request.temperature, max_tokens=request.max_tokens
        )
        return {"response": response}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in /generate")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["generation"],
    responses=ERROR_RESPONSES,
)
async def chat(request: ChatRequest, authenticated: bool = Depends(verify_api_key)):
    """Generate chat completion using the multi-model wrapper."""
    multi_model = await _get_multi_model(
        primary_model=request.primary_model,
        fallback_models=request.fallback_models,
        model_weights=request.model_weights,
    )
    _require_models(multi_model)
    try:
        response = await multi_model.chat(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return {"response": response}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/embeddings",
    response_model=EmbeddingsResponse,
    tags=["embeddings"],
    responses=ERROR_RESPONSES,
)
async def embeddings(request: EmbeddingsRequest, authenticated: bool = Depends(verify_api_key)):
    """Generate embeddings using the multi-model wrapper."""
    multi_model = await _get_multi_model(
        primary_model=request.primary_model,
        fallback_models=request.fallback_models,
        model_weights=request.model_weights,
    )
    _require_models(multi_model)
    try:
        embeddings = await multi_model.embeddings(request.text)
        return {"embeddings": embeddings}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in /embeddings")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Health check endpoint (no auth, no provider keys required)."""
    return {"status": "healthy", "version": __version__}


@app.get("/ready", response_model=HealthResponse, tags=["system"])
async def readiness_check():
    """Readiness probe; the app holds no startup state beyond imports."""
    return {"status": "ready", "version": __version__}
