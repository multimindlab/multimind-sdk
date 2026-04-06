"""
FastAPI-based API interface for the MultiModelWrapper.
"""

import logging
import os
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import asyncio
import json
from typing import Tuple, Any
from functools import lru_cache
from ..models.factory import ModelFactory
from ..models.multi_model import MultiModelWrapper

app = FastAPI(title="Multi-Model API")
logger = logging.getLogger(__name__)

API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []


def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> bool:
    """Verify the API key from request header."""
    if not API_KEYS:
        return True
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

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

class GenerateRequest(BaseModel):
    prompt: str
    primary_model: str = "openai"
    fallback_models: List[str] = Field(default_factory=list)
    model_weights: Optional[Dict[str, float]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    primary_model: str = "openai"
    fallback_models: List[str] = Field(default_factory=list)
    model_weights: Optional[Dict[str, float]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class EmbeddingsRequest(BaseModel):
    text: Union[str, List[str]]
    primary_model: str = "openai"
    fallback_models: List[str] = Field(default_factory=list)
    model_weights: Optional[Dict[str, float]] = None

@app.post("/generate")
async def generate(request: GenerateRequest, authenticated: bool = Depends(verify_api_key)):
    """Generate text using the multi-model wrapper."""
    try:
        multi_model = await _get_multi_model(
            primary_model=request.primary_model,
            fallback_models=request.fallback_models,
            model_weights=request.model_weights,
        )
        
        response = await multi_model.generate(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"response": response}
    except Exception as e:
        logger.exception("Unhandled error in /generate")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat")
async def chat(request: ChatRequest, authenticated: bool = Depends(verify_api_key)):
    """Generate chat completion using the multi-model wrapper."""
    try:
        multi_model = await _get_multi_model(
            primary_model=request.primary_model,
            fallback_models=request.fallback_models,
            model_weights=request.model_weights,
        )
        
        response = await multi_model.chat(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"response": response}
    except Exception as e:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/embeddings")
async def embeddings(request: EmbeddingsRequest, authenticated: bool = Depends(verify_api_key)):
    """Generate embeddings using the multi-model wrapper."""
    try:
        multi_model = await _get_multi_model(
            primary_model=request.primary_model,
            fallback_models=request.fallback_models,
            model_weights=request.model_weights,
        )
        
        embeddings = await multi_model.embeddings(request.text)
        return {"embeddings": embeddings}
    except Exception as e:
        logger.exception("Unhandled error in /embeddings")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 