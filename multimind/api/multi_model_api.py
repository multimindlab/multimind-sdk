"""
FastAPI-based API interface for the MultiModelWrapper.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import asyncio
from ..models.factory import ModelFactory
from ..models.multi_model import MultiModelWrapper

app = FastAPI(title="Multi-Model API")

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
async def generate(request: GenerateRequest):
    """Generate text using the multi-model wrapper."""
    try:
        factory = ModelFactory()
        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=request.primary_model,
            fallback_models=request.fallback_models,
            model_weights=request.model_weights
        )
        
        response = await multi_model.generate(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """Generate chat completion using the multi-model wrapper."""
    try:
        factory = ModelFactory()
        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=request.primary_model,
            fallback_models=request.fallback_models,
            model_weights=request.model_weights
        )
        
        response = await multi_model.chat(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
async def embeddings(request: EmbeddingsRequest):
    """Generate embeddings using the multi-model wrapper."""
    try:
        factory = ModelFactory()
        multi_model = MultiModelWrapper(
            model_factory=factory,
            primary_model=request.primary_model,
            fallback_models=request.fallback_models,
            model_weights=request.model_weights
        )
        
        embeddings = await multi_model.embeddings(request.text)
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 