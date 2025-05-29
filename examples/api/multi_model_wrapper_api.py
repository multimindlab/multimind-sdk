from fastapi import FastAPI, HTTPException
from model_wrapper import ModelWrapper
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Query API",
    description="API for querying multiple LLM providers"
)
wrapper = ModelWrapper()

@app.get("/models")
async def list_models() -> List[str]:
    """Get list of available models."""
    return wrapper.available_models()

@app.get("/query")
async def query_model(
    prompt: str,
    model: str,
    ollama_model: Optional[str] = "mistral",
    hf_model_id: Optional[str] = "mistralai/Mistral-7B-v0.1"
):
    """Query a specific model."""
    available = wrapper.available_models()
    if model not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model} is not available. Available models: {', '.join(available)}"
        )
    
    logger.info(f"Querying {model} with prompt: {prompt}")
    
    result = wrapper.query_model(
        model=model,
        prompt=prompt,
        ollama_model=ollama_model,
        hf_model_id=hf_model_id
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result 