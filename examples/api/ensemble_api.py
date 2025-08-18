"""
FastAPI interface for the MultiMind Ensemble system.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from pathlib import Path

from multimind import Router, TaskType, AdvancedEnsemble, EnsembleMethod

app = FastAPI(
    title="MultiMind Ensemble API",
    description="API for using the MultiMind Ensemble system",
    version="1.0.0"
)

class TextGenerationRequest(BaseModel):
    prompt: str
    providers: List[str] = ["openai", "anthropic", "ollama"]
    method: str = EnsembleMethod.WEIGHTED_VOTING.value
    weights: Optional[Dict[str, float]] = None

class EmbeddingRequest(BaseModel):
    text: str
    providers: List[str] = ["openai", "huggingface"]
    weights: Optional[Dict[str, float]] = None

class CodeReviewRequest(BaseModel):
    code: str
    providers: List[str] = ["openai", "anthropic", "ollama"]

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    """Generate text using ensemble of models."""
    try:
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
        # Get results from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.TEXT_GENERATION,
                request.prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
            )
            for provider in request.providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod(request.method),
            task_type=TaskType.TEXT_GENERATION,
            weights=request.weights
        )
        
        return {
            "result": combined_result.result.result,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings using ensemble of models."""
    try:
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
        # Get embeddings from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.EMBEDDINGS,
                request.text,
                provider=provider,
                model="text-embedding-ada-002" if provider == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
            )
            for provider in request.providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.EMBEDDINGS,
            weights=request.weights or {
                "openai": 0.6,
                "huggingface": 0.4
            }
        )
        
        return {
            "embedding": combined_result.result.result.tolist(),
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/review")
async def review_code(request: CodeReviewRequest):
    """Review code using ensemble of models."""
    try:
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
        # Prepare prompt
        prompt = f"""Please review the following code and provide feedback on:
1. Code quality
2. Potential bugs
3. Security issues
4. Performance improvements
5. Best practices

Code:
{request.code}"""
        
        # Get reviews from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "codellama"
            )
            for provider in request.providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.RANK_BASED,
            task_type=TaskType.TEXT_GENERATION
        )
        
        return {
            "review": combined_result.result.result,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    providers: List[str] = ["openai", "anthropic"]
):
    """Analyze image using ensemble of models."""
    try:
        router = Router()
        ensemble = AdvancedEnsemble(router)
        
        # Read image file
        image_data = await image.read()
        
        # Get analysis from all providers
        results = await asyncio.gather(*[
            router.route(
                TaskType.IMAGE_ANALYSIS,
                image_data,
                provider=provider,
                model="gpt-4-vision-preview" if provider == "openai" else "claude-3-sonnet"
            )
            for provider in providers
        ])
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.IMAGE_ANALYSIS,
            confidence_threshold=0.7
        )
        
        return {
            "analysis": combined_result.result.result,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 