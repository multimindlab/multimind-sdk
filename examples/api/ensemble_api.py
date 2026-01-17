"""
FastAPI interface for the MultiMind Ensemble system.
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from pathlib import Path

from multimind import Router, TaskType, AdvancedEnsemble, EnsembleMethod, TaskConfig, RoutingStrategy
from multimind.core.provider import ProviderConfig
from multimind.providers.openai import OpenAIProvider
from multimind.providers.claude import ClaudeProvider
from multimind.providers.ollama import OllamaProvider

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="MultiMind Ensemble API",
    description="API for using the MultiMind Ensemble system",
    version="1.0.0"
)

# Provider registry (same as CLI)
PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "openai": {
        "env": ["OPENAI_API_KEY"],
        "adapter": OpenAIProvider,
        "capabilities": {
            TaskType.TEXT_GENERATION,
            TaskType.EMBEDDINGS,
            TaskType.IMAGE_ANALYSIS,
        },
    },
    "anthropic": {
        "env": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "adapter": ClaudeProvider,
        "capabilities": {
            TaskType.TEXT_GENERATION,
            TaskType.IMAGE_ANALYSIS,
        },
    },
    "ollama": {
        "env": [],
        "adapter": OllamaProvider,
        "capabilities": {
            TaskType.TEXT_GENERATION,
            TaskType.EMBEDDINGS,
            TaskType.IMAGE_ANALYSIS,
        },
    },
}

def _get_env_value(env_vars: List[str]) -> Optional[str]:
    """Get environment variable value from a list of possible names."""
    for var in env_vars:
        value = os.getenv(var)
        if value:
            return value
    return None

def _prepare_router(providers: List[str]) -> Tuple[Router, List[str]]:
    """Prepare router with registered providers."""
    router = Router()
    registered: List[str] = []
    
    for name in providers:
        name_lower = name.lower()
        spec = PROVIDER_REGISTRY.get(name_lower)
        if not spec:
            # Skip unsupported providers
            continue
        
        env_vars = spec["env"]
        api_key = _get_env_value(env_vars) if env_vars else None
        
        # For Ollama (no API key needed)
        if name_lower == "ollama":
            ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
            provider_adapter = spec["adapter"](
                ProviderConfig(api_key=None, api_base=ollama_base, timeout=ollama_timeout)
            )
        elif not api_key:
            # Skip providers without API keys
            continue
        else:
            provider_adapter = spec["adapter"](ProviderConfig(api_key=api_key))
        
        router.register_provider(name_lower, provider_adapter)
        registered.append(name_lower)
    
    # Configure default tasks
    if registered:
        for task_type in [TaskType.TEXT_GENERATION, TaskType.EMBEDDINGS, TaskType.IMAGE_ANALYSIS]:
            task_providers = [
                p for p in registered
                if task_type in PROVIDER_REGISTRY.get(p, {}).get("capabilities", set())
            ]
            if task_providers:
                weight = 1.0 / len(task_providers)
                router.configure_task(
                    task_type,
                    TaskConfig(
                        preferred_providers=task_providers,
                        fallback_providers=[],
                        routing_strategy=RoutingStrategy.ENSEMBLE,
                        ensemble_config={
                            "method": "weighted_voting",
                            "weights": {p: weight for p in task_providers},
                        },
                    ),
                )
    
    router.fallback_policy.notify_user = False
    return router, registered

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
        router, registered_providers = _prepare_router(request.providers)
        if not registered_providers:
            raise HTTPException(
                status_code=400,
                detail="No providers could be registered. Check API keys and provider names."
            )
        
        ensemble = AdvancedEnsemble(router)
        
        # Filter to only registered providers that support text generation
        text_providers = [
            p for p in registered_providers
            if TaskType.TEXT_GENERATION in PROVIDER_REGISTRY.get(p, {}).get("capabilities", set())
        ]
        
        if not text_providers:
            raise HTTPException(
                status_code=400,
                detail="No registered providers support text generation."
            )
        
        # Get results from all providers with error handling and timeout
        async def get_result(provider: str):
            try:
                model = "gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
                # Use asyncio.wait_for to add timeout per provider
                return await asyncio.wait_for(
                    router.route(
                        TaskType.TEXT_GENERATION,
                        request.prompt,
                        provider=provider,
                        model=model
                    ),
                    timeout=60.0  # 60 second timeout per provider
                )
            except asyncio.TimeoutError:
                # Provider timed out, skip it
                return None
            except Exception as e:
                # Return None for failed providers, we'll filter them out
                return None
        
        results = await asyncio.gather(*[get_result(provider) for provider in text_providers], return_exceptions=True)
        # Filter out None results and exceptions
        results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="All providers failed to generate text."
            )
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod(request.method),
            task_type=TaskType.TEXT_GENERATION,
            weights=request.weights
        )
        
        # Access text from GenerationResult (it has 'text' attribute)
        result_obj = combined_result.result
        if hasattr(result_obj, 'text'):
            result_text = result_obj.text
        else:
            result_text = str(result_obj)
        
        return {
            "result": result_text,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings using ensemble of models."""
    try:
        router, registered_providers = _prepare_router(request.providers)
        if not registered_providers:
            raise HTTPException(
                status_code=400,
                detail="No providers could be registered. Check API keys and provider names."
            )
        
        ensemble = AdvancedEnsemble(router)
        
        # Filter to only registered providers that support embeddings
        embedding_providers = [
            p for p in registered_providers
            if TaskType.EMBEDDINGS in PROVIDER_REGISTRY.get(p, {}).get("capabilities", set())
        ]
        
        if not embedding_providers:
            raise HTTPException(
                status_code=400,
                detail="No registered providers support embeddings."
            )
        
        # Get embeddings from all providers with error handling and timeout
        async def get_result(provider: str):
            try:
                model = "text-embedding-ada-002" if provider == "openai" else "mistral"
                # Use asyncio.wait_for to add timeout per provider
                return await asyncio.wait_for(
                    router.route(
                        TaskType.EMBEDDINGS,
                        request.text,
                        provider=provider,
                        model=model
                    ),
                    timeout=60.0  # 60 second timeout per provider
                )
            except asyncio.TimeoutError:
                # Provider timed out, skip it
                return None
            except Exception as e:
                return None
        
        results = await asyncio.gather(*[get_result(provider) for provider in embedding_providers], return_exceptions=True)
        # Filter out None results and exceptions
        results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="All providers failed to generate embeddings."
            )
        
        # Use equal weights if not provided
        if not request.weights:
            weight = 1.0 / len(results)
            weights = {getattr(r, 'provider_name', 'unknown'): weight for r in results}
        else:
            weights = request.weights
        
        # Combine results
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.EMBEDDINGS,
            weights=weights
        )
        
        # Access embedding from EmbeddingResult (it has 'embedding' attribute, not 'result')
        embedding = combined_result.result.embedding
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            embedding = list(embedding) if hasattr(embedding, '__iter__') else [embedding]
        
        return {
            "embedding": embedding,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/review")
async def review_code(request: CodeReviewRequest):
    """Review code using ensemble of models."""
    try:
        router, registered_providers = _prepare_router(request.providers)
        if not registered_providers:
            raise HTTPException(
                status_code=400,
                detail="No providers could be registered. Check API keys and provider names."
            )
        
        ensemble = AdvancedEnsemble(router)
        
        # Filter to only registered providers that support text generation
        text_providers = [
            p for p in registered_providers
            if TaskType.TEXT_GENERATION in PROVIDER_REGISTRY.get(p, {}).get("capabilities", set())
        ]
        
        if not text_providers:
            raise HTTPException(
                status_code=400,
                detail="No registered providers support text generation."
            )
        
        # Prepare prompt
        prompt = f"""Please review the following code and provide feedback on:
1. Code quality
2. Potential bugs
3. Security issues
4. Performance improvements
5. Best practices

Code:
{request.code}"""
        
        # Get reviews from all providers with error handling and timeout
        async def get_result(provider: str):
            try:
                model = "gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "codellama"
                # Use asyncio.wait_for to add timeout per provider
                return await asyncio.wait_for(
                    router.route(
                        TaskType.TEXT_GENERATION,
                        prompt,
                        provider=provider,
                        model=model
                    ),
                    timeout=60.0  # 60 second timeout per provider
                )
            except asyncio.TimeoutError:
                # Provider timed out, skip it
                return None
            except Exception as e:
                return None
        
        results = await asyncio.gather(*[get_result(provider) for provider in text_providers], return_exceptions=True)
        # Filter out None results and exceptions
        results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="All providers failed to review code."
            )
        
        # Combine results
        # Use CONFIDENCE_CASCADE for code review 
        combined_result = await ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.TEXT_GENERATION,
            confidence_threshold=0.7
        )
        
        # Access text from GenerationResult (it has 'text' attribute)
        result_obj = combined_result.result
        if hasattr(result_obj, 'text'):
            review_text = result_obj.text
        else:
            review_text = str(result_obj)
        
        return {
            "review": review_text,
            "confidence": combined_result.confidence.score,
            "explanation": combined_result.confidence.explanation,
            "provider_votes": combined_result.provider_votes
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    providers: List[str] = ["openai", "anthropic"]
):
    """Analyze image using ensemble of models."""
    try:
        router, registered_providers = _prepare_router(providers)
        if not registered_providers:
            raise HTTPException(
                status_code=400,
                detail="No providers could be registered. Check API keys and provider names."
            )
        
        ensemble = AdvancedEnsemble(router)
        
        # Filter to only registered providers that support image analysis
        image_providers = [
            p for p in registered_providers
            if TaskType.IMAGE_ANALYSIS in PROVIDER_REGISTRY.get(p, {}).get("capabilities", set())
        ]
        
        if not image_providers:
            raise HTTPException(
                status_code=400,
                detail="No registered providers support image analysis."
            )
        
        # Read image file
        image_data = await image.read()
        
        # Get analysis from all providers with error handling
        async def get_result(provider: str):
            try:
                model = "gpt-4-vision-preview" if provider == "openai" else "claude-3-sonnet"
                return await router.route(
                    TaskType.IMAGE_ANALYSIS,
                    image_data,
                    provider=provider,
                    model=model
                )
            except Exception as e:
                return None
        
        results = await asyncio.gather(*[get_result(provider) for provider in image_providers])
        results = [r for r in results if r is not None]  # Filter out None results
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="All providers failed to analyze image."
            )
        
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 