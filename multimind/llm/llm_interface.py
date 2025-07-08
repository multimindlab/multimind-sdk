"""
Advanced LLM interface module for managing LLM connections and generation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import time
from datetime import datetime
import logging
from ..models.base import BaseLLM
from ..prompts.advanced_prompting import AdvancedPrompting, PromptType, PromptStrategy

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop_sequences: List[str]
    custom_params: Dict[str, Any]

@dataclass
class GenerationResult:
    """Generation result with metadata."""
    text: str
    metadata: Dict[str, Any]
    usage: Dict[str, int]
    model: str
    latency: float

class ModelType(Enum):
    """Types of language models."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class ErrorHandlingStrategy(Enum):
    """Strategies for handling generation errors."""
    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    RAISE = "raise"

@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling."""
    strategy: str
    max_retries: int
    retry_delay: float
    fallback_model: Optional[str]
    custom_params: Dict[str, Any]

class EnsembleStrategy(Enum):
    MAJORITY = "majority"
    SEMANTIC = "semantic"
    CONFIDENCE = "confidence"
    LLM = "llm"
    CUSTOM = "custom"

class LLMInterface:
    """Advanced LLM interface with multiple model support."""

    def __init__(
        self,
        models: Dict[str, BaseLLM],
        default_model: str,
        error_config: Optional[ErrorHandlingConfig] = None,
        ensemble_strategy: str = "llm",
        custom_ensemble_fn: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize LLM interface.
        
        Args:
            models: Dictionary of model name to LLM instance
            default_model: Name of default model to use
            error_config: Optional error handling configuration
            ensemble_strategy: Ensemble strategy for combining results
            custom_ensemble_fn: Optional custom ensemble function
            **kwargs: Additional parameters
        """
        self.models = models
        self.default_model = default_model
        self.error_config = error_config or self._get_default_error_config()
        self.kwargs = kwargs
        self.ensemble_strategy = EnsembleStrategy(ensemble_strategy)
        self.custom_ensemble_fn = custom_ensemble_fn
        
        # Initialize advanced prompting
        self.prompting = AdvancedPrompting(model=models[default_model])
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency": 0.0
        }

    def _get_default_error_config(self) -> ErrorHandlingConfig:
        """Get default error handling configuration."""
        return ErrorHandlingConfig(
            strategy=ErrorHandlingStrategy.RETRY.value,
            max_retries=3,
            retry_delay=1.0,
            fallback_model=None,
            custom_params={}
        )

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text using specified model.
        
        Args:
            prompt: Input prompt
            config: Optional generation configuration
            model_name: Optional model name
            **kwargs: Additional parameters
            
        Returns:
            Generation result
        """
        # Select model
        model = self.models.get(
            model_name or self.default_model,
            self.models[self.default_model]
        )
        
        # Get generation config
        gen_config = config or self._get_default_config(model)
        
        # Update metrics
        self.metrics["total_requests"] += 1
        
        try:
            # Generate text
            start_time = time.time()
            
            result = await self._generate_with_retry(
                model,
                prompt,
                gen_config,
                **kwargs
            )
            
            latency = time.time() - start_time
            
            # Update metrics
            self.metrics["successful_requests"] += 1
            self.metrics["total_tokens"] += result.get("usage", {}).get("total_tokens", 0)
            self.metrics["total_latency"] += latency
            
            return GenerationResult(
                text=result["text"],
                metadata=result.get("metadata", {}),
                usage=result.get("usage", {}),
                model=model_name or self.default_model,
                latency=latency
            )
            
        except Exception as e:
            # Update metrics
            self.metrics["failed_requests"] += 1
            
            # Handle error
            return await self._handle_generation_error(
                e,
                prompt,
                gen_config,
                model_name,
                **kwargs
            )

    def _get_default_config(self, model: BaseLLM) -> GenerationConfig:
        """Get default generation configuration."""
        return GenerationConfig(
            model_name=model.model_name,
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop_sequences=[],
            custom_params={}
        )

    async def _generate_with_retry(
        self,
        model: BaseLLM,
        prompt: str,
        config: GenerationConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with retry logic."""
        retries = 0
        last_error = None
        
        while retries <= self.error_config.max_retries:
            try:
                # Generate text
                result = await model.generate(
                    prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    stop=config.stop_sequences,
                    **{**config.custom_params, **kwargs}
                )
                
                # Handle both string and object responses
                if isinstance(result, str):
                    return {
                        "text": result,
                        "metadata": {},
                        "usage": {"total_tokens": len(result.split())}
                    }
                else:
                    return {
                        "text": result.text,
                        "metadata": result.metadata,
                        "usage": result.usage
                    }
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.error_config.max_retries:
                    # Wait before retrying
                    await asyncio.sleep(
                        self.error_config.retry_delay * retries
                    )
                    
                    # Log retry
                    self.logger.warning(
                        f"Generation failed, retrying ({retries}/{self.error_config.max_retries}): {str(e)}"
                    )
                else:
                    # Log failure
                    self.logger.error(
                        f"Generation failed after {retries} retries: {str(e)}"
                    )
                    raise last_error

    async def _handle_generation_error(
        self,
        error: Exception,
        prompt: str,
        config: GenerationConfig,
        model_name: Optional[str],
        **kwargs
    ) -> GenerationResult:
        """Handle generation error based on strategy."""
        if self.error_config.strategy == ErrorHandlingStrategy.RAISE.value:
            raise error
        
        elif self.error_config.strategy == ErrorHandlingStrategy.RETRY.value:
            # Retry strategy should have already been handled in _generate_with_retry
            # If we get here, all retries failed, so raise the error
            raise error
        
        elif self.error_config.strategy == ErrorHandlingStrategy.FALLBACK.value:
            if not self.error_config.fallback_model:
                raise ValueError("Fallback model not specified")
            
            # Try fallback model
            try:
                return await self.generate(
                    prompt,
                    config,
                    model_name=self.error_config.fallback_model,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(
                    f"Fallback generation failed: {str(e)}"
                )
                raise e
        
        elif self.error_config.strategy == ErrorHandlingStrategy.IGNORE.value:
            # Return empty result
            return GenerationResult(
                text="",
                metadata={"error": str(error)},
                usage={"total_tokens": 0},
                model=model_name or self.default_model,
                latency=0.0
            )
        
        else:
            raise ValueError(f"Unsupported error handling strategy: {self.error_config.strategy}")

    async def generate_with_router(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text using model router."""
        if not self.models:
            raise ValueError("Models required for model routing")
        
        # Analyze prompt
        model_choice = await self._route_prompt(prompt)
        
        # Generate using chosen model
        return await self.generate(
            prompt,
            config,
            model_name=model_choice,
            **kwargs
        )

    async def _route_prompt(self, prompt: str) -> str:
        """Route prompt to appropriate model."""
        if not self.models:
            return self.default_model
        
        # Analyze prompt
        prompt_analysis = await self.prompting.analyze_prompt(prompt)
        
        # Get model capabilities
        model_capabilities = {
            name: model.get_capabilities()
            for name, model in self.models.items()
        }
        
        # Score each model
        model_scores = {}
        for name, capabilities in model_capabilities.items():
            score = self._score_model_fit(
                capabilities,
                prompt_analysis
            )
            model_scores[name] = score
        
        # Choose best model
        return max(
            model_scores.items(),
            key=lambda x: x[1]
        )[0]

    def _score_model_fit(
        self,
        capabilities: Dict[str, Any],
        prompt_analysis: Dict[str, Any]
    ) -> float:
        """Score how well a model fits the prompt."""
        score = 0.0
        
        # Check task type
        if prompt_analysis["task_type"] in capabilities.get("supported_tasks", []):
            score += 0.3
        
        # Check complexity
        if prompt_analysis["complexity"] <= capabilities.get("max_complexity", 0):
            score += 0.2
        
        # Check domain
        if prompt_analysis["domain"] in capabilities.get("supported_domains", []):
            score += 0.2
        
        # Check language
        if prompt_analysis["language"] in capabilities.get("supported_languages", []):
            score += 0.1
        
        # Check context length
        if prompt_analysis["context_length"] <= capabilities.get("max_context_length", 0):
            score += 0.2
        
        return score

    async def generate_with_ensemble(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        ensemble_strategy: Optional[str] = None,
        custom_ensemble_fn: Optional[Any] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate text using model ensemble with configurable strategy."""
        # Generate with each model
        results = await asyncio.gather(*[
            self.generate(
                prompt,
                config,
                model_name=name,
                **kwargs
            )
            for name in self.models
        ])

        # Determine strategy
        strategy = EnsembleStrategy(ensemble_strategy) if ensemble_strategy else self.ensemble_strategy
        custom_fn = custom_ensemble_fn or self.custom_ensemble_fn

        if strategy == EnsembleStrategy.MAJORITY:
            texts = [r.text for r in results]
            return max(
                results,
                key=lambda x: texts.count(x.text)
            )
        elif strategy == EnsembleStrategy.SEMANTIC:
            return await self._semantic_voting(results)
        elif strategy == EnsembleStrategy.CONFIDENCE:
            return self._confidence_weighted(results)
        elif strategy == EnsembleStrategy.LLM:
            if not self.models:
                raise ValueError("Models required for LLM ensemble strategy")
            combined = await self._combine_ensemble_results(
                prompt,
                results
            )
            return GenerationResult(
                text=combined["text"],
                metadata={
                    "ensemble_results": [
                        {
                            "model": r.model,
                            "text": r.text,
                            "score": r.metadata.get("score", 0.0)
                        }
                        for r in results
                    ],
                    **combined["metadata"]
                },
                usage={
                    "total_tokens": sum(
                        r.usage.get("total_tokens", 0)
                        for r in results
                    )
                },
                model="ensemble",
                latency=sum(r.latency for r in results)
            )
        elif strategy == EnsembleStrategy.CUSTOM and custom_fn:
            return await custom_fn(prompt, results)
        else:
            raise ValueError(f"Unsupported ensemble strategy: {strategy}")

    async def _semantic_voting(self, results: List[GenerationResult]) -> GenerationResult:
        """Select the most semantically central answer using embedding similarity."""
        try:
            from ..embeddings.embedding import get_embedding, cosine_similarity
        except ImportError:
            raise ImportError("Semantic voting requires embedding utilities.")
        texts = [r.text for r in results]
        embeddings = [get_embedding(text) for text in texts]
        # Compute average similarity for each answer
        avg_sims = []
        for i, emb in enumerate(embeddings):
            sims = [cosine_similarity(emb, other) for j, other in enumerate(embeddings) if i != j]
            avg_sims.append(sum(sims) / (len(sims) or 1))
        # Pick the answer with highest average similarity
        idx = avg_sims.index(max(avg_sims))
        return results[idx]

    def _confidence_weighted(self, results: List[GenerationResult]) -> GenerationResult:
        """Weighted voting by model confidence/score."""
        # Assume each result has a 'score' in metadata
        scores = [r.metadata.get("score", 1.0) for r in results]
        if sum(scores) == 0:
            # Fallback to majority
            texts = [r.text for r in results]
            return max(
                results,
                key=lambda x: texts.count(x.text)
            )
        # Weighted selection: pick the answer with highest total score
        score_map = {}
        for r, s in zip(results, scores):
            score_map[r.text] = score_map.get(r.text, 0) + s
        best_text = max(score_map.items(), key=lambda x: x[1])[0]
        for r in results:
            if r.text == best_text:
                return r
        return results[0]  # fallback

    async def _combine_ensemble_results(
        self,
        prompt: str,
        results: List[GenerationResult]
    ) -> Dict[str, Any]:
        """Combine ensemble results using LLM."""
        # Format results
        results_text = "\n\n".join(
            f"Model {i+1} ({r.model}):\n{r.text}"
            for i, r in enumerate(results)
        )
        
        # Generate combination prompt
        combination_prompt = f"""
        Given the following prompt and multiple model responses, generate a combined response that:
        1. Takes the best parts from each response
        2. Resolves any contradictions
        3. Provides a coherent and comprehensive answer
        
        Prompt: {prompt}
        
        Model responses:
        {results_text}
        
        Combined response:
        """
        
        # Generate combined response
        combined_text = await self.models[self.default_model].generate(combination_prompt)
        
        return {
            "text": combined_text,
            "metadata": {
                "combination_method": "llm",
                "source_responses": len(results)
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get generation metrics."""
        return {
            **self.metrics,
            "average_latency": (
                self.metrics["total_latency"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0.0
            ),
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0.0
            )
        }

    def reset_metrics(self) -> None:
        """Reset generation metrics."""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency": 0.0
        } 