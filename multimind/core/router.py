"""
Router for managing provider selection and request routing.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
from enum import Enum
import asyncio
import time
from .provider import ProviderAdapter, GenerationResult, EmbeddingResult, ImageAnalysisResult
from ..observability.metrics import MetricsCollector

class RoutingStrategy(str, Enum):
    """Routing strategies for provider selection."""
    COST_BASED = "cost_based"
    LATENCY_BASED = "latency_based"
    QUALITY_BASED = "quality_based"
    ENSEMBLE = "ensemble"
    CASCADE = "cascade"

class TaskType(str, Enum):
    """Types of tasks that can be performed."""
    TEXT_GENERATION = "text_generation"
    EMBEDDINGS = "embeddings"
    IMAGE_ANALYSIS = "image_analysis"

class TaskConfig(BaseModel):
    """Configuration for a task."""
    preferred_providers: List[str]
    fallback_providers: List[str]
    routing_strategy: RoutingStrategy
    ensemble_config: Optional[Dict[str, Any]] = None

class Router:
    """Router for managing provider selection and request routing."""
    
    def __init__(self):
        """Initialize the router."""
        self.providers: Dict[str, ProviderAdapter] = {}
        self.task_configs: Dict[TaskType, TaskConfig] = {}
        self.metrics = MetricsCollector()
    
    def register_provider(self, name: str, provider: ProviderAdapter):
        """Register a provider with the router."""
        self.providers[name] = provider
    
    def configure_task(self, task_type: TaskType, config: TaskConfig):
        """Configure a task with the given configuration."""
        self.task_configs[task_type] = config
    
    async def route(
        self,
        task_type: TaskType,
        input_data: Any,
        **kwargs
    ) -> Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]:
        """Route a request to the appropriate provider(s)."""
        if task_type not in self.task_configs:
            raise ValueError(f"No configuration found for task type: {task_type}")
        
        config = self.task_configs[task_type]
        start_time = time.time()
        
        try:
            if config.routing_strategy == RoutingStrategy.ENSEMBLE:
                result = await self._handle_ensemble(task_type, input_data, config, **kwargs)
            elif config.routing_strategy == RoutingStrategy.CASCADE:
                result = await self._handle_cascade(task_type, input_data, config, **kwargs)
            else:
                result = await self._handle_single_provider(task_type, input_data, config, **kwargs)
            
            # Record successful request metrics
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_latency(
                provider=result.provider,
                task_type=task_type,
                model=kwargs.get("model", "unknown"),
                latency_ms=latency_ms,
                metadata={"request_id": kwargs.get("request_id")}
            )
            
            if hasattr(result, "cost"):
                self.metrics.record_cost(
                    provider=result.provider,
                    task_type=task_type,
                    model=kwargs.get("model", "unknown"),
                    cost=result.cost,
                    metadata={"request_id": kwargs.get("request_id")}
                )
            
            if hasattr(result, "tokens"):
                self.metrics.record_tokens(
                    provider=result.provider,
                    task_type=task_type,
                    model=kwargs.get("model", "unknown"),
                    tokens=result.tokens,
                    metadata={"request_id": kwargs.get("request_id")}
                )
            
            return result
            
        except Exception as e:
            # Record error metrics
            self.metrics.record_error(
                provider=kwargs.get("provider", "unknown"),
                task_type=task_type,
                model=kwargs.get("model", "unknown"),
                error_type=type(e).__name__,
                error_message=str(e),
                metadata={"request_id": kwargs.get("request_id")}
            )
            raise
    
    async def _handle_single_provider(
        self,
        task_type: TaskType,
        input_data: Any,
        config: TaskConfig,
        **kwargs
    ) -> Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]:
        """Handle routing to a single provider."""
        provider_name = config.preferred_providers[0]
        provider = self.providers[provider_name]
        
        if task_type == TaskType.TEXT_GENERATION:
            return await provider.generate_text(input_data, **kwargs)
        elif task_type == TaskType.EMBEDDINGS:
            return await provider.generate_embeddings(input_data, **kwargs)
        elif task_type == TaskType.IMAGE_ANALYSIS:
            return await provider.analyze_image(input_data, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _handle_ensemble(
        self,
        task_type: TaskType,
        input_data: Any,
        config: TaskConfig,
        **kwargs
    ) -> Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]:
        """Handle ensemble routing strategy."""
        if not config.ensemble_config:
            raise ValueError("Ensemble configuration is required for ensemble routing")
        
        results = []
        for provider_name in config.preferred_providers:
            provider = self.providers[provider_name]
            try:
                if task_type == TaskType.TEXT_GENERATION:
                    result = await provider.generate_text(input_data, **kwargs)
                elif task_type == TaskType.EMBEDDINGS:
                    result = await provider.generate_embeddings(input_data, **kwargs)
                elif task_type == TaskType.IMAGE_ANALYSIS:
                    result = await provider.analyze_image(input_data, **kwargs)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
                
                results.append(result)
            except Exception as e:
                self.metrics.record_error(
                    provider=provider_name,
                    task_type=task_type,
                    model=kwargs.get("model", "unknown"),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    metadata={"request_id": kwargs.get("request_id")}
                )
        
        if not results:
            raise Exception("All providers failed in ensemble routing")
        
        # Use weighted voting for ensemble results
        if config.ensemble_config["method"] == "weighted_voting":
            weights = config.ensemble_config["weights"]
            weighted_results = []
            for result in results:
                weight = weights.get(result.provider, 1.0)
                weighted_results.append((result, weight))
            
            # For now, just return the result with highest weight
            return max(weighted_results, key=lambda x: x[1])[0]
        else:
            # Default to first successful result
            return results[0]
    
    async def _handle_cascade(
        self,
        task_type: TaskType,
        input_data: Any,
        config: TaskConfig,
        **kwargs
    ) -> Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]:
        """Handle cascade routing strategy."""
        errors = []
        
        # Try preferred providers first
        for provider_name in config.preferred_providers:
            provider = self.providers[provider_name]
            try:
                if task_type == TaskType.TEXT_GENERATION:
                    return await provider.generate_text(input_data, **kwargs)
                elif task_type == TaskType.EMBEDDINGS:
                    return await provider.generate_embeddings(input_data, **kwargs)
                elif task_type == TaskType.IMAGE_ANALYSIS:
                    return await provider.analyze_image(input_data, **kwargs)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
            except Exception as e:
                errors.append((provider_name, e))
        
        # Try fallback providers if all preferred providers fail
        for provider_name in config.fallback_providers:
            provider = self.providers[provider_name]
            try:
                if task_type == TaskType.TEXT_GENERATION:
                    return await provider.generate_text(input_data, **kwargs)
                elif task_type == TaskType.EMBEDDINGS:
                    return await provider.generate_embeddings(input_data, **kwargs)
                elif task_type == TaskType.IMAGE_ANALYSIS:
                    return await provider.analyze_image(input_data, **kwargs)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
            except Exception as e:
                errors.append((provider_name, e))
        
        # If all providers fail, raise an exception with error details
        error_messages = [f"{p}: {str(e)}" for p, e in errors]
        raise Exception(f"All providers failed in cascade routing: {', '.join(error_messages)}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return self.metrics.get_summary()
    
    def save_metrics(self, filepath: Optional[str] = None):
        """Save metrics to a file."""
        self.metrics.save_metrics(filepath)