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
import numpy as np

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

class ProviderPerformanceTracker:
    """Tracks provider performance for adaptive routing and weighting."""
    def __init__(self):
        self.metrics = {}
        # metrics: {provider: {"success": int, "fail": int, "latency": [float], "quality": [float], "feedback": [float]}}

    def record(self, provider: str, success: bool, latency: float = None, quality: float = None, feedback: float = None):
        if provider not in self.metrics:
            self.metrics[provider] = {"success": 0, "fail": 0, "latency": [], "quality": [], "feedback": []}
        if success:
            self.metrics[provider]["success"] += 1
        else:
            self.metrics[provider]["fail"] += 1
        if latency is not None:
            self.metrics[provider]["latency"].append(latency)
        if quality is not None:
            self.metrics[provider]["quality"].append(quality)
        if feedback is not None:
            self.metrics[provider]["feedback"].append(feedback)

    def get_score(self, provider: str) -> float:
        m = self.metrics.get(provider, None)
        if not m:
            return 1.0
        total = m["success"] + m["fail"]
        success_rate = m["success"] / total if total > 0 else 1.0
        avg_latency = np.mean(m["latency"]) if m["latency"] else 1.0
        avg_quality = np.mean(m["quality"]) if m["quality"] else 1.0
        avg_feedback = np.mean(m["feedback"]) if m["feedback"] else 1.0
        # Score: success * (1/latency) * quality * feedback
        return success_rate * (1.0 / (avg_latency + 1e-3)) * avg_quality * avg_feedback

    def get_best_provider(self, providers: List[str]) -> str:
        scores = {p: self.get_score(p) for p in providers}
        return max(scores.items(), key=lambda x: x[1])[0]

    def submit_feedback(self, provider: str, feedback: float):
        self.record(provider, success=True, feedback=feedback)

class FallbackPolicy:
    """Centralized fallback policy for routing and provider selection."""
    def __init__(self, strategy: str = "switch_provider", max_retries: int = 1, notify_user: bool = True):
        self.strategy = strategy  # retry, switch_provider, notify_user, raise
        self.max_retries = max_retries
        self.notify_user = notify_user
        self.failure_counts = {}  # {provider: int}

    def record_failure(self, provider: str):
        self.failure_counts[provider] = self.failure_counts.get(provider, 0) + 1

    def should_switch(self, provider: str) -> bool:
        # Switch if failures exceed threshold
        return self.failure_counts.get(provider, 0) >= self.max_retries

    def get_fallback_message(self, provider: str, error: Exception) -> str:
        return f"[Fallback] Switched from provider {provider} due to error: {str(error)}"

class Router:
    """Router for managing provider selection and request routing."""
    
    def __init__(self):
        """Initialize the router."""
        self.providers: Dict[str, ProviderAdapter] = {}
        self.task_configs: Dict[TaskType, TaskConfig] = {}
        self.metrics = MetricsCollector()
        self.performance_tracker = ProviderPerformanceTracker()
        self.fallback_policy = FallbackPolicy()
    
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
        use_adaptive_routing: bool = True,
        **kwargs
    ) -> Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]:
        """Handle routing to a single provider (adaptive if enabled, with fallback policy)."""
        if use_adaptive_routing and len(config.preferred_providers) > 1:
            provider_name = self.performance_tracker.get_best_provider(config.preferred_providers)
        else:
            provider_name = config.preferred_providers[0]
        provider = self.providers[provider_name]
        start = time.time()
        try:
            if task_type == TaskType.TEXT_GENERATION:
                result = await provider.generate_text(input_data, **kwargs)
            elif task_type == TaskType.EMBEDDINGS:
                result = await provider.generate_embeddings(input_data, **kwargs)
            elif task_type == TaskType.IMAGE_ANALYSIS:
                result = await provider.analyze_image(input_data, **kwargs)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            latency = time.time() - start
            quality = getattr(result, 'quality', None) or (result.metadata.get('quality') if hasattr(result, 'metadata') else None)
            feedback = getattr(result, 'feedback', None) or (result.metadata.get('feedback') if hasattr(result, 'metadata') else None)
            self.performance_tracker.record(provider_name, success=True, latency=latency, quality=quality, feedback=feedback)
            return result
        except Exception as e:
            latency = time.time() - start
            self.performance_tracker.record(provider_name, success=False, latency=latency)
            self.fallback_policy.record_failure(provider_name)
            # Centralized fallback logic
            if self.fallback_policy.strategy == "retry" and self.fallback_policy.failure_counts[provider_name] <= self.fallback_policy.max_retries:
                # Retry the same provider
                return await self._handle_single_provider(task_type, input_data, config, use_adaptive_routing, **kwargs)
            elif self.fallback_policy.strategy == "switch_provider" and len(config.preferred_providers) > 1:
                # Switch to next best provider
                remaining = [p for p in config.preferred_providers if p != provider_name]
                if remaining:
                    next_provider = self.performance_tracker.get_best_provider(remaining)
                    if self.fallback_policy.notify_user:
                        print(self.fallback_policy.get_fallback_message(provider_name, e))
                    # Try next provider
                    config_copy = config.copy()
                    config_copy.preferred_providers = remaining
                    return await self._handle_single_provider(task_type, input_data, config_copy, use_adaptive_routing, **kwargs)
            if self.fallback_policy.notify_user:
                print(self.fallback_policy.get_fallback_message(provider_name, e))
            raise
    
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

    def submit_feedback(self, provider: str, feedback: float):
        """Submit user feedback for a provider (1.0=good, 0.0=bad, or any float)."""
        self.performance_tracker.submit_feedback(provider, feedback)