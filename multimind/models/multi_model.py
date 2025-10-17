"""
Enhanced Multi-model wrapper with intelligent model selection and routing.
"""

from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
import asyncio
import time
from datetime import datetime
from .base import BaseLLM
from .factory import ModelFactory

class ModelMetrics:
    """Class to track model performance metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.error_rates: List[float] = []
        self.token_usage: List[int] = []
        self.last_used: Optional[datetime] = None
        self.success_count: int = 0
        self.error_count: int = 0

    def update_metrics(self, response_time: float, error: bool = False, tokens: int = 0):
        """Update model metrics."""
        self.response_times.append(response_time)
        self.error_rates.append(1.0 if error else 0.0)
        self.token_usage.append(tokens)
        self.last_used = datetime.now()
        if error:
            self.error_count += 1
        else:
            self.success_count += 1

    def get_performance_score(self) -> float:
        """Calculate performance score based on metrics."""
        if not self.response_times:
            return 0.0
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        avg_error_rate = sum(self.error_rates) / len(self.error_rates)
        success_rate = self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0
        
        # Normalize metrics (lower is better for response time and error rate)
        response_score = 1.0 / (1.0 + avg_response_time)
        error_score = 1.0 - avg_error_rate
        
        # Combine scores with weights
        return (response_score * 0.4 + error_score * 0.3 + success_rate * 0.3)

class MultiModelWrapper(BaseLLM):
    """Enhanced wrapper class for managing multiple AI models with intelligent routing."""

    def __init__(
        self,
        model_factory: ModelFactory,
        primary_model: str = "openai",
        fallback_models: Optional[List[str]] = None,
        model_weights: Optional[Dict[str, float]] = None,
        auto_optimize: bool = True,
        performance_window: int = 100,
        **kwargs
    ):
        """
        Initialize the multi-model wrapper.
        
        Args:
            model_factory: ModelFactory instance for creating model instances
            primary_model: Primary model provider to use
            fallback_models: List of fallback model providers
            model_weights: Dictionary of model weights for routing decisions
            auto_optimize: Whether to automatically optimize model selection
            performance_window: Number of requests to consider for performance metrics
            **kwargs: Additional arguments passed to model initialization
        """
        super().__init__(model_name="multi-model", **kwargs)
        self.model_factory = model_factory
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.model_weights = model_weights or {}
        self.auto_optimize = auto_optimize
        self.performance_window = performance_window
        self.kwargs = kwargs
        
        # Initialize models and metrics
        self.models: Dict[str, BaseLLM] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.task_history: List[Dict[str, Any]] = []
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize all available models and their metrics."""
        # Initialize primary model
        try:
            self.models[self.primary_model] = self.model_factory.get_model(
                self.primary_model,
                **self.kwargs
            )
            self.model_metrics[self.primary_model] = ModelMetrics()
        except Exception as e:
            print(f"Warning: Failed to initialize primary model {self.primary_model}: {e}")

        # Initialize fallback models
        for model in self.fallback_models:
            try:
                self.models[model] = self.model_factory.get_model(
                    model,
                    **self.kwargs
                )
                self.model_metrics[model] = ModelMetrics()
            except Exception as e:
                print(f"Warning: Failed to initialize fallback model {model}: {e}")

    def _analyze_task(self, task_type: str, **kwargs) -> Dict[str, float]:
        """
        Analyze task characteristics to determine optimal model weights.
        
        Args:
            task_type: Type of task (e.g., 'chat', 'completion', 'embedding')
            **kwargs: Additional context for task analysis
            
        Returns:
            Dictionary of model weights based on task analysis
        """
        weights = self.model_weights.copy()
        
        # Adjust weights based on task type
        if task_type == "creative":
            weights["openai"] = weights.get("openai", 0.0) * 1.2
        elif task_type == "technical":
            weights["claude"] = weights.get("claude", 0.0) * 1.2
        elif task_type == "code":
            weights["openai"] = weights.get("openai", 0.0) * 1.1
            weights["claude"] = weights.get("claude", 0.0) * 1.1
        
        # Adjust weights based on performance metrics
        if self.auto_optimize:
            for model, metrics in self.model_metrics.items():
                if model in weights:
                    performance_score = metrics.get_performance_score()
                    weights[model] *= (1.0 + performance_score)
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights

    async def _select_model(self, task_type: str, **kwargs) -> Tuple[str, BaseLLM]:
        """
        Intelligently select the best model for the given task.
        
        Args:
            task_type: Type of task (e.g., 'chat', 'completion', 'embedding')
            **kwargs: Additional context for model selection
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        # If only one model is available, use it
        if len(self.models) == 1:
            model_name = list(self.models.keys())[0]
            return model_name, self.models[model_name]

        # Analyze task and get optimized weights
        weights = self._analyze_task(task_type, **kwargs)
        
        # Select model with highest weight
        if weights:
            available_models = {
                name: model for name, model in self.models.items()
                if name in weights
            }
            if available_models:
                selected_model = max(
                    available_models.items(),
                    key=lambda x: weights[x[0]]
                )
                return selected_model

        # Default to primary model if available
        if self.primary_model in self.models:
            return self.primary_model, self.models[self.primary_model]

        # Fallback to first available model
        model_name = list(self.models.keys())[0]
        return model_name, self.models[model_name]

    async def _execute_with_metrics(
        self,
        model_name: str,
        model: BaseLLM,
        operation: str,
        **kwargs
    ) -> Any:
        """
        Execute model operation with performance tracking.
        
        Args:
            model_name: Name of the model
            model: Model instance
            operation: Operation to execute
            **kwargs: Arguments for the operation
            
        Returns:
            Operation result
        """
        start_time = time.time()
        try:
            if operation == "generate":
                result = await model.generate(**kwargs)
            elif operation == "chat":
                result = await model.chat(**kwargs)
            elif operation == "embeddings":
                result = await model.embeddings(**kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Update metrics
            self.model_metrics[model_name].update_metrics(
                response_time=time.time() - start_time,
                error=False
            )
            return result
        except Exception as e:
            # Update metrics
            self.model_metrics[model_name].update_metrics(
                response_time=time.time() - start_time,
                error=True
            )
            raise e

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using the most appropriate model."""
        model_name, model = await self._select_model("completion", **kwargs)
        try:
            return await self._execute_with_metrics(
                model_name,
                model,
                "generate",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            # Try fallback models if primary fails
            for fallback in self.fallback_models:
                if fallback in self.models and fallback != model_name:
                    try:
                        return await self._execute_with_metrics(
                            fallback,
                            self.models[fallback],
                            "generate",
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        )
                    except Exception:
                        continue
            raise e

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text stream using the most appropriate model."""
        model_name, model = await self._select_model("completion_stream", **kwargs)
        try:
            async for chunk in model.generate_stream(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield chunk
        except Exception as e:
            # Try fallback models if primary fails
            for fallback in self.fallback_models:
                if fallback in self.models and fallback != model_name:
                    try:
                        async for chunk in self.models[fallback].generate_stream(
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        ):
                            yield chunk
                        return
                    except Exception:
                        continue
            raise e

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using the most appropriate model."""
        model_name, model = await self._select_model("chat", **kwargs)
        try:
            return await self._execute_with_metrics(
                model_name,
                model,
                "chat",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            # Try fallback models if primary fails
            for fallback in self.fallback_models:
                if fallback in self.models and fallback != model_name:
                    try:
                        return await self._execute_with_metrics(
                            fallback,
                            self.models[fallback],
                            "chat",
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        )
                    except Exception:
                        continue
            raise e

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion stream using the most appropriate model."""
        model_name, model = await self._select_model("chat_stream", **kwargs)
        try:
            async for chunk in model.chat_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield chunk
        except Exception as e:
            # Try fallback models if primary fails
            for fallback in self.fallback_models:
                if fallback in self.models and fallback != model_name:
                    try:
                        async for chunk in self.models[fallback].chat_stream(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        ):
                            yield chunk
                        return
                    except Exception:
                        continue
            raise e

    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using the most appropriate model."""
        model_name, model = await self._select_model("embeddings", **kwargs)
        try:
            return await self._execute_with_metrics(
                model_name,
                model,
                "embeddings",
                text=text,
                **kwargs
            )
        except Exception as e:
            # Try fallback models if primary fails
            for fallback in self.fallback_models:
                if fallback in self.models and fallback != model_name:
                    try:
                        return await self._execute_with_metrics(
                            fallback,
                            self.models[fallback],
                            "embeddings",
                            text=text,
                            **kwargs
                        )
                    except Exception:
                        continue
            raise e

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models."""
        return {
            model: {
                "performance_score": metrics.get_performance_score(),
                "success_rate": metrics.success_count / (metrics.success_count + metrics.error_count) if (metrics.success_count + metrics.error_count) > 0 else 0,
                "avg_response_time": sum(metrics.response_times) / len(metrics.response_times) if metrics.response_times else 0,
                "error_rate": sum(metrics.error_rates) / len(metrics.error_rates) if metrics.error_rates else 0
            }
            for model, metrics in self.model_metrics.items()
        } 