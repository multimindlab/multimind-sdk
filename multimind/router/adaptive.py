"""
Adaptive router implementation with data-driven model selection.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from .router import ModelRouter
from .strategy import RoutingStrategy
from ..models.base import BaseLLM
from ..memory.importance import ImportanceScorer

class AdaptiveRouter(ModelRouter):
    """Router that adapts model selection based on performance data."""

    def __init__(
        self,
        providers: Dict[str, BaseLLM],
        default_strategy: Optional[RoutingStrategy] = None,
        importance_scorer: Optional[ImportanceScorer] = None,
        performance_window: int = 100,
        adaptation_rate: float = 0.1,
        min_samples: int = 10,
        enable_learning: bool = True
    ):
        """Initialize the adaptive router."""
        super().__init__(providers, default_strategy)
        self.importance_scorer = importance_scorer
        self.performance_window = performance_window
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        self.enable_learning = enable_learning
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.task_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize metrics for each model
        for model_id in self.providers:
            self.model_metrics[model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_tokens": 0,
                "total_latency": 0.0,
                "total_cost": 0.0,
                "task_performance": {},
                "error_rates": {},
                "last_updated": datetime.now()
            }

    async def select_model(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Tuple[str, BaseLLM]:
        """Select model based on performance data and task requirements."""
        # Get task-specific metrics
        task_metrics = self.task_metrics.get(task_type, {})
        
        # Calculate model scores
        model_scores = {}
        for model_id, model in self.providers.items():
            score = await self._calculate_model_score(
                model_id,
                task_type,
                input_data,
                task_metrics
            )
            model_scores[model_id] = score
        
        # Select best model
        best_model_id = max(model_scores.items(), key=lambda x: x[1])[0]
        return best_model_id, self.providers[best_model_id]

    async def _calculate_model_score(
        self,
        model_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        task_metrics: Dict[str, Any]
    ) -> float:
        """Calculate score for model selection."""
        model_metrics = self.model_metrics[model_id]
        
        # Base score components
        success_rate = model_metrics["successful_requests"] / max(1, model_metrics["total_requests"])
        avg_latency = model_metrics["total_latency"] / max(1, model_metrics["total_requests"])
        avg_cost = model_metrics["total_cost"] / max(1, model_metrics["total_tokens"])
        
        # Task-specific performance
        task_performance = model_metrics["task_performance"].get(task_type, 0.5)
        
        # Calculate importance if scorer is available
        importance = 1.0
        if self.importance_scorer:
            importance_result = await self.importance_scorer.score(
                str(input_data),
                {"timestamp": datetime.now().isoformat()},
                task_type
            )
            importance = importance_result["score"]
        
        # Combine scores with weights
        score = (
            0.3 * success_rate +
            0.2 * (1.0 / (1.0 + avg_latency)) +  # Normalize latency
            0.2 * (1.0 / (1.0 + avg_cost)) +     # Normalize cost
            0.2 * task_performance +
            0.1 * importance
        )
        
        return score

    async def record_performance(
        self,
        model_id: str,
        task_type: str,
        success: bool,
        tokens: int,
        latency: float,
        cost: float,
        error_type: Optional[str] = None
    ) -> None:
        """Record performance metrics for model adaptation."""
        # Update model metrics
        model_metrics = self.model_metrics[model_id]
        model_metrics["total_requests"] += 1
        if success:
            model_metrics["successful_requests"] += 1
        model_metrics["total_tokens"] += tokens
        model_metrics["total_latency"] += latency
        model_metrics["total_cost"] += cost
        
        # Update error rates
        if error_type:
            if error_type not in model_metrics["error_rates"]:
                model_metrics["error_rates"][error_type] = 0
            model_metrics["error_rates"][error_type] += 1
        
        # Update task performance
        if task_type not in model_metrics["task_performance"]:
            model_metrics["task_performance"][task_type] = 0.5
        model_metrics["task_performance"][task_type] = (
            0.9 * model_metrics["task_performance"][task_type] +
            0.1 * (1.0 if success else 0.0)
        )
        
        # Record performance history
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "task_type": task_type,
            "success": success,
            "tokens": tokens,
            "latency": latency,
            "cost": cost,
            "error_type": error_type
        }
        self.performance_history.append(performance_data)
        
        # Update task metrics
        if task_type not in self.task_metrics:
            self.task_metrics[task_type] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_tokens": 0,
                "total_latency": 0.0,
                "total_cost": 0.0,
                "model_performance": {}
            }
        
        task_metrics = self.task_metrics[task_type]
        task_metrics["total_requests"] += 1
        if success:
            task_metrics["successful_requests"] += 1
        task_metrics["total_tokens"] += tokens
        task_metrics["total_latency"] += latency
        task_metrics["total_cost"] += cost
        
        # Update model performance for task
        if model_id not in task_metrics["model_performance"]:
            task_metrics["model_performance"][model_id] = 0.5
        task_metrics["model_performance"][model_id] = (
            0.9 * task_metrics["model_performance"][model_id] +
            0.1 * (1.0 if success else 0.0)
        )
        
        # Adapt routing strategy if enabled
        if self.enable_learning and len(self.performance_history) >= self.min_samples:
            await self._adapt_routing_strategy()

    async def _adapt_routing_strategy(self) -> None:
        """Adapt routing strategy based on performance data."""
        if not self.default_strategy:
            return
        
        # Get recent performance data
        recent_performance = self.performance_history[-self.performance_window:]
        
        # Calculate performance metrics
        success_rates = {}
        avg_latencies = {}
        avg_costs = {}
        
        for model_id in self.providers:
            model_data = [p for p in recent_performance if p["model_id"] == model_id]
            if not model_data:
                continue
            
            success_rates[model_id] = sum(1 for p in model_data if p["success"]) / len(model_data)
            avg_latencies[model_id] = sum(p["latency"] for p in model_data) / len(model_data)
            avg_costs[model_id] = sum(p["cost"] for p in model_data) / len(model_data)
        
        # Update strategy weights
        if hasattr(self.default_strategy, "weights"):
            weights = self.default_strategy.weights
            
            # Calculate new weights based on performance
            total_requests = sum(len([p for p in recent_performance if p["model_id"] == model_id])
                               for model_id in self.providers)
            
            for model_id in self.providers:
                if model_id in success_rates:
                    model_requests = len([p for p in recent_performance if p["model_id"] == model_id])
                    current_weight = weights.get(model_id, 1.0)
                    
                    # Calculate performance score
                    performance_score = (
                        0.4 * success_rates[model_id] +
                        0.3 * (1.0 / (1.0 + avg_latencies[model_id])) +
                        0.3 * (1.0 / (1.0 + avg_costs[model_id]))
                    )
                    
                    # Update weight
                    new_weight = (
                        (1 - self.adaptation_rate) * current_weight +
                        self.adaptation_rate * performance_score
                    )
                    weights[model_id] = new_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for model_id in weights:
                    weights[model_id] /= total_weight
            
            self.default_strategy.weights = weights

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the router."""
        return {
            "model_metrics": {
                model_id: {
                    "success_rate": metrics["successful_requests"] / max(1, metrics["total_requests"]),
                    "avg_latency": metrics["total_latency"] / max(1, metrics["total_requests"]),
                    "avg_cost": metrics["total_cost"] / max(1, metrics["total_tokens"]),
                    "task_performance": metrics["task_performance"],
                    "error_rates": metrics["error_rates"]
                }
                for model_id, metrics in self.model_metrics.items()
            },
            "task_metrics": {
                task_type: {
                    "success_rate": metrics["successful_requests"] / max(1, metrics["total_requests"]),
                    "avg_latency": metrics["total_latency"] / max(1, metrics["total_requests"]),
                    "avg_cost": metrics["total_cost"] / max(1, metrics["total_tokens"]),
                    "model_performance": metrics["model_performance"]
                }
                for task_type, metrics in self.task_metrics.items()
            },
            "strategy_weights": (
                self.default_strategy.weights
                if self.default_strategy and hasattr(self.default_strategy, "weights")
                else {}
            )
        } 