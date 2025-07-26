"""
Main router interface for model selection and request routing.
"""

from typing import List, Dict, Any, Optional, Type
from ..models.base import BaseLLM
from .strategy import RoutingStrategy, CostAwareStrategy, LatencyAwareStrategy, HybridStrategy, ParetoFrontStrategy, LearningBasedStrategy
from .fallback import FallbackHandler

class ModelRouter:
    """
    Routes requests to appropriate models with strategy, feedback, and fallback support.

    Features:
    - Supports cost, latency, hybrid, Pareto-front, and learning-based (bandit) routing strategies
    - Allows dynamic switching of routing strategy
    - Integrates user/model feedback to adapt routing (including learning-based)
    - Centralized, user-informative fallback with explanations
    - Provides explain() method for model selection rationale
    Usage:
        router = ModelRouter(strategy=LearningBasedStrategy())
        ...
        # After each request, provide feedback:
        router.update_learning_feedback(model_name, reward)
    """

    def __init__(self, strategy: Optional[RoutingStrategy] = None):
        self.models: Dict[str, BaseLLM] = {}
        self.strategy = strategy or CostAwareStrategy()
        self.fallback = FallbackHandler()
        self.feedback_history: List[Dict[str, Any]] = []
        self.last_explanation: Optional[str] = None

    def register_model(self, name: str, model: BaseLLM) -> None:
        """Register a model with the router."""
        self.models[name] = model

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Set the routing strategy."""
        self.strategy = strategy

    def set_strategy_by_name(self, name: str, **kwargs) -> None:
        """Set routing strategy by name: 'cost', 'latency', 'hybrid', 'pareto', 'learning'."""
        if name == "cost":
            self.strategy = CostAwareStrategy()
        elif name == "latency":
            self.strategy = LatencyAwareStrategy()
        elif name == "hybrid":
            self.strategy = HybridStrategy(**kwargs)
        elif name == "pareto":
            self.strategy = ParetoFrontStrategy(**kwargs)
        elif name == "learning":
            self.strategy = LearningBasedStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {name}")

    def set_fallback_chain(self, model_names: List[str]) -> None:
        """Set the fallback chain for model selection."""
        self.fallback.set_chain(model_names)

    def add_feedback(self, model_name: str, success: bool, feedback: Optional[str] = None) -> None:
        """Add user/model feedback for routing adaptation."""
        self.feedback_history.append({
            "model": model_name,
            "success": success,
            "feedback": feedback
        })

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Aggregate feedback for each model."""
        stats = {}
        for entry in self.feedback_history:
            m = entry["model"]
            stats.setdefault(m, {"success": 0, "fail": 0, "feedback": []})
            if entry["success"]:
                stats[m]["success"] += 1
            else:
                stats[m]["fail"] += 1
            if entry["feedback"]:
                stats[m]["feedback"].append(entry["feedback"])
        return stats

    async def get_model(
        self,
        model_name: Optional[str] = None,
        explain: bool = False,
        **kwargs
    ) -> BaseLLM:
        """Get a model instance based on strategy and fallback. If explain=True, store rationale."""
        if model_name and model_name in self.models:
            self.last_explanation = f"Model '{model_name}' selected by explicit request."
            return self.models[model_name]

        # Use strategy to select model
        selected_model = await self.strategy.select_model(
            list(self.models.values()),
            **kwargs
        )

        if selected_model:
            self.last_explanation = f"Model '{getattr(selected_model, 'model_name', str(selected_model))}' selected by strategy {self.strategy.__class__.__name__}."
            return selected_model

        # Fall back to fallback chain
        fallback_model = await self.fallback.get_model(self.models)
        self.last_explanation = f"Fallback: model '{getattr(fallback_model, 'model_name', str(fallback_model))}' selected from fallback chain."
        return fallback_model

    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        explain: bool = False,
        **kwargs
    ) -> str:
        """Generate text using the appropriate model. If explain=True, returns (result, explanation)."""
        model = await self.get_model(model_name, explain=explain, **kwargs)
        try:
            result = await model.generate(prompt, **kwargs)
            if explain:
                return result, self.last_explanation
            return result
        except Exception as e:
            if await self.fallback.should_retry(e):
                if explain:
                    self.last_explanation = f"Retrying after error: {e}"  # Add to explanation
                return await self.generate(prompt, explain=explain, **kwargs)
            if explain:
                self.last_explanation = f"Error: {e}"
                return None, self.last_explanation
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        explain: bool = False,
        **kwargs
    ) -> str:
        """Generate chat completion using the appropriate model. If explain=True, returns (result, explanation)."""
        model = await self.get_model(model_name, explain=explain, **kwargs)
        try:
            result = await model.chat(messages, **kwargs)
            if explain:
                return result, self.last_explanation
            return result
        except Exception as e:
            if await self.fallback.should_retry(e):
                if explain:
                    self.last_explanation = f"Retrying after error: {e}"
                return await self.chat(messages, explain=explain, **kwargs)
            if explain:
                self.last_explanation = f"Error: {e}"
                return None, self.last_explanation
            raise

    def explain(self) -> Optional[str]:
        """Return the last model selection/fallback explanation."""
        return self.last_explanation

    def update_learning_feedback(self, model_name: str, reward: float):
        """
        Update feedback for learning-based routing (bandit/RL).
        Args:
            model_name: Name of the model selected
            reward: Numeric reward (e.g., 1.0 for success, 0.0 for fail, or any feedback)
        """
        if hasattr(self.strategy, 'update_feedback'):
            self.strategy.update_feedback(model_name, reward)