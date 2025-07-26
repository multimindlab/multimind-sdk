"""
Multi-modal router implementation with cost-aware switching and MCP integration.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from ..models.base import BaseLLM
from .router import ModelRouter
from .strategy import RoutingStrategy
from ..api.mcp.registry import WorkflowRegistry

class ModalityType:
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class MultiModalRequest:
    """Request structure for multi-modal inputs."""
    def __init__(
        self,
        content: Dict[str, Any],
        modalities: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.modalities = modalities
        self.constraints = constraints or {}
        self.timestamp = datetime.now()

class CostTracker:
    """Tracks costs for different models and modalities."""
    def __init__(self):
        self.costs: Dict[str, Dict[str, float]] = {}
        self.usage_history: List[Dict[str, Any]] = []
    
    def record_usage(
        self,
        model_id: str,
        modality: str,
        tokens: int,
        cost: float
    ) -> None:
        """Record model usage and cost."""
        if model_id not in self.costs:
            self.costs[model_id] = {}
        if modality not in self.costs[model_id]:
            self.costs[model_id][modality] = 0.0
        
        self.costs[model_id][modality] += cost
        self.usage_history.append({
            "model_id": model_id,
            "modality": modality,
            "tokens": tokens,
            "cost": cost,
            "timestamp": datetime.now()
        })
    
    def get_cost(self, model_id: str, modality: str) -> float:
        """Get cost for a model and modality."""
        return self.costs.get(model_id, {}).get(modality, 0.0)

class PerformanceMetrics:
    """Tracks performance metrics for models."""
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    def record_metric(
        self,
        model_id: str,
        metric_name: str,
        value: float
    ) -> None:
        """Record a performance metric."""
        if model_id not in self.metrics:
            self.metrics[model_id] = {}
        self.metrics[model_id][metric_name] = value
    
    def get_metric(self, model_id: str, metric_name: str) -> float:
        """Get a performance metric."""
        return self.metrics.get(model_id, {}).get(metric_name, 0.0)

class MultiModalRouter(ModelRouter):
    """Router for handling multi-modal requests with cost-aware switching."""
    
    def __init__(self, strategy: Optional[RoutingStrategy] = None):
        super().__init__(strategy)
        self.modality_registry: Dict[str, Dict[str, BaseLLM]] = {}
        self.cost_tracker = CostTracker()
        self.performance_metrics = PerformanceMetrics()
        self.workflow_registry = WorkflowRegistry()
    
    def register_modality_model(
        self,
        modality: str,
        model_id: str,
        model: BaseLLM
    ) -> None:
        """Register a model for a specific modality."""
        if modality not in self.modality_registry:
            self.modality_registry[modality] = {}
        self.modality_registry[modality][model_id] = model
    
    async def _analyze_modalities(
        self,
        request: MultiModalRequest
    ) -> List[str]:
        """Analyze input to determine required modalities."""
        return request.modalities
    
    async def _get_routing_strategy(
        self,
        modalities: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get cost-aware routing strategy based on modalities and constraints."""
        strategy = {}
        for modality in modalities:
            available_models = self.modality_registry.get(modality, {})
            if not available_models:
                continue
            
            # Calculate scores for each model
            model_scores = {}
            for model_id, model in available_models.items():
                cost = self.cost_tracker.get_cost(model_id, modality)
                performance = self.performance_metrics.get_metric(
                    model_id,
                    "success_rate"
                )
                
                # Combine metrics into a score
                score = (
                    0.7 * (1.0 / (1.0 + cost)) +  # Cost component
                    0.3 * performance  # Performance component
                )
                model_scores[model_id] = score
            
            # Select best model for modality
            if model_scores:
                best_model = max(model_scores.items(), key=lambda x: x[1])[0]
                strategy[modality] = best_model
        
        return strategy
    
    async def _execute_with_switching(
        self,
        request: MultiModalRequest,
        strategy: Dict[str, str]
    ) -> Dict[str, Any]:
        """Execute request with dynamic model switching."""
        results = {}
        
        for modality, model_id in strategy.items():
            model = self.modality_registry[modality][model_id]
            try:
                # Execute model
                result = await model.process(request.content[modality])
                results[modality] = result
                
                # Record metrics
                self.cost_tracker.record_usage(
                    model_id,
                    modality,
                    result.get("tokens", 0),
                    result.get("cost", 0.0)
                )
                
            except Exception as e:
                # Handle failure and switch models if needed
                if await self._should_switch_model(model_id, modality):
                    new_model_id = await self._get_fallback_model(
                        modality,
                        model_id
                    )
                    if new_model_id:
                        # Retry with new model
                        model = self.modality_registry[modality][new_model_id]
                        result = await model.process(request.content[modality])
                        results[modality] = result
        
        return results
    
    async def _should_switch_model(
        self,
        model_id: str,
        modality: str
    ) -> bool:
        """Determine if model should be switched based on performance."""
        success_rate = self.performance_metrics.get_metric(
            model_id,
            "success_rate"
        )
        return success_rate < 0.8  # Switch if success rate below 80%
    
    async def _get_fallback_model(
        self,
        modality: str,
        current_model_id: str
    ) -> Optional[str]:
        """Get fallback model for a modality."""
        available_models = list(self.modality_registry[modality].keys())
        if len(available_models) > 1:
            # Return next model in list
            current_idx = available_models.index(current_model_id)
            next_idx = (current_idx + 1) % len(available_models)
            return available_models[next_idx]
        return None
    
    async def route_request(
        self,
        request: MultiModalRequest
    ) -> Dict[str, Any]:
        """Route a multi-modal request to appropriate models."""
        # 1. Analyze modalities
        modalities = await self._analyze_modalities(request)
        
        # 2. Get routing strategy
        strategy = await self._get_routing_strategy(
            modalities,
            request.constraints
        )
        
        # 3. Execute with dynamic switching
        return await self._execute_with_switching(request, strategy) 