"""
Advanced multi-task and cross-model features for PEFT methods.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Se
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging
from enum import Enum
from .adaptive_peft import (
    AdaptiveUniPELTPlusTuner,
    AdaptiveEnhancedMAMTuner,
    UniPELTPlusMethod,
    MethodImportance,
    DynamicComponentWeighting
)

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks supported for multi-task adaptation."""
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_LABELING = "sequence_labeling"
    TEXT_GENERATION = "text_generation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"

class TaskConfig:
    """Configuration for a specific task in multi-task learning."""

    def __init__(
        self,
        task_type: TaskType,
        task_name: str,
        metrics: List[str],
        importance: float = 1.0,
        method_preferences: Optional[Dict[UniPELTPlusMethod, float]] = None,
        data_config: Optional[Dict[str, Any]] = None
    ):
        self.task_type = task_type
        self.task_name = task_name
        self.metrics = metrics
        self.importance = importance
        self.method_preferences = method_preferences or {}
        self.data_config = data_config or {}
        self.performance_history = []

class MultiTaskMethodSelector:
    """Method selection optimized for multi-task learning."""

    def __init__(
        self,
        tasks: List[TaskConfig],
        available_methods: List[UniPELTPlusMethod],
        resource_constraints: Optional[Dict[str, Any]] = None
    ):
        self.tasks = {task.task_name: task for task in tasks}
        self.available_methods = available_methods
        self.resource_constraints = resource_constraints or {
            "max_trainable_params": 1e6,
            "max_memory_gb": 8,
            "max_training_time_hours": 1
        }
        self.task_method_performance = {}
        self.method_task_importance = {}

    def update_task_performance(
        self,
        task_name: str,
        method: UniPELTPlusMethod,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics for a method on a specific task."""
        if task_name not in self.task_method_performance:
            self.task_method_performance[task_name] = {}
        if method not in self.task_method_performance[task_name]:
            self.task_method_performance[task_name][method] = []
        self.task_method_performance[task_name][method].append(metrics)

    def get_method_task_importance(
        self,
        method: UniPELTPlusMethod,
        task_name: str
    ) -> float:
        """Calculate method importance for a specific task."""
        if method not in self.method_task_importance:
            self.method_task_importance[method] = {}

        if task_name not in self.method_task_importance[method]:
            # Calculate based on performance and preferences
            task = self.tasks[task_name]
            if method in task.method_preferences:
                base_importance = task.method_preferences[method]
            else:
                base_importance = 0.5

            if task_name in self.task_method_performance and method in self.task_method_performance[task_name]:
                performance = self.task_method_performance[task_name][method][-1]
                performance_score = sum(performance.values()) / len(performance)
                importance = (base_importance + performance_score) / 2
            else:
                importance = base_importance

            self.method_task_importance[method][task_name] = importance

        return self.method_task_importance[method][task_name]

    def select_methods_for_tasks(
        self,
        model_size: int,
        active_tasks: List[str]
    ) -> Dict[str, List[UniPELTPlusMethod]]:
        """Select optimal methods for each task."""
        task_methods = {}
        total_params = 0
        total_memory = 0
        total_time = 0

        for task_name in active_tasks:
            task = self.tasks[task_name]

            # Calculate method scores for this task
            method_scores = []
            for method in self.available_methods:
                importance = self.get_method_task_importance(method, task_name)
                method_scores.append((method, importance))

            # Sort methods by score
            sorted_methods = [m for m, _ in sorted(method_scores, key=lambda x: x[1], reverse=True)]

            # Select methods within constraints
            selected_methods = []
            for method in sorted_methods:
                usage = self.estimate_resource_usage(method, model_size)

                if (total_params + usage["params"] > self.resource_constraints["max_trainable_params"] or
                    total_memory + usage["memory"] > self.resource_constraints["max_memory_gb"] or
                    total_time + usage["time"] > self.resource_constraints["max_training_time_hours"]):
                    continue

                selected_methods.append(method)
                total_params += usage["params"]
                total_memory += usage["memory"]
                total_time += usage["time"]

            task_methods[task_name] = selected_methods

        return task_methods

    def estimate_resource_usage(self, method: UniPELTPlusMethod, model_size: int) -> Dict[str, float]:
        """Estimate resource usage for a method (reused from AdaptiveMethodSelector)."""
        # Base estimates (can be refined based on empirical data)
        estimates = {
            UniPELTPlusMethod.LORA: {
                "params": model_size * 0.01,
                "memory": model_size * 0.02,
                "time": 0.1
            },
            # ... (other method estimates)
        }
        return estimates.get(method, {
            "params": model_size * 0.01,
            "memory": model_size * 0.02,
            "time": 0.1
        })

class CrossModelTransfer:
    """Cross-model transfer learning for PEFT methods."""

    def __init__(
        self,
        source_model: str,
        target_model: str,
        transfer_config: Optional[Dict[str, Any]] = None
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.transfer_config = transfer_config or {
            "transfer_strategy": "selective",  # or "full"
            "similarity_threshold": 0.8,
            "adaptation_rate": 0.1
        }
        self.method_mappings = {}
        self.performance_history = {}

    def analyze_model_similarity(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Analyze similarity between source and target model components."""
        similarities = {}
        for name, source_param in source_weights.items():
            if name in target_weights:
                target_param = target_weights[name]
                if source_param.shape == target_param.shape:
                    similarity = F.cosine_similarity(
                        source_param.view(1, -1),
                        target_param.view(1, -1)
                    ).item()
                    similarities[name] = similarity
        return similarities

    def transfer_method_weights(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_model: nn.Module,
        method: UniPELTPlusMethod
    ) -> None:
        """Transfer weights from source to target model for a specific method."""
        similarities = self.analyze_model_similarity(
            source_weights,
            {name: param for name, param in target_model.named_parameters()}
        )

        if self.transfer_config["transfer_strategy"] == "selective":
            # Only transfer highly similar components
            for name, similarity in similarities.items():
                if similarity >= self.transfer_config["similarity_threshold"]:
                    source_param = source_weights[name]
                    target_param = target_model.get_parameter(name)
                    with torch.no_grad():
                        target_param.data = (
                            (1 - self.transfer_config["adaptation_rate"]) * target_param.data +
                            self.transfer_config["adaptation_rate"] * source_param.data
                        )
        else:
            # Transfer all compatible components
            for name, source_param in source_weights.items():
                if name in target_model.state_dict():
                    target_param = target_model.get_parameter(name)
                    if source_param.shape == target_param.shape:
                        with torch.no_grad():
                            target_param.data = (
                                (1 - self.transfer_config["adaptation_rate"]) * target_param.data +
                                self.transfer_config["adaptation_rate"] * source_param.data
                            )

class MultiTaskUniPELTPlusTuner(AdaptiveUniPELTPlusTuner):
    """UniPELT++ with multi-task adaptation support."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        tasks: List[TaskConfig],
        available_methods: List[UniPELTPlusMethod],
        model_type: str = "causal_lm",
        method_configs: Optional[Dict[UniPELTPlusMethod, Dict[str, Any]]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        resource_constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize multi-task selector
        self.task_selector = MultiTaskMethodSelector(
            tasks=tasks,
            available_methods=available_methods,
            resource_constraints=resource_constraints
        )

        # Get initial method selection for all tasks
        initial_methods = set()
        for task_name in [task.task_name for task in tasks]:
            task_methods = self.task_selector.select_methods_for_tasks(
                model_size=1e9,  # Estimate based on model name
                active_tasks=[task_name]
            )[task_name]
            initial_methods.update(task_methods)

        super().__init__(
            base_model_name=base_model_name,
            output_dir=output_dir,
            available_methods=list(initial_methods),
            model_type=model_type,
            method_configs=method_configs,
            training_args=training_args,
            model_config=model_config,
            resource_constraints=resource_constraints
        )

        self.tasks = tasks
        self.task_weights = DynamicComponentWeighting(
            num_components=len(tasks),
            initial_weights=[task.importance for task in tasks]
        )

    def train(
        self,
        train_datasets: Dict[str, Union[HFDataset, List[str]]],
        eval_datasets: Optional[Dict[str, Union[HFDataset, List[str]]]] = None,
        **kwargs
    ) -> None:
        """Train with multi-task adaptation."""
        if self.model is None:
            self._prepare_model()

        # Add multi-task callback
        class MultiTaskCallback(TrainerCallback):
            def __init__(self, tuner):
                self.tuner = tuner

            def on_evaluate(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                metrics: Dict[str, float],
                **kwargs
            ):
                # Update task performance
                for task_name, task_metrics in metrics.items():
                    if task_name in self.tuner.tasks:
                        for method in self.tuner.methods:
                            self.tuner.task_selector.update_task_performance(
                                task_name=task_name,
                                method=method,
                                metrics=task_metrics
                            )

                # Update task weights
                task_metrics = [
                    {metric: metrics.get(f"{task.task_name}_{metric}", 0.0)
                     for metric in task.metrics}
                    for task in self.tuner.tasks
                ]
                self.tuner.task_weights.update_weights(task_metrics)

                # Adapt methods for each task
                for task_name in train_datasets.keys():
                    new_methods = self.tuner.task_selector.select_methods_for_tasks(
                        model_size=sum(p.numel() for p in self.tuner.model.parameters()),
                        active_tasks=[task_name]
                    )[task_name]

                    if set(new_methods) != set(self.tuner.methods):
                        logger.info(f"Adapting methods for {task_name}: {[m.value for m in new_methods]}")
                        self.tuner._adapt_methods(new_methods)

        # Add callback to trainer
        if "callbacks" not in self.training_args:
            self.training_args["callbacks"] = []
        self.training_args["callbacks"].append(MultiTaskCallback(self))

        # Train with base class method
        super().train(train_datasets, eval_datasets, **kwargs)

class CrossModelUniPELTPlusTuner(AdaptiveUniPELTPlusTuner):
    """UniPELT++ with cross-model transfer support."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        source_model_path: str,
        available_methods: List[UniPELTPlusMethod],
        model_type: str = "causal_lm",
        method_configs: Optional[Dict[UniPELTPlusMethod, Dict[str, Any]]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        transfer_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize cross-model transfer
        self.transfer = CrossModelTransfer(
            source_model=source_model_path,
            target_model=base_model_name,
            transfer_config=transfer_config
        )

        super().__init__(
            base_model_name=base_model_name,
            output_dir=output_dir,
            available_methods=available_methods,
            model_type=model_type,
            method_configs=method_configs,
            training_args=training_args,
            model_config=model_config
        )

    def _prepare_model(self) -> None:
        """Prepare model with cross-model transfer."""
        super()._prepare_model()

        # Load source model weights
        source_weights = {}
        for method in self.methods:
            method_weights = self._load_method_weights(method)
            if method_weights:
                source_weights[method] = method_weights

        # Transfer weights
        for method, weights in source_weights.items():
            self.transfer.transfer_method_weights(
                source_weights=weights,
                target_model=self.model,
                method=method
            )

    def _load_method_weights(self, method: UniPELTPlusMethod) -> Optional[Dict[str, torch.Tensor]]:
        """Load weights for a specific method from source model."""
        try:
            source_model = PeftModel.from_pretrained(
                self.transfer.source_model,
                method.value
            )
            return {
                name: param.data.clone()
                for name, param in source_model.named_parameters()
                if method.value in name
            }
        except Exception as e:
            logger.warning(f"Failed to load weights for {method.value}: {e}")
            return None