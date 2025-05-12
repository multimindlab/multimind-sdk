"""
Advanced optimization features for PEFT methods including task-specific hyperparameter optimization
and cross-task knowledge distillation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set
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
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import optuna
from .multitask_peft import (
    MultiTaskUniPELTPlusTuner,
    TaskConfig,
    TaskType,
    UniPELTPlusMethod
)

logger = logging.getLogger(__name__)

class HyperparameterSpace:
    """Define hyperparameter search space for PEFT methods."""
    
    def __init__(
        self,
        method: UniPELTPlusMethod,
        task_type: TaskType,
        space_config: Optional[Dict[str, Any]] = None
    ):
        self.method = method
        self.task_type = task_type
        self.space_config = space_config or self._get_default_space()
        
    def _get_default_space(self) -> Dict[str, Any]:
        """Get default hyperparameter space based on method and task type."""
        base_space = {
            "learning_rate": (1e-5, 1e-3),
            "weight_decay": (0.0, 0.1),
            "warmup_ratio": (0.0, 0.1),
            "gradient_accumulation_steps": (1, 8),
            "max_grad_norm": (0.1, 1.0)
        }
        
        method_spaces = {
            UniPELTPlusMethod.LORA: {
                "r": (4, 32),
                "alpha": (8, 64),
                "dropout": (0.0, 0.2)
            },
            UniPELTPlusMethod.ADAPTER: {
                "adapter_size": (64, 512),
                "adapter_dropout": (0.0, 0.2)
            },
            UniPELTPlusMethod.PROMPT: {
                "prompt_length": (10, 100),
                "prompt_dropout": (0.0, 0.2)
            }
        }
        
        task_spaces = {
            TaskType.TEXT_CLASSIFICATION: {
                "batch_size": (8, 64),
                "label_smoothing": (0.0, 0.1)
            },
            TaskType.SEQUENCE_LABELING: {
                "batch_size": (4, 32),
                "crf_dropout": (0.0, 0.2)
            },
            TaskType.TEXT_GENERATION: {
                "batch_size": (2, 16),
                "beam_size": (1, 8),
                "temperature": (0.5, 1.5)
            }
        }
        
        space = {**base_space}
        if self.method in method_spaces:
            space.update(method_spaces[self.method])
        if self.task_type in task_spaces:
            space.update(task_spaces[self.task_type])
            
        return space

class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        n_trials: int = 20,
        n_initial_points: int = 5
    ):
        self.space = hyperparameter_space
        self.n_trials = n_trials
        self.n_initial_points = n_initial_points
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=10
        )
        self.X = []  # Hyperparameter configurations
        self.y = []  # Performance scores
        
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest next hyperparameter configuration using Bayesian optimization."""
        if len(self.X) < self.n_initial_points:
            # Random sampling for initial points
            return self._random_sample()
            
        # Convert observations to numpy arrays
        X = np.array(self.X)
        y = np.array(self.y)
        
        # Fit GP model
        self.gp.fit(X, y)
        
        # Generate candidate points
        candidates = self._generate_candidates()
        
        # Predict mean and uncertainty
        y_pred, y_std = self.gp.predict(candidates, return_std=True)
        
        # Select point with highest acquisition value (UCB)
        acquisition = y_pred + 2 * y_std
        best_idx = np.argmax(acquisition)
        
        # Convert back to hyperparameter dict
        return self._array_to_params(candidates[best_idx])
        
    def update(self, params: Dict[str, Any], score: float) -> None:
        """Update optimizer with new observation."""
        self.X.append(self._params_to_array(params))
        self.y.append(score)
        
    def _random_sample(self) -> Dict[str, Any]:
        """Generate random hyperparameter configuration."""
        params = {}
        for name, (low, high) in self.space.space_config.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = np.random.randint(low, high + 1)
            else:
                params[name] = np.random.uniform(low, high)
        return params
        
    def _generate_candidates(self, n_candidates: int = 1000) -> np.ndarray:
        """Generate candidate points for optimization."""
        candidates = []
        for _ in range(n_candidates):
            candidates.append(self._params_to_array(self._random_sample()))
        return np.array(candidates)
        
    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert hyperparameter dict to array."""
        return np.array([params[name] for name in self.space.space_config.keys()])
        
    def _array_to_params(self, array: np.ndarray) -> Dict[str, Any]:
        """Convert array to hyperparameter dict."""
        return {
            name: array[i]
            for i, name in enumerate(self.space.space_config.keys())
        }

class KnowledgeDistillation:
    """Cross-task knowledge distillation for PEFT methods."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        distillation_config: Optional[Dict[str, Any]] = None
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_config = distillation_config or {
            "temperature": 2.0,
            "alpha": 0.5,  # Weight for distillation loss
            "distillation_strategy": "soft",  # or "hard"
            "layer_matching": "auto"  # or "manual"
        }
        self.layer_mappings = self._compute_layer_mappings()
        
    def _compute_layer_mappings(self) -> Dict[str, str]:
        """Compute layer mappings between teacher and student."""
        if self.distillation_config["layer_matching"] == "auto":
            return self._auto_layer_matching()
        return self.distillation_config.get("manual_mappings", {})
        
    def _auto_layer_matching(self) -> Dict[str, str]:
        """Automatically match layers between teacher and student."""
        teacher_layers = {
            name: param.shape
            for name, param in self.teacher_model.named_parameters()
            if "lora" in name or "adapter" in name
        }
        student_layers = {
            name: param.shape
            for name, param in self.student_model.named_parameters()
            if "lora" in name or "adapter" in name
        }
        
        mappings = {}
        for t_name, t_shape in teacher_layers.items():
            best_match = None
            best_similarity = -1
            
            for s_name, s_shape in student_layers.items():
                if s_name in mappings.values():
                    continue
                    
                # Compute shape similarity
                if t_shape == s_shape:
                    similarity = 1.0
                else:
                    # Use cosine similarity of flattened shapes
                    t_flat = torch.tensor(t_shape).float()
                    s_flat = torch.tensor(s_shape).float()
                    similarity = F.cosine_similarity(
                        t_flat.view(1, -1),
                        s_flat.view(1, -1)
                    ).item()
                    
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = s_name
                    
            if best_match and best_similarity > 0.8:
                mappings[t_name] = best_match
                
        return mappings
        
    def compute_distillation_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss between teacher and student."""
        if self.distillation_config["distillation_strategy"] == "soft":
            return self._compute_soft_distillation_loss(
                teacher_outputs,
                student_outputs,
                labels
            )
        else:
            return self._compute_hard_distillation_loss(
                teacher_outputs,
                student_outputs,
                labels
            )
            
    def _compute_soft_distillation_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft distillation loss using KL divergence."""
        temperature = self.distillation_config["temperature"]
        alpha = self.distillation_config["alpha"]
        
        # Compute soft targets
        teacher_logits = teacher_outputs["logits"] / temperature
        student_logits = student_outputs["logits"] / temperature
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)
        
        # Task-specific loss
        task_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return alpha * distillation_loss + (1 - alpha) * task_loss
        
    def _compute_hard_distillation_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute hard distillation loss using teacher predictions."""
        alpha = self.distillation_config["alpha"]
        
        # Get teacher predictions
        teacher_preds = torch.argmax(teacher_outputs["logits"], dim=-1)
        
        # Compute losses
        distillation_loss = F.cross_entropy(
            student_outputs["logits"],
            teacher_preds
        )
        task_loss = F.cross_entropy(
            student_outputs["logits"],
            labels
        )
        
        # Combined loss
        return alpha * distillation_loss + (1 - alpha) * task_loss

class OptimizedMultiTaskTuner(MultiTaskUniPELTPlusTuner):
    """Multi-task tuner with task-specific hyperparameter optimization."""
    
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
        optimization_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            base_model_name=base_model_name,
            output_dir=output_dir,
            tasks=tasks,
            available_methods=available_methods,
            model_type=model_type,
            method_configs=method_configs,
            training_args=training_args,
            model_config=model_config,
            resource_constraints=resource_constraints
        )
        
        self.optimization_config = optimization_config or {
            "n_trials": 20,
            "n_initial_points": 5,
            "optimization_metric": "f1"
        }
        
        # Initialize optimizers for each task
        self.task_optimizers = {
            task.task_name: BayesianOptimizer(
                hyperparameter_space=HyperparameterSpace(
                    method=method,
                    task_type=task.task_type
                ),
                n_trials=self.optimization_config["n_trials"],
                n_initial_points=self.optimization_config["n_initial_points"]
            )
            for task in tasks
            for method in available_methods
        }
        
    def train(
        self,
        train_datasets: Dict[str, Union[HFDataset, List[str]]],
        eval_datasets: Optional[Dict[str, Union[HFDataset, List[str]]]] = None,
        **kwargs
    ) -> None:
        """Train with task-specific hyperparameter optimization."""
        if self.model is None:
            self._prepare_model()
            
        # Add optimization callback
        class OptimizationCallback(TrainerCallback):
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
                # Update optimizers with new observations
                for task_name, task_metrics in metrics.items():
                    if task_name in self.tuner.tasks:
                        for method in self.tuner.methods:
                            optimizer = self.tuner.task_optimizers.get(
                                f"{task_name}_{method.value}"
                            )
                            if optimizer:
                                score = task_metrics.get(
                                    self.tuner.optimization_config["optimization_metric"],
                                    0.0
                                )
                                optimizer.update(
                                    params=self.tuner.method_configs[method],
                                    score=score
                                )
                                
                                # Get new hyperparameters
                                new_params = optimizer.suggest_hyperparameters()
                                self.tuner.method_configs[method].update(new_params)
                                
                                logger.info(
                                    f"Updated hyperparameters for {task_name} "
                                    f"using {method.value}: {new_params}"
                                )
                                
        # Add callback to trainer
        if "callbacks" not in self.training_args:
            self.training_args["callbacks"] = []
        self.training_args["callbacks"].append(OptimizationCallback(self))
        
        # Train with base class method
        super().train(train_datasets, eval_datasets, **kwargs)

class DistilledMultiTaskTuner(MultiTaskUniPELTPlusTuner):
    """Multi-task tuner with cross-task knowledge distillation."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        tasks: List[TaskConfig],
        available_methods: List[UniPELTPlusMethod],
        teacher_model_path: str,
        model_type: str = "causal_lm",
        method_configs: Optional[Dict[UniPELTPlusMethod, Dict[str, Any]]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        distillation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            base_model_name=base_model_name,
            output_dir=output_dir,
            tasks=tasks,
            available_methods=available_methods,
            model_type=model_type,
            method_configs=method_configs,
            training_args=training_args,
            model_config=model_config
        )
        
        # Load teacher model
        self.teacher_model = self._load_teacher_model(teacher_model_path)
        
        # Initialize distillation
        self.distillation = KnowledgeDistillation(
            teacher_model=self.teacher_model,
            student_model=self.model,
            distillation_config=distillation_config
        )
        
    def _load_teacher_model(self, model_path: str) -> nn.Module:
        """Load teacher model for distillation."""
        try:
            return PeftModel.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise
            
    def train(
        self,
        train_datasets: Dict[str, Union[HFDataset, List[str]]],
        eval_datasets: Optional[Dict[str, Union[HFDataset, List[str]]]] = None,
        **kwargs
    ) -> None:
        """Train with cross-task knowledge distillation."""
        if self.model is None:
            self._prepare_model()
            
        # Add distillation callback
        class DistillationCallback(TrainerCallback):
            def __init__(self, tuner):
                self.tuner = tuner
                
            def on_step_end(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs
            ):
                # Get teacher outputs
                with torch.no_grad():
                    teacher_outputs = self.tuner.teacher_model(
                        **self.tuner.current_batch
                    )
                    
                # Get student outputs
                student_outputs = self.tuner.model(
                    **self.tuner.current_batch
                )
                
                # Compute distillation loss
                distillation_loss = self.tuner.distillation.compute_distillation_loss(
                    teacher_outputs=teacher_outputs,
                    student_outputs=student_outputs,
                    labels=self.tuner.current_batch["labels"]
                )
                
                # Update model with combined loss
                distillation_loss.backward()
                
        # Add callback to trainer
        if "callbacks" not in self.training_args:
            self.training_args["callbacks"] = []
        self.training_args["callbacks"].append(DistillationCallback(self))
        
        # Train with base class method
        super().train(train_datasets, eval_datasets, **kwargs) 