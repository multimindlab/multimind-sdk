"""
Advanced meta-learning features including few-shot learning and transfer learning.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
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
from .meta_learning import (
    MetaLearner,
    MultiTeacherDistillation,
    MetaOptimizedMultiTaskTuner,
    MultiTeacherDistilledTuner
)
from .multitask_peft import (
    MultiTaskUniPELTPlusTuner,
    TaskConfig,
    TaskType,
    UniPELTPlusMethod
)

logger = logging.getLogger(__name__)

class MAMLLearner:
    """Model-Agnostic Meta-Learning (MAML) for few-shot learning."""
    
    def __init__(
        self,
        model: nn.Module,
        maml_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.maml_config = maml_config or {
            "n_way": 5,
            "k_shot": 5,
            "n_query": 5,
            "n_episodes": 100,
            "inner_lr": 0.01,
            "outer_lr": 0.001,
            "adaptation_steps": 5,
            "first_order": False,  # Use first-order approximation
            "meta_batch_size": 4
        }
        
        # Initialize meta-optimizer
        self.meta_optimizer = Adam(
            self.model.parameters(),
            lr=self.maml_config["outer_lr"]
        )
        
    def _clone_model(self) -> nn.Module:
        """Create a clone of the model for inner loop updates."""
        return type(self.model)(**self.model.config.to_dict())
        
    def _inner_loop(
        self,
        task_data: Dict[str, torch.Tensor],
        clone: nn.Module
    ) -> Tuple[nn.Module, float]:
        """Perform inner loop adaptation."""
        # Initialize task-specific optimizer
        task_optimizer = Adam(
            clone.parameters(),
            lr=self.maml_config["inner_lr"]
        )
        
        # Split into support and query sets
        support_data = {
            k: v[:self.maml_config["k_shot"]]
            for k, v in task_data.items()
        }
        query_data = {
            k: v[self.maml_config["k_shot"]:]
            for k, v in task_data.items()
        }
        
        # Adaptation steps
        for _ in range(self.maml_config["adaptation_steps"]):
            # Forward pass on support set
            outputs = clone(**support_data)
            loss = outputs.loss
            
            # Backward pass
            task_optimizer.zero_grad()
            if self.maml_config["first_order"]:
                # First-order approximation
                loss.backward()
            else:
                # Full second-order
                grad = torch.autograd.grad(
                    loss,
                    clone.parameters(),
                    create_graph=True
                )
                for param, g in zip(clone.parameters(), grad):
                    param.grad = g
            task_optimizer.step()
            
        # Evaluate on query set
        with torch.no_grad():
            query_outputs = clone(**query_data)
            query_loss = query_outputs.loss
            
        return clone, query_loss
        
    def meta_train(
        self,
        train_tasks: List[Dict[str, torch.Tensor]],
        val_tasks: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> None:
        """Meta-train using MAML."""
        for episode in range(self.maml_config["n_episodes"]):
            # Sample meta-batch of tasks
            meta_batch = np.random.choice(
                train_tasks,
                size=min(self.maml_config["meta_batch_size"], len(train_tasks)),
                replace=False
            )
            
            meta_loss = 0.0
            for task in meta_batch:
                # Clone model for inner loop
                clone = self._clone_model()
                clone.load_state_dict(self.model.state_dict())
                
                # Inner loop adaptation
                adapted_model, task_loss = self._inner_loop(task, clone)
                meta_loss += task_loss
                
            # Outer loop update
            meta_loss /= len(meta_batch)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            if (episode + 1) % 10 == 0:
                logger.info(f"MAML Episode {episode + 1}/{self.maml_config['n_episodes']}, "
                          f"Meta-loss: {meta_loss.item():.4f}")
                          
    def adapt_to_task(
        self,
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict[str, Any]]:
        """Adapt to new task using MAML."""
        # Clone model
        adapted_model = self._clone_model()
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Inner loop adaptation
        adapted_model, _ = self._inner_loop(
            {**support_data, **query_data},
            adapted_model
        )
        
        # Evaluate on query set
        with torch.no_grad():
            outputs = adapted_model(**query_data)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == query_data["labels"]).float().mean().item()
            
        return accuracy, {
            "accuracy": accuracy,
            "predictions": predictions.cpu().numpy(),
            "adapted_model": adapted_model
        }

class ReptileLearner:
    """Reptile meta-learning for few-shot learning."""
    
    def __init__(
        self,
        model: nn.Module,
        reptile_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.reptile_config = reptile_config or {
            "n_way": 5,
            "k_shot": 5,
            "n_query": 5,
            "n_episodes": 100,
            "inner_lr": 0.01,
            "outer_lr": 0.001,
            "adaptation_steps": 5,
            "meta_batch_size": 4,
            "epsilon": 1.0  # Reptile step size
        }
        
        # Initialize meta-optimizer
        self.meta_optimizer = Adam(
            self.model.parameters(),
            lr=self.reptile_config["outer_lr"]
        )
        
    def _clone_model(self) -> nn.Module:
        """Create a clone of the model for inner loop updates."""
        return type(self.model)(**self.model.config.to_dict())
        
    def _inner_loop(
        self,
        task_data: Dict[str, torch.Tensor],
        clone: nn.Module
    ) -> nn.Module:
        """Perform inner loop adaptation."""
        # Initialize task-specific optimizer
        task_optimizer = Adam(
            clone.parameters(),
            lr=self.reptile_config["inner_lr"]
        )
        
        # Split into support and query sets
        support_data = {
            k: v[:self.reptile_config["k_shot"]]
            for k, v in task_data.items()
        }
        
        # Adaptation steps
        for _ in range(self.reptile_config["adaptation_steps"]):
            # Forward pass on support set
            outputs = clone(**support_data)
            loss = outputs.loss
            
            # Backward pass
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
            
        return clone
        
    def meta_train(
        self,
        train_tasks: List[Dict[str, torch.Tensor]],
        val_tasks: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> None:
        """Meta-train using Reptile."""
        for episode in range(self.reptile_config["n_episodes"]):
            # Sample meta-batch of tasks
            meta_batch = np.random.choice(
                train_tasks,
                size=min(self.reptile_config["meta_batch_size"], len(train_tasks)),
                replace=False
            )
            
            # Initialize accumulated parameter update
            accumulated_update = {
                name: torch.zeros_like(param)
                for name, param in self.model.named_parameters()
            }
            
            for task in meta_batch:
                # Clone model for inner loop
                clone = self._clone_model()
                clone.load_state_dict(self.model.state_dict())
                
                # Inner loop adaptation
                adapted_model = self._inner_loop(task, clone)
                
                # Accumulate parameter updates
                for (name, param), (_, adapted_param) in zip(
                    self.model.named_parameters(),
                    adapted_model.named_parameters()
                ):
                    accumulated_update[name] += adapted_param - param
                    
            # Reptile update
            for name, param in self.model.named_parameters():
                update = accumulated_update[name] / len(meta_batch)
                param.data += self.reptile_config["epsilon"] * update
                
            if (episode + 1) % 10 == 0:
                logger.info(f"Reptile Episode {episode + 1}/{self.reptile_config['n_episodes']}")
                
    def adapt_to_task(
        self,
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict[str, Any]]:
        """Adapt to new task using Reptile."""
        # Clone model
        adapted_model = self._clone_model()
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Inner loop adaptation
        adapted_model = self._inner_loop(
            {**support_data, **query_data},
            adapted_model
        )
        
        # Evaluate on query set
        with torch.no_grad():
            outputs = adapted_model(**query_data)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == query_data["labels"]).float().mean().item()
            
        return accuracy, {
            "accuracy": accuracy,
            "predictions": predictions.cpu().numpy(),
            "adapted_model": adapted_model
        }

class FewShotLearner:
    """Few-shot learning for PEFT methods with multiple strategies."""
    
    def __init__(
        self,
        model: nn.Module,
        few_shot_config: Optional[Dict[str, Any]] = None,
        strategy: str = "prototype"  # or "maml" or "reptile"
    ):
        self.model = model
        self.strategy = strategy
        self.few_shot_config = few_shot_config or {}
        
        # Initialize strategy-specific learner
        if strategy == "prototype":
            self.learner = FewShotLearner(model, few_shot_config)
        elif strategy == "maml":
            self.learner = MAMLLearner(model, few_shot_config)
        elif strategy == "reptile":
            self.learner = ReptileLearner(model, few_shot_config)
        else:
            raise ValueError(f"Unknown few-shot learning strategy: {strategy}")
            
    def meta_train(
        self,
        train_tasks: List[Dict[str, torch.Tensor]],
        val_tasks: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> None:
        """Meta-train using selected strategy."""
        self.learner.meta_train(train_tasks, val_tasks)
        
    def adapt_to_task(
        self,
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor]
    ) -> Tuple[float, Dict[str, Any]]:
        """Adapt to new task using selected strategy."""
        return self.learner.adapt_to_task(support_data, query_data)

class FewShotMetaTuner(MetaOptimizedMultiTaskTuner):
    """Meta-tuner with few-shot learning capabilities."""
    
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
        few_shot_config: Optional[Dict[str, Any]] = None,
        few_shot_strategy: str = "prototype",
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
        
        # Initialize few-shot learner with selected strategy
        self.few_shot_learner = FewShotLearner(
            model=self.model,
            few_shot_config=few_shot_config,
            strategy=few_shot_strategy
        )
        
    def train(
        self,
        train_datasets: Dict[str, Any],
        eval_datasets: Optional[Dict[str, Any]] = None,
        few_shot_tasks: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs
    ) -> None:
        """Train with few-shot learning capabilities."""
        if few_shot_tasks:
            # Meta-train few-shot learner
            self.few_shot_learner.meta_train(
                train_tasks=few_shot_tasks,
                val_tasks=eval_datasets
            )
            
        # Train with base class method
        super().train(train_datasets, eval_datasets, **kwargs)
        
    def adapt_to_new_task(
        self,
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Adapt to new task using few-shot learning."""
        # Extract features
        support_features = self._extract_features(support_data)
        query_features = self._extract_features(query_data)
        
        # Adapt using few-shot learner
        accuracy, metrics = self.few_shot_learner.adapt_to_task(
            support_features=support_features,
            support_labels=support_data["labels"],
            query_features=query_features,
            query_labels=query_data["labels"]
        )
        
        return {
            "accuracy": accuracy,
            "metrics": metrics,
            "adapted_model": self.model
        }
        
    def _extract_features(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from input data."""
        with torch.no_grad():
            outputs = self.model(**data, output_hidden_states=True)
            # Use last hidden state as features
            return outputs.hidden_states[-1].mean(dim=1)

class TransferLearner:
    """Transfer learning for PEFT methods."""
    
    def __init__(
        self,
        model: nn.Module,
        transfer_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.transfer_config = transfer_config or {
            "transfer_strategy": "frozen",  # or "fine_tune"
            "layer_selection": "auto",  # or "manual"
            "similarity_threshold": 0.8,
            "adaptation_lr": 0.001,
            "warmup_steps": 100
        }
        
        # Initialize layer importance scores
        self.layer_importance = {}
        
    def compute_layer_similarity(
        self,
        source_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute similarity between source and target layers."""
        similarities = {}
        
        for layer_name in source_features.keys():
            if layer_name in target_features:
                # Compute cosine similarity
                source_flat = source_features[layer_name].view(1, -1)
                target_flat = target_features[layer_name].view(1, -1)
                
                similarity = F.cosine_similarity(
                    source_flat,
                    target_flat
                ).item()
                
                similarities[layer_name] = similarity
                
        return similarities
        
    def select_transfer_layers(
        self,
        similarities: Dict[str, float]
    ) -> List[str]:
        """Select layers for transfer based on similarity."""
        if self.transfer_config["layer_selection"] == "auto":
            # Select layers above similarity threshold
            return [
                layer
                for layer, sim in similarities.items()
                if sim >= self.transfer_config["similarity_threshold"]
            ]
        else:
            # Use manually specified layers
            return self.transfer_config.get("manual_layers", [])
            
    def adapt_to_target(
        self,
        target_data: Dict[str, torch.Tensor],
        selected_layers: List[str]
    ) -> Dict[str, Any]:
        """Adapt model to target task using transfer learning."""
        # Freeze non-selected layers if using frozen strategy
        if self.transfer_config["transfer_strategy"] == "frozen":
            for name, param in self.model.named_parameters():
                if name not in selected_layers:
                    param.requires_grad = False
                    
        # Initialize optimizer for selected layers
        optimizer = Adam(
            [p for n, p in self.model.named_parameters() if n in selected_layers],
            lr=self.transfer_config["adaptation_lr"]
        )
        
        # Initialize scheduler
        scheduler = LambdaLR(
            optimizer,
            lambda step: min(1.0, step / self.transfer_config["warmup_steps"])
        )
        
        # Training loop
        best_accuracy = 0.0
        best_state = None
        
        for step in range(self.transfer_config.get("max_steps", 1000)):
            # Forward pass
            outputs = self.model(**target_data)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluate
            if (step + 1) % 100 == 0:
                accuracy = self._evaluate_step(target_data)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_state = {
                        name: param.clone()
                        for name, param in self.model.named_parameters()
                        if name in selected_layers
                    }
                    
        # Restore best state
        if best_state:
            for name, param in self.model.named_parameters():
                if name in best_state:
                    param.data.copy_(best_state[name])
                    
        return {
            "best_accuracy": best_accuracy,
            "selected_layers": selected_layers,
            "transfer_strategy": self.transfer_config["transfer_strategy"]
        }
        
    def _evaluate_step(self, eval_data: Dict[str, torch.Tensor]) -> float:
        """Evaluate model on target task."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**eval_data)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == eval_data["labels"]).float().mean().item()
        self.model.train()
        return accuracy

class TransferMetaTuner(MetaOptimizedMultiTaskTuner):
    """Meta-tuner with transfer learning capabilities."""
    
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
        transfer_config: Optional[Dict[str, Any]] = None,
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
        
        # Initialize transfer learner
        self.transfer_learner = TransferLearner(
            model=self.model,
            transfer_config=transfer_config
        )
        
    def train(
        self,
        train_datasets: Dict[str, Any],
        eval_datasets: Optional[Dict[str, Any]] = None,
        source_tasks: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs
    ) -> None:
        """Train with transfer learning capabilities."""
        if source_tasks:
            # Extract features from source tasks
            source_features = self._extract_task_features(source_tasks)
            
            # Compute layer similarities
            target_features = self._extract_task_features(train_datasets)
            similarities = self.transfer_learner.compute_layer_similarity(
                source_features,
                target_features
            )
            
            # Select layers for transfer
            selected_layers = self.transfer_learner.select_transfer_layers(similarities)
            
            # Adapt to target tasks
            transfer_results = self.transfer_learner.adapt_to_target(
                target_data=train_datasets,
                selected_layers=selected_layers
            )
            
            logger.info(f"Transfer learning results: {transfer_results}")
            
        # Train with base class method
        super().train(train_datasets, eval_datasets, **kwargs)
        
    def _extract_task_features(
        self,
        tasks: Union[Dict[str, Any], List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Extract features from tasks."""
        features = {}
        
        if isinstance(tasks, dict):
            # Single task
            with torch.no_grad():
                outputs = self.model(**tasks, output_hidden_states=True)
                for i, hidden_state in enumerate(outputs.hidden_states):
                    features[f"layer_{i}"] = hidden_state.mean(dim=1)
        else:
            # Multiple tasks
            for task in tasks:
                with torch.no_grad():
                    outputs = self.model(**task, output_hidden_states=True)
                    for i, hidden_state in enumerate(outputs.hidden_states):
                        if f"layer_{i}" not in features:
                            features[f"layer_{i}"] = []
                        features[f"layer_{i}"].append(hidden_state.mean(dim=1))
                        
            # Average features across tasks
            for layer in features:
                features[layer] = torch.stack(features[layer]).mean(dim=0)
                
        return features 