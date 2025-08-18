"""
Routing strategies for model selection based on cost and latency.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
try:
    from ..models.base import BaseLLM
except ImportError:
    # Fallback for when running as standalone
    class BaseLLM:
        pass
import numpy as np
import random

# Optional torch import for advanced strategies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Advanced routing strategies will be disabled.")

class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    async def select_model(
        self,
        models: List[BaseLLM],
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select a model based on the strategy."""
        pass

class CostAwareStrategy(RoutingStrategy):
    """Selects model based on cost per token."""

    async def select_model(
        self,
        models: List[BaseLLM],
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select the model with lowest expected cost."""
        if not models:
            return None

        min_cost = float('inf')
        selected_model = None

        for model in models:
            cost = await model.get_cost(kwargs.get('prompt_tokens', 0), kwargs.get('max_completion_tokens', 0))
            if cost < min_cost:
                min_cost = cost
                selected_model = model

        return selected_model

class LatencyAwareStrategy(RoutingStrategy):
    """Selects model based on latency."""

    async def select_model(
        self,
        models: List[BaseLLM],
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select the model with lowest latency."""
        if not models:
            return None

        min_latency = float('inf')
        selected_model = None

        for model in models:
            latency = await model.get_latency()
            if latency is not None and latency < min_latency:
                min_latency = latency
                selected_model = model

        return selected_model

class HybridStrategy(RoutingStrategy):
    """Combines cost and latency awareness."""

    def __init__(self, cost_weight: float = 0.5, latency_weight: float = 0.5):
        self.cost_weight = cost_weight
        self.latency_weight = latency_weight

    async def select_model(
        self,
        models: List[BaseLLM],
        prompt_tokens: int,
        max_completion_tokens: int,
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select model based on weighted cost and latency."""
        if not models:
            return None

        best_score = float('inf')
        selected_model = None

        for model in models:
            cost = await model.get_cost(prompt_tokens, max_completion_tokens)
            latency = await model.get_latency() or float('inf')

            # Normalize and combine scores
            cost_score = cost * self.cost_weight
            latency_score = latency * self.latency_weight
            total_score = cost_score + latency_score

            if total_score < best_score:
                best_score = total_score
                selected_model = model

        return selected_model

class ParetoFrontStrategy(RoutingStrategy):
    """Selects model(s) on the Pareto front for cost, latency, and optionally quality."""
    def __init__(self, objectives: List[str] = ["cost", "latency"], secondary: str = "cost"):
        self.objectives = objectives
        self.secondary = secondary

    async def select_model(
        self,
        models: List[BaseLLM],
        prompt_tokens: int = 0,
        max_completion_tokens: int = 0,
        **kwargs
    ) -> Optional[BaseLLM]:
        """Select a model on the Pareto front, breaking ties by the secondary metric."""
        if not models:
            return None
        # Gather objective values for each model
        values = []
        for model in models:
            cost = await model.get_cost(prompt_tokens, max_completion_tokens)
            latency = await model.get_latency() or float('inf')
            quality = None
            if hasattr(model, 'get_quality'):
                try:
                    quality = await model.get_quality()
                except Exception:
                    quality = None
            values.append({"model": model, "cost": cost, "latency": latency, "quality": quality})
        # Build array for Pareto computation
        arr = []
        for v in values:
            row = []
            for obj in self.objectives:
                val = v.get(obj, float('inf'))
                # For quality, higher is better; for cost/latency, lower is better
                if obj == "quality" and val is not None:
                    row.append(-val)  # Negate so higher is better
                else:
                    row.append(val)
            arr.append(row)
        arr = np.array(arr)
        # Find Pareto front (non-dominated)
        is_efficient = np.ones(arr.shape[0], dtype=bool)
        for i, c in enumerate(arr):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(arr[is_efficient] < c, axis=1) | (np.arange(arr.shape[0])[is_efficient] == i)
        pareto_indices = np.where(is_efficient)[0]
        pareto_models = [values[i]["model"] for i in pareto_indices]
        # Break ties by secondary metric
        if not pareto_models:
            return None
        if self.secondary in self.objectives:
            best = min(pareto_models, key=lambda m: getattr(values[models.index(m)], self.secondary, float('inf')))
        else:
            best = pareto_models[0]
        return best

class LearningBasedStrategy(RoutingStrategy):
    """
    Learning-based routing strategy using contextual bandits (epsilon-greedy).
    Tracks model rewards and adapts selection policy based on feedback.
    Usage:
        strategy = LearningBasedStrategy(epsilon=0.1)
        # On each selection, call strategy.update_feedback(model_name, reward)
    """
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.model_stats = {}  # model_name -> {'count': int, 'reward': float}
    async def select_model(
        self,
        models: List[BaseLLM],
        **kwargs
    ) -> Optional[BaseLLM]:
        if not models:
            return None
        # Initialize stats for new models
        for model in models:
            name = getattr(model, 'model_name', str(model))
            if name not in self.model_stats:
                self.model_stats[name] = {'count': 0, 'reward': 0.0}
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            selected = random.choice(models)
        else:
            # Select model with highest average reward
            best_score = float('-inf')
            selected = models[0]
            for model in models:
                name = getattr(model, 'model_name', str(model))
                stats = self.model_stats[name]
                avg_reward = stats['reward'] / stats['count'] if stats['count'] > 0 else 0.0
                if avg_reward > best_score:
                    best_score = avg_reward
                    selected = model
        return selected
    def update_feedback(self, model_name: str, reward: float):
        """
        Update feedback for a model after a selection.
        Args:
            model_name: Name of the model selected
            reward: Numeric reward (e.g., 1.0 for success, 0.0 for fail, or any feedback)
        """
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {'count': 0, 'reward': 0.0}
        self.model_stats[model_name]['count'] += 1
        self.model_stats[model_name]['reward'] += reward

class DeepRLRouterStrategy(RoutingStrategy):
    """
    Deep RL-based routing strategy using a simple DQN agent.
    Learns to select models based on state (features) and reward feedback.
    Usage:
        strategy = DeepRLRouterStrategy(model_names, state_dim, epsilon=0.1)
        # On each selection, call strategy.update_feedback(state, action_idx, reward, next_state, done)
    Requirements:
        - torch (PyTorch)
        - state must be a numeric vector (e.g., [latency, cost, ...])
    """
    def __init__(self, model_names, state_dim, epsilon=0.1, gamma=0.95, lr=0.01, hidden_dim=32):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepRLRouterStrategy. Please install torch.")
        
        self.model_names = model_names
        self.n_actions = len(model_names)
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = []  # (state, action, reward, next_state, done)
        self.batch_size = 16
        self.device = torch.device('cpu')
        class QNet(nn.Module):
            def __init__(self, state_dim, n_actions, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, n_actions)
                )
            def forward(self, x):
                return self.net(x)
        self.qnet = QNet(state_dim, self.n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    async def select_model(self, models: List[BaseLLM], state: list = None, **kwargs) -> Optional[BaseLLM]:
        if not models or state is None:
            return random.choice(models) if models else None
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                qvals = self.qnet(state_tensor)
                action = int(torch.argmax(qvals).item())
        return next((m for m in models if getattr(m, 'model_name', str(m)) == self.model_names[action]), models[0])
    
    def update_feedback(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
            qvals = self.qnet(states)
            qvals = qvals.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_qvals = self.qnet(next_states).max(1)[0]
            targets = rewards + self.gamma * next_qvals * (1 - dones)
            loss = self.loss_fn(qvals, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()