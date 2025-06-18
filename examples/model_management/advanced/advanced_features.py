"""
Example demonstrating advanced features of the MultiModelWrapper.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

class AdvancedMultiModelWrapper(MultiModelWrapper):
    """Extended MultiModelWrapper with advanced features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limits = {}  # Track rate limits per model
        self.cost_tracking = {}  # Track costs per model
        self.model_specialties = {}  # Track model specialties
        self.conversation_history = []  # Track conversation history
        self.last_used = {}  # Track last usage time per model

    async def _check_rate_limits(self, model_name: str) -> bool:
        """Check if model is within rate limits."""
        if model_name not in self.rate_limits:
            return True
        limit = self.rate_limits[model_name]
        current_time = datetime.now()
        if current_time - limit['last_reset'] > timedelta(minutes=1):
            limit['count'] = 0
            limit['last_reset'] = current_time
        return limit['count'] < limit['max_requests']

    async def _update_rate_limits(self, model_name: str):
        """Update rate limit tracking."""
        if model_name not in self.rate_limits:
            self.rate_limits[model_name] = {
                'count': 0,
                'max_requests': 60,  # Default: 60 requests per minute
                'last_reset': datetime.now()
            }
        self.rate_limits[model_name]['count'] += 1

    async def _track_cost(self, model_name: str, tokens: int):
        """Track cost based on token usage."""
        if model_name not in self.cost_tracking:
            self.cost_tracking[model_name] = 0
        # Example cost calculation (adjust based on actual pricing)
        cost_per_token = {
            'openai': 0.00002,
            'claude': 0.000015,
            'ollama': 0.0  # Local model, no cost
        }
        self.cost_tracking[model_name] += tokens * cost_per_token.get(model_name, 0)

    async def _analyze_specialties(self, prompt: str) -> Dict[str, float]:
        """Analyze which model is best suited for the prompt."""
        specialties = {
            'openai': {
                'creative': 0.9,
                'code': 0.8,
                'translation': 0.9
            },
            'claude': {
                'technical': 0.9,
                'analysis': 0.9,
                'reasoning': 0.9
            },
            'ollama': {
                'simple': 0.9,
                'local': 0.9,
                'fast': 0.9
            }
        }
        
        # Simple keyword-based analysis
        prompt_lower = prompt.lower()
        scores = {}
        for model, model_specialties in specialties.items():
            score = 0
            for specialty, weight in model_specialties.items():
                if specialty in prompt_lower:
                    score += weight
            scores[model] = score
        return scores

    async def _should_use_caching(self, prompt: str) -> bool:
        """Determine if response should be cached."""
        # Example caching strategy
        return len(prompt) < 100  # Cache short prompts

    async def generate_with_advanced_features(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        track_cost: bool = True,
        **kwargs
    ) -> str:
        """Generate text with advanced features."""
        # Analyze model specialties
        specialty_scores = await self._analyze_specialties(prompt)
        
        # Adjust model weights based on specialties
        adjusted_weights = self.model_weights.copy()
        for model, score in specialty_scores.items():
            if model in adjusted_weights:
                adjusted_weights[model] *= (1 + score)
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        # Select model with highest adjusted weight
        selected_model = max(adjusted_weights.items(), key=lambda x: x[1])[0]
        
        # Check rate limits
        if not await self._check_rate_limits(selected_model):
            # Try fallback models
            for fallback in self.fallback_models:
                if await self._check_rate_limits(fallback):
                    selected_model = fallback
                    break
        
        # Generate response
        start_time = time.time()
        try:
            response = await self.models[selected_model].generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Update metrics
            await self._update_rate_limits(selected_model)
            if track_cost:
                await self._track_cost(selected_model, len(response.split()))
            
            # Update last used time
            self.last_used[selected_model] = datetime.now()
            
            return response
        except Exception as e:
            # Handle errors and try fallback
            for fallback in self.fallback_models:
                if fallback != selected_model:
                    try:
                        response = await self.models[fallback].generate(
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        )
                        await self._update_rate_limits(fallback)
                        if track_cost:
                            await self._track_cost(fallback, len(response.split()))
                        self.last_used[fallback] = datetime.now()
                        return response
                    except Exception:
                        continue
            raise e

    def get_advanced_metrics(self) -> Dict[str, Any]:
        """Get advanced metrics including costs and rate limits."""
        return {
            'costs': self.cost_tracking,
            'rate_limits': {
                model: {
                    'current_count': limit['count'],
                    'max_requests': limit['max_requests'],
                    'time_until_reset': (limit['last_reset'] + timedelta(minutes=1) - datetime.now()).total_seconds()
                }
                for model, limit in self.rate_limits.items()
            },
            'last_used': {
                model: (datetime.now() - last_used).total_seconds()
                for model, last_used in self.last_used.items()
            }
        }

async def run_advanced_features_examples():
    # Initialize the model factory
    factory = ModelFactory()

    # Create an advanced multi-model wrapper
    advanced_model = AdvancedMultiModelWrapper(
        model_factory=factory,
        primary_model="openai",
        fallback_models=["claude", "ollama"],
        model_weights={
            "openai": 0.4,
            "claude": 0.4,
            "ollama": 0.2
        },
        auto_optimize=True,
        performance_window=100
    )

    # Example 1: Advanced model selection with specialties
    print("Example 1: Advanced model selection with specialties")
    prompts = [
        {
            "type": "creative",
            "prompt": "Write a creative story about a robot learning to paint",
            "expected_model": "openai"
        },
        {
            "type": "technical",
            "prompt": "Analyze the time complexity of this algorithm",
            "expected_model": "claude"
        },
        {
            "type": "simple",
            "prompt": "What is the capital of France?",
            "expected_model": "ollama"
        }
    ]

    for prompt in prompts:
        print(f"\nExecuting {prompt['type']} prompt:")
        response = await advanced_model.generate_with_advanced_features(
            prompt=prompt["prompt"],
            temperature=0.7
        )
        print(f"Response: {response}")

    # Example 2: Rate limiting and cost tracking
    print("\nExample 2: Rate limiting and cost tracking")
    print("Running multiple requests to test rate limiting...")
    
    for _ in range(5):
        response = await advanced_model.generate_with_advanced_features(
            prompt="What is machine learning?",
            temperature=0.7,
            track_cost=True
        )
        await asyncio.sleep(0.1)

    # Get advanced metrics
    metrics = advanced_model.get_advanced_metrics()
    print("\nAdvanced Metrics:")
    print("Costs:", metrics['costs'])
    print("Rate Limits:", metrics['rate_limits'])
    print("Last Used:", metrics['last_used'])

    # Example 3: Error handling and fallback
    print("\nExample 3: Error handling and fallback")
    try:
        # Simulate a failure in the primary model
        print("Simulating primary model failure...")
        response = await advanced_model.generate_with_advanced_features(
            prompt="This should trigger fallback",
            temperature=0.7,
            force_error=True  # This would be handled by your error simulation
        )
    except Exception as e:
        print(f"Error handled: {str(e)}")
        print("Fallback should have been triggered automatically")

    # Example 4: Performance optimization
    print("\nExample 4: Performance optimization")
    print("Testing performance with different configurations...")
    
    configurations = [
        {
            "name": "balanced",
            "weights": {"openai": 0.4, "claude": 0.4, "ollama": 0.2},
            "rate_limits": {"max_requests": 60}
        },
        {
            "name": "high_performance",
            "weights": {"openai": 0.6, "claude": 0.3, "ollama": 0.1},
            "rate_limits": {"max_requests": 100}
        }
    ]

    for config in configurations:
        print(f"\nTesting {config['name']} configuration:")
        advanced_model.model_weights = config["weights"]
        advanced_model.rate_limits = {
            model: {
                "count": 0,
                "max_requests": config["rate_limits"]["max_requests"],
                "last_reset": datetime.now()
            }
            for model in advanced_model.models
        }
        
        # Run performance test
        start_time = time.time()
        for _ in range(3):
            await advanced_model.generate_with_advanced_features(
                prompt="What is artificial intelligence?",
                temperature=0.7
            )
            await asyncio.sleep(0.1)
        
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
        # Show metrics
        metrics = advanced_model.get_advanced_metrics()
        print(f"Metrics for {config['name']}:")
        print("Costs:", metrics['costs'])
        print("Rate Limits:", metrics['rate_limits'])

    # Show final metrics
    print("\nFinal Advanced Metrics:")
    final_metrics = advanced_model.get_advanced_metrics()
    print("Total Costs:", final_metrics['costs'])
    print("Rate Limit Status:", final_metrics['rate_limits'])
    print("Model Usage Times:", final_metrics['last_used'])

if __name__ == "__main__":
    asyncio.run(run_advanced_features_examples()) 