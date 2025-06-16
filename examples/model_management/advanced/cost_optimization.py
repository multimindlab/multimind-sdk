"""
Example demonstrating cost optimization in model management.
"""

import asyncio
from typing import Dict, Any, List
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper
from multimind.metrics.cost_tracker import CostTracker

class CostOptimizedWrapper(MultiModelWrapper):
    """Wrapper that optimizes for cost while maintaining quality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_tracker = CostTracker()
        self.quality_threshold = 0.8
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with cost optimization."""
        # Get cost estimates for each model
        cost_estimates = {}
        for model_name in self.available_models:
            model = self.models[model_name]
            cost_estimates[model_name] = await model.estimate_cost(prompt)
        
        # Sort models by cost
        sorted_models = sorted(
            cost_estimates.items(),
            key=lambda x: x[1]
        )
        
        # Try models in order of cost
        for model_name, cost in sorted_models:
            try:
                # Check if we've exceeded budget
                if self.cost_tracker.get_total_cost() + cost > self.budget:
                    continue
                
                # Generate with current model
                response = await self.models[model_name].generate(
                    prompt=prompt,
                    **kwargs
                )
                
                # Track cost
                self.cost_tracker.track_cost(model_name, cost)
                
                # Check quality
                quality_score = await self._evaluate_quality(response)
                if quality_score >= self.quality_threshold:
                    return response
                
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                continue
        
        # If all models fail, use the most expensive one
        return await self.models[sorted_models[-1][0]].generate(
            prompt=prompt,
            **kwargs
        )
    
    async def _evaluate_quality(self, response: str) -> float:
        """Evaluate the quality of a response."""
        # This is a simple example - in practice, you'd use a more sophisticated
        # quality evaluation method
        return 0.9  # Placeholder

async def main():
    """Run the cost optimization example."""
    # Initialize the model factory
    factory = ModelFactory()
    
    # Create a cost-optimized wrapper
    wrapper = CostOptimizedWrapper(
        model_factory=factory,
        primary_model="gpt-3.5-turbo",  # Cheaper model as primary
        fallback_models=["gpt-4", "claude"],  # More expensive models as fallback
        budget=0.1  # Set a budget of $0.1
    )
    
    # Example prompts of varying complexity
    prompts = [
        "What is the weather like today?",  # Simple query
        "Explain quantum computing in detail.",  # Complex query
        "Write a detailed analysis of climate change.",  # Very complex query
    ]
    
    print("Running cost optimization examples...")
    print("Budget: $0.1")
    print("\nProcessing prompts:")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response = await wrapper.generate(prompt)
        print(f"Response: {response[:100]}...")  # Print first 100 chars
        print(f"Total cost so far: ${wrapper.cost_tracker.get_total_cost():.4f}")
        print(f"Models used: {wrapper.cost_tracker.get_model_usage()}")

if __name__ == "__main__":
    asyncio.run(main()) 