"""
Example demonstrating fine-tuning integration and advanced model management features.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

class FineTunedMultiModelWrapper(MultiModelWrapper):
    """Extended MultiModelWrapper with fine-tuning and advanced model management."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_tuned_models = {}  # Track fine-tuned model versions
        self.training_data = {}  # Store training data per model
        self.model_versions = {}  # Track model versions and their performance
        self.custom_models = {}  # Store custom model configurations
        self.evaluation_metrics = {}  # Store model evaluation results

    async def _prepare_training_data(
        self,
        model_name: str,
        training_data: List[Dict[str, str]],
        validation_split: float = 0.1
    ) -> Dict[str, List[Dict[str, str]]]:
        """Prepare training data for fine-tuning."""
        # Split data into training and validation sets
        split_idx = int(len(training_data) * (1 - validation_split))
        return {
            'training': training_data[:split_idx],
            'validation': training_data[split_idx:]
        }

    async def _evaluate_model(
        self,
        model_name: str,
        test_data: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        metrics = {
            'accuracy': 0.0,
            'response_time': 0.0,
            'token_usage': 0
        }
        
        total_time = 0
        correct_responses = 0
        
        for example in test_data:
            start_time = time.time()
            try:
                response = await self.models[model_name].generate(
                    prompt=example['prompt'],
                    temperature=0.3
                )
                total_time += time.time() - start_time
                
                # Simple accuracy check (customize based on your needs)
                if example.get('expected_response') and example['expected_response'] in response:
                    correct_responses += 1
                
                metrics['token_usage'] += len(response.split())
            except Exception as e:
                print(f"Error evaluating model {model_name}: {e}")
        
        if test_data:
            metrics['accuracy'] = correct_responses / len(test_data)
            metrics['response_time'] = total_time / len(test_data)
        
        return metrics

    async def fine_tune_model(
        self,
        model_name: str,
        training_data: List[Dict[str, str]],
        validation_split: float = 0.1,
        epochs: int = 3,
        learning_rate: float = 1e-5,
        **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune a model with custom training data."""
        print(f"Starting fine-tuning for model {model_name}...")
        
        # Prepare training data
        data_splits = await self._prepare_training_data(
            model_name,
            training_data,
            validation_split
        )
        
        # Store training data
        self.training_data[model_name] = data_splits
        
        # Simulate fine-tuning process
        # In a real implementation, this would call the model's fine-tuning API
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # Simulate training progress
            await asyncio.sleep(1)
        
        # Evaluate the fine-tuned model
        evaluation_metrics = await self._evaluate_model(
            model_name,
            data_splits['validation']
        )
        
        # Store fine-tuned model information
        model_version = f"{model_name}_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fine_tuned_models[model_version] = {
            'base_model': model_name,
            'training_data_size': len(training_data),
            'evaluation_metrics': evaluation_metrics,
            'created_at': datetime.now().isoformat(),
            'parameters': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                **kwargs
            }
        }
        
        return {
            'model_version': model_version,
            'evaluation_metrics': evaluation_metrics,
            'training_stats': {
                'epochs': epochs,
                'training_samples': len(data_splits['training']),
                'validation_samples': len(data_splits['validation'])
            }
        }

    async def create_custom_model(
        self,
        model_name: str,
        base_model: str,
        custom_config: Dict[str, Any]
    ) -> str:
        """Create a custom model configuration."""
        model_id = f"{model_name}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.custom_models[model_id] = {
            'base_model': base_model,
            'config': custom_config,
            'created_at': datetime.now().isoformat()
        }
        return model_id

    async def compare_models(
        self,
        model_versions: List[str],
        test_data: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare performance of different model versions."""
        results = {}
        for version in model_versions:
            if version in self.fine_tuned_models:
                metrics = await self._evaluate_model(
                    self.fine_tuned_models[version]['base_model'],
                    test_data
                )
                results[version] = metrics
        return results

    def get_model_versions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all model versions."""
        return {
            'fine_tuned': self.fine_tuned_models,
            'custom': self.custom_models,
            'evaluation_metrics': self.evaluation_metrics
        }

async def run_fine_tuning_examples():
    # Initialize the model factory
    factory = ModelFactory()

    # Create a fine-tuned multi-model wrapper
    fine_tuned_model = FineTunedMultiModelWrapper(
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

    # Example 1: Fine-tuning a model
    print("Example 1: Fine-tuning a model")
    training_data = [
        {
            "prompt": "What is the capital of France?",
            "expected_response": "Paris"
        },
        {
            "prompt": "What is the largest planet in our solar system?",
            "expected_response": "Jupiter"
        },
        {
            "prompt": "Who wrote Romeo and Juliet?",
            "expected_response": "William Shakespeare"
        }
    ]

    fine_tuning_result = await fine_tuned_model.fine_tune_model(
        model_name="openai",
        training_data=training_data,
        epochs=3,
        learning_rate=1e-5
    )
    print("\nFine-tuning results:")
    print(json.dumps(fine_tuning_result, indent=2))

    # Example 2: Creating a custom model
    print("\nExample 2: Creating a custom model")
    custom_config = {
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5
    }

    custom_model_id = await fine_tuned_model.create_custom_model(
        model_name="specialized",
        base_model="openai",
        custom_config=custom_config
    )
    print(f"Created custom model with ID: {custom_model_id}")

    # Example 3: Model comparison
    print("\nExample 3: Comparing model versions")
    test_data = [
        {
            "prompt": "What is the capital of Japan?",
            "expected_response": "Tokyo"
        },
        {
            "prompt": "What is the chemical symbol for gold?",
            "expected_response": "Au"
        }
    ]

    comparison_results = await fine_tuned_model.compare_models(
        model_versions=[list(fine_tuned_model.fine_tuned_models.keys())[0]],
        test_data=test_data
    )
    print("\nModel comparison results:")
    print(json.dumps(comparison_results, indent=2))

    # Example 4: Advanced model management
    print("\nExample 4: Advanced model management")
    model_versions = fine_tuned_model.get_model_versions()
    print("\nModel versions information:")
    print(json.dumps(model_versions, indent=2))

    # Example 5: Model evaluation and metrics
    print("\nExample 5: Model evaluation and metrics")
    evaluation_data = [
        {
            "prompt": "Explain quantum computing in simple terms",
            "expected_response": "quantum"
        },
        {
            "prompt": "What is machine learning?",
            "expected_response": "learning"
        }
    ]

    for model_name in fine_tuned_model.models:
        metrics = await fine_tuned_model._evaluate_model(
            model_name,
            evaluation_data
        )
        print(f"\nEvaluation metrics for {model_name}:")
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    asyncio.run(run_fine_tuning_examples()) 