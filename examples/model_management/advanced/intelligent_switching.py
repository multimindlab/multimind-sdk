"""
Example demonstrating intelligent model switching and optimization features.
"""

import asyncio
import time
from typing import List, Dict
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

async def run_intelligent_examples():
    # Initialize the model factory
    factory = ModelFactory()

    # Create a multi-model wrapper with intelligent switching
    multi_model = MultiModelWrapper(
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

    # Example 1: Task-specific model selection
    print("Example 1: Task-specific model selection")
    tasks = [
        {
            "type": "creative",
            "prompt": "Write a creative story about a robot learning to paint",
            "temperature": 0.9
        },
        {
            "type": "technical",
            "prompt": "Explain the concept of quantum entanglement in detail",
            "temperature": 0.3
        },
        {
            "type": "code",
            "prompt": "Write a Python function to implement quicksort",
            "temperature": 0.2
        }
    ]

    for task in tasks:
        print(f"\nExecuting {task['type']} task:")
        start_time = time.time()
        response = await multi_model.generate(
            prompt=task["prompt"],
            temperature=task["temperature"]
        )
        print(f"Response: {response}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Example 2: Performance-based model selection
    print("\nExample 2: Performance-based model selection")
    print("Running multiple requests to build performance metrics...")
    
    for _ in range(5):
        await multi_model.generate(
            prompt="What is artificial intelligence?",
            temperature=0.7
        )
        await asyncio.sleep(0.1)  # Small delay between requests

    # Get performance metrics
    performance = multi_model.get_model_performance()
    print("\nModel Performance Metrics:")
    for model, metrics in performance.items():
        print(f"\n{model}:")
        print(f"  Performance Score: {metrics['performance_score']:.2f}")
        print(f"  Success Rate: {metrics['success_rate']:.2f}")
        print(f"  Average Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"  Error Rate: {metrics['error_rate']:.2f}")

    # Example 3: Automatic fallback with performance tracking
    print("\nExample 3: Automatic fallback with performance tracking")
    try:
        # Simulate a failure in the primary model
        print("Simulating primary model failure...")
        response = await multi_model.generate(
            prompt="This should trigger fallback",
            temperature=0.7,
            force_error=True  # This would be handled by your error simulation
        )
    except Exception as e:
        print(f"Error handled: {str(e)}")
        print("Fallback should have been triggered automatically")

    # Example 4: Complex conversation with dynamic model selection
    print("\nExample 4: Complex conversation with dynamic model selection")
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "I need help with both creative writing and technical concepts."}
    ]

    # Creative task
    print("\nHandling creative task:")
    creative_response = await multi_model.chat(
        messages=conversation + [{"role": "user", "content": "Write a poem about technology"}],
        temperature=0.8
    )
    print(f"Creative Response: {creative_response}")

    # Technical task
    print("\nHandling technical task:")
    technical_response = await multi_model.chat(
        messages=conversation + [{"role": "user", "content": "Explain how neural networks work"}],
        temperature=0.3
    )
    print(f"Technical Response: {technical_response}")

    # Example 5: Batch processing with intelligent model selection
    print("\nExample 5: Batch processing with intelligent model selection")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming our world.",
        "The quantum realm holds many mysteries."
    ]

    print("Generating embeddings with intelligent model selection:")
    embeddings = await multi_model.embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Show final performance metrics
    print("\nFinal Model Performance Metrics:")
    final_performance = multi_model.get_model_performance()
    for model, metrics in final_performance.items():
        print(f"\n{model}:")
        print(f"  Performance Score: {metrics['performance_score']:.2f}")
        print(f"  Success Rate: {metrics['success_rate']:.2f}")
        print(f"  Average Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"  Error Rate: {metrics['error_rate']:.2f}")

if __name__ == "__main__":
    asyncio.run(run_intelligent_examples()) 