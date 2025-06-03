"""
Example demonstrating advanced intelligent model selection scenarios.
"""

import asyncio
import time
from typing import List, Dict, Any
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

async def run_advanced_selection_examples():
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

    # Example 1: Content-based model selection
    print("Example 1: Content-based model selection")
    content_types = [
        {
            "type": "creative_writing",
            "prompt": "Write a haiku about artificial intelligence",
            "temperature": 0.9,
            "expected_model": "openai"  # Better for creative tasks
        },
        {
            "type": "technical_analysis",
            "prompt": "Analyze the time complexity of the following algorithm: [code]",
            "temperature": 0.3,
            "expected_model": "claude"  # Better for technical analysis
        },
        {
            "type": "code_generation",
            "prompt": "Write a function to implement binary search in Python",
            "temperature": 0.2,
            "expected_model": "openai"  # Better for code generation
        }
    ]

    for content in content_types:
        print(f"\nExecuting {content['type']} task:")
        start_time = time.time()
        response = await multi_model.generate(
            prompt=content["prompt"],
            temperature=content["temperature"]
        )
        print(f"Response: {response}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Example 2: Context-aware model selection
    print("\nExample 2: Context-aware model selection")
    conversation_context = [
        {
            "context": "technical_discussion",
            "messages": [
                {"role": "system", "content": "You are a technical expert."},
                {"role": "user", "content": "Explain the concept of quantum computing."},
                {"role": "assistant", "content": "Quantum computing uses quantum bits..."},
                {"role": "user", "content": "How does quantum entanglement work?"}
            ],
            "expected_model": "claude"  # Better for technical explanations
        },
        {
            "context": "creative_writing",
            "messages": [
                {"role": "system", "content": "You are a creative writer."},
                {"role": "user", "content": "Write a short story about a robot."},
                {"role": "assistant", "content": "In a world where robots..."},
                {"role": "user", "content": "Continue the story with a plot twist."}
            ],
            "expected_model": "openai"  # Better for creative writing
        }
    ]

    for context in conversation_context:
        print(f"\nExecuting {context['context']} conversation:")
        response = await multi_model.chat(
            messages=context["messages"],
            temperature=0.7
        )
        print(f"Response: {response}")

    # Example 3: Performance-based dynamic weighting
    print("\nExample 3: Performance-based dynamic weighting")
    print("Running performance test with different model configurations...")
    
    # Test different model configurations
    configurations = [
        {"name": "balanced", "weights": {"openai": 0.4, "claude": 0.4, "ollama": 0.2}},
        {"name": "openai_heavy", "weights": {"openai": 0.7, "claude": 0.2, "ollama": 0.1}},
        {"name": "claude_heavy", "weights": {"openai": 0.2, "claude": 0.7, "ollama": 0.1}}
    ]

    for config in configurations:
        print(f"\nTesting {config['name']} configuration:")
        multi_model.model_weights = config["weights"]
        
        # Run multiple requests to build performance metrics
        for _ in range(3):
            await multi_model.generate(
                prompt="What is machine learning?",
                temperature=0.7
            )
            await asyncio.sleep(0.1)

        # Get performance metrics
        performance = multi_model.get_model_performance()
        print(f"Performance metrics for {config['name']}:")
        for model, metrics in performance.items():
            print(f"\n{model}:")
            print(f"  Performance Score: {metrics['performance_score']:.2f}")
            print(f"  Success Rate: {metrics['success_rate']:.2f}")
            print(f"  Average Response Time: {metrics['avg_response_time']:.2f}s")

    # Example 4: Task-specific optimization
    print("\nExample 4: Task-specific optimization")
    tasks = [
        {
            "name": "code_review",
            "prompt": "Review this Python code for best practices:\n\ndef example():\n    x = 1\n    return x",
            "temperature": 0.3,
            "expected_model": "claude"  # Better for code review
        },
        {
            "name": "language_translation",
            "prompt": "Translate this to French: 'Hello, how are you?'",
            "temperature": 0.3,
            "expected_model": "openai"  # Better for translation
        },
        {
            "name": "mathematical_proof",
            "prompt": "Prove that the square root of 2 is irrational",
            "temperature": 0.3,
            "expected_model": "claude"  # Better for mathematical reasoning
        }
    ]

    for task in tasks:
        print(f"\nExecuting {task['name']} task:")
        start_time = time.time()
        response = await multi_model.generate(
            prompt=task["prompt"],
            temperature=task["temperature"]
        )
        print(f"Response: {response}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Example 5: Adaptive model selection
    print("\nExample 5: Adaptive model selection")
    print("Testing adaptive model selection with varying complexity...")
    
    complexity_levels = [
        {
            "level": "simple",
            "prompt": "What is the capital of France?",
            "expected_model": "ollama"  # Simple queries can use lighter models
        },
        {
            "level": "moderate",
            "prompt": "Explain the concept of recursion in programming",
            "expected_model": "openai"  # Moderate complexity needs balanced model
        },
        {
            "level": "complex",
            "prompt": "Analyze the implications of quantum computing on cryptography",
            "expected_model": "claude"  # Complex topics need more capable models
        }
    ]

    for level in complexity_levels:
        print(f"\nExecuting {level['level']} complexity task:")
        start_time = time.time()
        response = await multi_model.generate(
            prompt=level["prompt"],
            temperature=0.7
        )
        print(f"Response: {response}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

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
    asyncio.run(run_advanced_selection_examples()) 