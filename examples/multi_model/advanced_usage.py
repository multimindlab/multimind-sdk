"""
Advanced usage examples of the MultiModelWrapper demonstrating complex scenarios.
"""

import asyncio
import json
from typing import List, Dict
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

async def run_advanced_examples():
    # Initialize the model factory
    factory = ModelFactory()

    # Example 1: Complex model configuration with multiple fallbacks
    print("Example 1: Complex model configuration")
    multi_model = MultiModelWrapper(
        model_factory=factory,
        primary_model="openai",
        fallback_models=["claude", "ollama"],
        model_weights={
            "openai": 0.5,
            "claude": 0.3,
            "ollama": 0.2
        }
    )

    # Example 2: Task-specific model selection
    print("\nExample 2: Task-specific model selection")
    tasks = [
        {
            "type": "creative",
            "prompt": "Write a creative story about a robot learning to paint",
            "temperature": 0.9,
            "model_weights": {"openai": 0.8, "claude": 0.2}  # Prefer OpenAI for creative tasks
        },
        {
            "type": "technical",
            "prompt": "Explain the concept of quantum entanglement in detail",
            "temperature": 0.3,
            "model_weights": {"claude": 0.8, "openai": 0.2}  # Prefer Claude for technical tasks
        }
    ]

    for task in tasks:
        print(f"\nExecuting {task['type']} task:")
        response = await multi_model.generate(
            prompt=task["prompt"],
            temperature=task["temperature"],
            model_weights=task["model_weights"]
        )
        print(f"Response: {response}")

    # Example 3: Complex chat conversation with context management
    print("\nExample 3: Complex chat conversation")
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant with expertise in multiple domains."},
        {"role": "user", "content": "I need help with both creative writing and technical concepts."},
        {"role": "assistant", "content": "I can help you with both! What would you like to explore first?"},
        {"role": "user", "content": "First, help me write a poem about technology, then explain how neural networks work."}
    ]

    # Split the complex request into two parts
    poem_prompt = "Write a short poem about technology and its impact on humanity."
    tech_prompt = "Explain how neural networks work in simple terms."

    # Get creative response for the poem
    print("\nGenerating poem:")
    poem_response = await multi_model.chat(
        messages=conversation + [{"role": "user", "content": poem_prompt}],
        temperature=0.8,
        model_weights={"openai": 0.7, "claude": 0.3}
    )
    print(f"Poem: {poem_response}")

    # Get technical response for neural networks
    print("\nExplaining neural networks:")
    tech_response = await multi_model.chat(
        messages=conversation + [{"role": "user", "content": tech_prompt}],
        temperature=0.3,
        model_weights={"claude": 0.7, "openai": 0.3}
    )
    print(f"Explanation: {tech_response}")

    # Example 4: Batch processing with different models
    print("\nExample 4: Batch processing")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming our world.",
        "The quantum realm holds many mysteries."
    ]

    # Generate embeddings for all texts
    print("Generating embeddings for multiple texts:")
    embeddings = await multi_model.embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Example 5: Streaming with model fallback
    print("\nExample 5: Streaming with fallback")
    print("Generating stream with potential fallback:")
    try:
        async for chunk in multi_model.generate_stream(
            prompt="Write a detailed analysis of the impact of AI on healthcare:",
            temperature=0.7,
            max_tokens=200
        ):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\nFallback triggered: {str(e)}")
        # The wrapper will automatically try fallback models

if __name__ == "__main__":
    asyncio.run(run_advanced_examples()) 