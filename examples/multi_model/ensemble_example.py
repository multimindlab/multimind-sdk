"""
Example demonstrating how to use multiple models simultaneously with the MultimindSDK.
"""

import asyncio
import os
from typing import List, Dict, Any
from multimind.core.provider import ProviderConfig
from multimind.core.router import Router, TaskType, TaskConfig, RoutingStrategy
from multimind.providers.openai import OpenAIProvider
from multimind.providers.claude import ClaudeProvider

async def main():
    # Initialize providers
    openai_config = ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )
    claude_config = ProviderConfig(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com"
    )
    
    openai_provider = OpenAIProvider(openai_config)
    claude_provider = ClaudeProvider(claude_config)
    
    # Initialize router
    router = Router()
    router.register_provider("openai", openai_provider)
    router.register_provider("claude", claude_provider)
    
    # Configure tasks
    text_generation_config = TaskConfig(
        preferred_providers=["openai", "claude"],
        fallback_providers=[],
        routing_strategy=RoutingStrategy.ENSEMBLE,
        ensemble_config={
            "method": "weighted_voting",
            "weights": {
                "openai": 0.6,
                "claude": 0.4
            },
            "min_confidence": 0.7
        }
    )
    
    chat_config = TaskConfig(
        preferred_providers=["claude", "openai"],
        fallback_providers=[],
        routing_strategy=RoutingStrategy.CASCADE,
        cascade_config={
            "quality_threshold": 0.8,
            "max_retries": 2
        }
    )
    
    router.configure_task(TaskType.TEXT_GENERATION, text_generation_config)
    router.configure_task(TaskType.CHAT, chat_config)
    
    # Example 1: Text generation with ensemble
    prompt = "Write a short story about a robot learning to paint."
    print("\nGenerating text with ensemble strategy...")
    result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        model="gpt-4"  # This will be used as a base model
    )
    print(f"Ensemble result: {result.result}")
    print(f"Cost: ${result.cost_estimate_usd:.4f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    
    # Example 2: Chat with cascade
    messages = [
        {"role": "user", "content": "What are the key differences between Python and JavaScript?"}
    ]
    print("\nChatting with cascade strategy...")
    result = await router.route(
        TaskType.CHAT,
        messages,
        model="claude-3-sonnet"  # This will be used as the primary model
    )
    print(f"Cascade result: {result.result}")
    print(f"Cost: ${result.cost_estimate_usd:.4f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    
    # Example 3: Custom ensemble with different models
    print("\nCustom ensemble with different models...")
    results = await asyncio.gather(*[
        router.route(
            TaskType.TEXT_GENERATION,
            "Explain quantum computing in simple terms.",
            model=model
        )
        for model in ["gpt-4", "claude-3-sonnet"]
    ])
    
    # Combine results manually
    combined_result = await combine_results(results)
    print(f"Combined result: {combined_result}")
    
    # Print usage statistics
    print("\nUsage Statistics:")
    for provider_name, stats in router.usage_stats.items():
        print(f"\n{provider_name}:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Successful requests: {stats['successful_requests']}")
        print(f"  Failed requests: {stats['failed_requests']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Total cost: ${stats['total_cost']:.4f}")
        print(f"  Average latency: {stats['average_latency']:.0f}ms")

async def combine_results(results: List[Any]) -> str:
    """Combine multiple results using a simple voting mechanism."""
    # In a real implementation, this would be more sophisticated
    # For example, using semantic similarity, confidence scores, etc.
    return results[0].result  # For now, just return the first result

if __name__ == "__main__":
    asyncio.run(main()) 