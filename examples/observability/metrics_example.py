"""
Example demonstrating how to use the metrics collection system.
"""

import asyncio
import os
from datetime import datetime, timedelta
from multimind.core.provider import ProviderConfig
from multimind.core.router import Router, TaskType, TaskConfig, RoutingStrategy
from multimind.observability.metrics import MetricsCollector
from multimind.providers.openai import OpenAIProvider
from multimind.providers.claude import ClaudeProvider

async def main():
    # Initialize metrics collector
    metrics = MetricsCollector()
    
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
            }
        }
    )
    
    router.configure_task(TaskType.TEXT_GENERATION, text_generation_config)
    
    # Example 1: Record metrics for successful requests
    print("\nExample 1: Recording metrics for successful requests")
    
    # Simulate some successful requests
    for i in range(3):
        # Record latency
        metrics.record_latency(
            provider="openai",
            task_type="text_generation",
            model="gpt-4",
            latency_ms=150.0 + i * 10,
            metadata={"request_id": f"req_{i}"}
        )
        
        # Record cost
        metrics.record_cost(
            provider="openai",
            task_type="text_generation",
            model="gpt-4",
            cost=0.002 * (i + 1),
            metadata={"request_id": f"req_{i}"}
        )
        
        # Record tokens
        metrics.record_tokens(
            provider="openai",
            task_type="text_generation",
            model="gpt-4",
            tokens=100 * (i + 1),
            metadata={"request_id": f"req_{i}"}
        )
    
    # Example 2: Record metrics for failed requests
    print("\nExample 2: Recording metrics for failed requests")
    
    # Simulate some failed requests
    for i in range(2):
        metrics.record_error(
            provider="claude",
            task_type="text_generation",
            model="claude-3-sonnet",
            error_type="RateLimitError",
            error_message="Too many requests",
            metadata={"request_id": f"error_{i}"}
        )
    
    # Example 3: Get metrics summary
    print("\nExample 3: Getting metrics summary")
    summary = metrics.get_summary()
    print("\nMetrics Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Example 4: Get filtered metrics
    print("\nExample 4: Getting filtered metrics")
    filtered_metrics = metrics.get_metrics(
        metric_type="Latency",
        provider="openai",
        start_time=datetime.now() - timedelta(hours=1)
    )
    print(f"\nFound {len(filtered_metrics)} latency metrics for OpenAI")
    
    # Example 5: Save metrics to file
    print("\nExample 5: Saving metrics to file")
    metrics.save_metrics("example_metrics.json")
    print("Metrics saved to example_metrics.json")

if __name__ == "__main__":
    asyncio.run(main()) 