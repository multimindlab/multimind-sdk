"""
Example demonstrating how to use the advanced ensemble system.
"""

import asyncio
import os
from multimind.core.provider import ProviderConfig
from multimind.core.router import Router, TaskType, TaskConfig, RoutingStrategy
from multimind.ensemble.advanced import AdvancedEnsemble, EnsembleMethod
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
    
    # Initialize advanced ensemble
    ensemble = AdvancedEnsemble(router)
    
    # Example 1: Weighted Voting
    print("\nExample 1: Weighted Voting")
    prompt = "Explain quantum computing in simple terms."
    
    # Get results from both providers (handle failures gracefully)
    results = []
    try:
        openai_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
    except Exception as e:
        print(f"Warning: OpenAI provider failed: {e}")
        results.append(None)
    
    try:
        claude_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
    except Exception as e:
        print(f"Warning: Claude provider failed: {e}")
        results.append(None)
    
    # Combine results using weighted voting (handles 0, 1, or 2+ results automatically)
    ensemble_result = await ensemble.combine_results(
        results=results,
        method=EnsembleMethod.WEIGHTED_VOTING,
        task_type=TaskType.TEXT_GENERATION,
        weights={"openai": 0.6, "claude": 0.4}
    )
    
    print(f"\nSelected Result: {ensemble_result.result.text}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 2: Confidence Cascade
    print("\nExample 2: Confidence Cascade")
    prompt = "Write a short story about a robot learning to paint."
    
    # Get results from both providers (handle failures gracefully)
    results = []
    try:
        openai_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
    except Exception as e:
        print(f"Warning: OpenAI provider failed: {e}")
        results.append(None)
    
    try:
        claude_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
    except Exception as e:
        print(f"Warning: Claude provider failed: {e}")
        results.append(None)
    
    # Combine results using confidence cascade (handles 0, 1, or 2+ results automatically)
    ensemble_result = await ensemble.combine_results(
        results=results,
        method=EnsembleMethod.CONFIDENCE_CASCADE,
        task_type=TaskType.TEXT_GENERATION,
        confidence_threshold=0.8
    )
    
    print(f"\nSelected Result: {ensemble_result.result.text}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 3: Parallel Voting with LLM Evaluator
    print("\nExample 3: Parallel Voting with LLM Evaluator")
    prompt = "Explain the concept of machine learning to a 10-year-old."
    
    # Get results from both providers (handle failures gracefully)
    results = []
    try:
        openai_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
    except Exception as e:
        print(f"Warning: OpenAI provider failed: {e}")
        results.append(None)
    
    try:
        claude_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
    except Exception as e:
        print(f"Warning: Claude provider failed: {e}")
        results.append(None)
    
    # Combine results using parallel voting (handles 0, 1, or 2+ results automatically)
    ensemble_result = await ensemble.combine_results(
        results=results,
        method=EnsembleMethod.PARALLEL_VOTING,
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"\nSelected Result: {ensemble_result.result.text}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 4: Majority Voting
    print("\nExample 4: Majority Voting")
    prompt = "What is the capital of France?"
    
    # Get results from both providers (handle failures gracefully)
    results = []
    try:
        openai_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
    except Exception as e:
        print(f"Warning: OpenAI provider failed: {e}")
        results.append(None)
    
    try:
        claude_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
    except Exception as e:
        print(f"Warning: Claude provider failed: {e}")
        results.append(None)
    
    # Combine results using majority voting (handles 0, 1, or 2+ results automatically)
    ensemble_result = await ensemble.combine_results(
        results=results,
        method=EnsembleMethod.MAJORITY_VOTING,
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"\nSelected Result: {ensemble_result.result.text}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 5: Rank-Based Selection
    print("\nExample 5: Rank-Based Selection")
    prompt = "Write a haiku about artificial intelligence."
    
    # Get results from both providers (handle failures gracefully)
    results = []
    try:
        openai_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="openai",
            model="gpt-4"
        )
        results.append(openai_result)
    except Exception as e:
        print(f"Warning: OpenAI provider failed: {e}")
        results.append(None)
    
    try:
        claude_result = await router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider="claude",
            model="claude-3-sonnet"
        )
        results.append(claude_result)
    except Exception as e:
        print(f"Warning: Claude provider failed: {e}")
        results.append(None)
    
    # Combine results using rank-based selection (handles 0, 1, or 2+ results automatically)
    ensemble_result = await ensemble.combine_results(
        results=results,
        method=EnsembleMethod.RANK_BASED,
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"\nSelected Result: {ensemble_result.result.text}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 