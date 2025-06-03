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
    
    # Get results from both providers
    openai_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="openai",
        model="gpt-4"
    )
    
    claude_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="claude",
        model="claude-3-sonnet"
    )
    
    # Combine results using weighted voting
    ensemble_result = await ensemble.combine_results(
        results=[openai_result, claude_result],
        method=EnsembleMethod.WEIGHTED_VOTING,
        task_type=TaskType.TEXT_GENERATION,
        weights={"openai": 0.6, "claude": 0.4}
    )
    
    print(f"\nSelected Result: {ensemble_result.result.result}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 2: Confidence Cascade
    print("\nExample 2: Confidence Cascade")
    prompt = "Write a short story about a robot learning to paint."
    
    # Get results from both providers
    openai_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="openai",
        model="gpt-4"
    )
    
    claude_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="claude",
        model="claude-3-sonnet"
    )
    
    # Combine results using confidence cascade
    ensemble_result = await ensemble.combine_results(
        results=[openai_result, claude_result],
        method=EnsembleMethod.CONFIDENCE_CASCADE,
        task_type=TaskType.TEXT_GENERATION,
        confidence_threshold=0.8
    )
    
    print(f"\nSelected Result: {ensemble_result.result.result}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 3: Parallel Voting with LLM Evaluator
    print("\nExample 3: Parallel Voting with LLM Evaluator")
    prompt = "Explain the concept of machine learning to a 10-year-old."
    
    # Get results from both providers
    openai_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="openai",
        model="gpt-4"
    )
    
    claude_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="claude",
        model="claude-3-sonnet"
    )
    
    # Combine results using parallel voting
    ensemble_result = await ensemble.combine_results(
        results=[openai_result, claude_result],
        method=EnsembleMethod.PARALLEL_VOTING,
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"\nSelected Result: {ensemble_result.result.result}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 4: Majority Voting
    print("\nExample 4: Majority Voting")
    prompt = "What is the capital of France?"
    
    # Get results from both providers
    openai_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="openai",
        model="gpt-4"
    )
    
    claude_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="claude",
        model="claude-3-sonnet"
    )
    
    # Combine results using majority voting
    ensemble_result = await ensemble.combine_results(
        results=[openai_result, claude_result],
        method=EnsembleMethod.MAJORITY_VOTING,
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"\nSelected Result: {ensemble_result.result.result}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")
    
    # Example 5: Rank-Based Selection
    print("\nExample 5: Rank-Based Selection")
    prompt = "Write a haiku about artificial intelligence."
    
    # Get results from both providers
    openai_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="openai",
        model="gpt-4"
    )
    
    claude_result = await router.route(
        TaskType.TEXT_GENERATION,
        prompt,
        provider="claude",
        model="claude-3-sonnet"
    )
    
    # Combine results using rank-based selection
    ensemble_result = await ensemble.combine_results(
        results=[openai_result, claude_result],
        method=EnsembleMethod.RANK_BASED,
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"\nSelected Result: {ensemble_result.result.result}")
    print(f"Confidence: {ensemble_result.confidence.score:.2f}")
    print(f"Explanation: {ensemble_result.confidence.explanation}")
    print("\nProvider Votes:")
    for provider, vote in ensemble_result.provider_votes.items():
        print(f"{provider}: {vote:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 