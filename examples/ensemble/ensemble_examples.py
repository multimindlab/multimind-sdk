"""
Comprehensive examples of using the MultiMind Ensemble system.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from multimind import Router, TaskType
from multimind.core.provider import (
    EmbeddingResult,
    GenerationResult,
    ImageAnalysisResult,
    ProviderConfig,
)
from multimind.ensemble import AdvancedEnsemble, EnsembleMethod
from multimind.providers.claude import ClaudeProvider
from multimind.providers.ollama import OllamaProvider
from multimind.providers.openai import OpenAIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serialize_ensemble_result(result):
    """Convert EnsembleResult or nested Pydantic models to JSON-serializable dict."""
    if hasattr(result, 'model_dump'):
        return result.model_dump()
    elif hasattr(result, 'dict'):
        return result.dict()
    elif isinstance(result, dict):
        return {k: serialize_ensemble_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [serialize_ensemble_result(item) for item in result]
    else:
        return result

class EnsembleExamples:
    def __init__(self):
        """Initialize the ensemble examples."""
        self.router = Router()
        self._register_providers()
        self.ensemble = AdvancedEnsemble(self.router)

    def _register_providers(self):
        """Register available providers with the router."""
        # Register OpenAI provider
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai_config = ProviderConfig(
                api_key=openai_api_key,
                base_url="https://api.openai.com/v1"
            )
            openai_provider = OpenAIProvider(openai_config)
            self.router.register_provider("openai", openai_provider)
            logger.info("Registered OpenAI provider")
        else:
            logger.warning("OPENAI_API_KEY not found. OpenAI provider will not be available.")

        # Register Anthropic (Claude) provider
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if anthropic_api_key:
            claude_config = ProviderConfig(
                api_key=anthropic_api_key,
                base_url="https://api.anthropic.com"
            )
            claude_provider = ClaudeProvider(claude_config)
            self.router.register_provider("anthropic", claude_provider)
            logger.info("Registered Anthropic provider")
        else:
            logger.warning("ANTHROPIC_API_KEY or CLAUDE_API_KEY not found. Anthropic provider will not be available.")

        # Register Ollama provider (no API key needed)
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
        ollama_config = ProviderConfig(
            api_key=None,
            api_base=ollama_base,
            timeout=ollama_timeout
        )
        ollama_provider = OllamaProvider(ollama_config)
        self.router.register_provider("ollama", ollama_provider)
        logger.info(f"Registered Ollama provider (base_url: {ollama_base})")

    async def run_text_generation_ensemble(
        self,
        prompt: str,
        providers: List[str] = ["openai", "anthropic", "ollama"]
    ) -> Dict[str, Any]:
        """Run text generation ensemble with multiple providers."""
        logger.info(f"Running text generation ensemble for prompt: {prompt}")

        # Filter providers to only those that are registered
        available_providers = [p for p in providers if p in self.router.providers]
        if not available_providers:
            raise ValueError(f"None of the requested providers {providers} are registered or available")

        # Get results from all providers
        results = await asyncio.gather(*[
            self.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
            )
            for provider in available_providers
        ])

        # Try different ensemble methods
        ensemble_results = {}

        # 1. Weighted Voting
        # If only one provider, use weight 1.0
        if len(available_providers) == 1:
            weights = {available_providers[0]: 1.0}
        else:
            weights = {p: 1.0 / len(available_providers) for p in available_providers}
            if "openai" in available_providers and "anthropic" in available_providers:
                weights["openai"] = 0.4
                weights["anthropic"] = 0.4
                if "ollama" in available_providers:
                    weights["ollama"] = 0.2
                else:
                    weights["openai"] = 0.5
                    weights["anthropic"] = 0.5
            elif "openai" in available_providers:
                weights["openai"] = 0.6
                remaining_weight = 0.4 / (len(available_providers) - 1)
                for p in available_providers:
                    if p != "openai":
                        weights[p] = remaining_weight
            elif "anthropic" in available_providers:
                weights["anthropic"] = 0.6
                remaining_weight = 0.4 / (len(available_providers) - 1)
                for p in available_providers:
                    if p != "anthropic":
                        weights[p] = remaining_weight

        weighted_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.TEXT_GENERATION,
            weights=weights
        )
        ensemble_results["weighted_voting"] = weighted_result

        # 2. Confidence Cascade
        confidence_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.TEXT_GENERATION,
            confidence_threshold=0.8
        )
        ensemble_results["confidence_cascade"] = confidence_result

        # 3. Parallel Voting
        parallel_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.PARALLEL_VOTING,
            task_type=TaskType.TEXT_GENERATION
        )
        ensemble_results["parallel_voting"] = parallel_result

        # 4. Majority Voting
        majority_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.MAJORITY_VOTING,
            task_type=TaskType.TEXT_GENERATION
        )
        ensemble_results["majority_voting"] = majority_result

        # 5. Rank Based
        rank_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.RANK_BASED,
            task_type=TaskType.TEXT_GENERATION
        )
        ensemble_results["rank_based"] = rank_result

        return ensemble_results

    async def run_embedding_ensemble(
        self,
        text: str,
        providers: List[str] = ["openai", "ollama"]
    ) -> Dict[str, Any]:
        """Run embedding ensemble with multiple providers."""
        logger.info(f"Running embedding ensemble for text: {text[:100]}...")

        # Filter providers to only those that are registered and support embeddings
        available_providers = [p for p in providers if p in self.router.providers]
        if not available_providers:
            raise ValueError(f"None of the requested providers {providers} are registered or available")

        # Get embeddings from all providers with error handling
        results = []
        successful_providers = []
        for provider in available_providers:
            try:
                model = "text-embedding-ada-002" if provider == "openai" else "nomic-embed-text"
                result = await self.router.route(
                    TaskType.EMBEDDINGS,
                    text,
                    provider=provider,
                    model=model
                )
                results.append(result)
                successful_providers.append(provider)
            except Exception as e:
                logger.warning(f"Failed to get embeddings from {provider}: {str(e)}. Skipping this provider.")
                # Continue with other providers
                continue

        if not results:
            raise ValueError(f"All embedding providers failed. Available providers: {available_providers}")

        # Combine embeddings using weighted voting
        # Use only successful providers for weight calculation
        # If only one provider succeeded, use weight 1.0
        if len(successful_providers) == 1:
            weights = {successful_providers[0]: 1.0}
        else:
            weights = {p: 1.0 / len(successful_providers) for p in successful_providers}
            if "openai" in successful_providers:
                weights["openai"] = 0.6
                remaining_weight = 0.4 / (len(successful_providers) - 1)
                for p in successful_providers:
                    if p != "openai":
                        weights[p] = remaining_weight

        combined_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.EMBEDDINGS,
            weights=weights
        )

        return combined_result

    async def run_image_analysis_ensemble(
        self,
        image_path: str,
        providers: List[str] = ["openai", "anthropic"]
    ) -> Dict[str, Any]:
        """Run image analysis ensemble with multiple providers."""
        logger.info(f"Running image analysis ensemble for image: {image_path}")

        # Filter providers to only those that are registered
        available_providers = [p for p in providers if p in self.router.providers]
        if not available_providers:
            raise ValueError(f"None of the requested providers {providers} are registered or available")

        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Get analysis from all providers
        results = await asyncio.gather(*[
            self.router.route(
                TaskType.IMAGE_ANALYSIS,
                image_data,
                provider=provider,
                model="gpt-4-vision-preview" if provider == "openai" else "claude-3-sonnet"
            )
            for provider in available_providers
        ])

        # Combine results using confidence cascade
        combined_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.CONFIDENCE_CASCADE,
            task_type=TaskType.IMAGE_ANALYSIS,
            confidence_threshold=0.7
        )

        return combined_result

    async def run_qa_ensemble(
        self,
        question: str,
        context: str,
        providers: List[str] = ["openai", "anthropic", "ollama"]
    ) -> Dict[str, Any]:
        """Run question answering ensemble with multiple providers."""
        logger.info(f"Running QA ensemble for question: {question}")

        # Filter providers to only those that are registered
        available_providers = [p for p in providers if p in self.router.providers]
        if not available_providers:
            raise ValueError(f"None of the requested providers {providers} are registered or available")

        # Prepare prompt with context
        prompt = f"""Context: {context}

Question: {question}

Please provide a detailed answer based on the context above."""

        # Get answers from all providers
        results = await asyncio.gather(*[
            self.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
            )
            for provider in available_providers
        ])

        # Combine results using parallel voting with LLM evaluation
        combined_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.PARALLEL_VOTING,
            task_type=TaskType.TEXT_GENERATION
        )

        return combined_result

    async def run_code_review_ensemble(
        self,
        code: str,
        providers: List[str] = ["openai", "anthropic", "ollama"]
    ) -> Dict[str, Any]:
        """Run code review ensemble with multiple providers."""
        logger.info("Running code review ensemble")

        # Filter providers to only those that are registered
        available_providers = [p for p in providers if p in self.router.providers]
        if not available_providers:
            raise ValueError(f"None of the requested providers {providers} are registered or available")

        # Prepare code review prompt
        prompt = f"""Please review the following code and provide feedback on:
1. Code quality
2. Potential bugs
3. Security issues
4. Performance improvements
5. Best practices

Code:
{code}"""

        # Get reviews from all providers
        results = await asyncio.gather(*[
            self.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
            )
            for provider in available_providers
        ])

        # Combine results using rank-based selection
        combined_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.RANK_BASED,
            task_type=TaskType.TEXT_GENERATION
        )

        return combined_result

async def main():
    """Run all ensemble examples."""
    examples = EnsembleExamples()

    # Check if any providers are registered
    if not examples.router.providers:
        raise ValueError(
            "No providers are registered. Please set at least one of the following environment variables:\n"
            "- OPENAI_API_KEY (for OpenAI)\n"
            "- ANTHROPIC_API_KEY or CLAUDE_API_KEY (for Anthropic/Claude)\n"
            "Note: Ollama is registered by default but requires Ollama to be running locally."
        )

    logger.info(f"Available providers: {list(examples.router.providers.keys())}")

    # 1. Text Generation Example
    text_result = await examples.run_text_generation_ensemble(
        "Explain the concept of ensemble learning in machine learning."
    )
    print("\nText Generation Results:")
    # Convert EnsembleResult objects to dictionaries for JSON serialization
    text_result_dict = serialize_ensemble_result(text_result)
    print(json.dumps(text_result_dict, indent=2, default=str))

    # 2. Embedding Example
    try:
        embedding_result = await examples.run_embedding_ensemble(
            "This is a sample text for embedding generation."
        )
        print("\nEmbedding Results:")
        embedding_result_dict = serialize_ensemble_result(embedding_result)
        print(json.dumps(embedding_result_dict, indent=2, default=str))
    except Exception as e:
        logger.error(f"Embedding example failed: {str(e)}")
        print(f"\nEmbedding Results: Skipped due to error - {str(e)}")

    # 3. QA Example
    qa_result = await examples.run_qa_ensemble(
        question="What is the capital of France?",
        context="Paris is the capital and largest city of France. It is known for its iconic Eiffel Tower and rich cultural heritage."
    )
    print("\nQA Results:")
    qa_result_dict = serialize_ensemble_result(qa_result)
    print(json.dumps(qa_result_dict, indent=2, default=str))

    # 4. Code Review Example
    code = """
    def calculate_factorial(n):
        if n < 0:
            return None
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    """
    code_review_result = await examples.run_code_review_ensemble(code)
    print("\nCode Review Results:")
    code_review_result_dict = serialize_ensemble_result(code_review_result)
    print(json.dumps(code_review_result_dict, indent=2, default=str))

    # 5. Image Analysis Example (if image path is provided)
    image_path = "path/to/your/image.jpg"  # Replace with actual image path
    if Path(image_path).exists():
        image_result = await examples.run_image_analysis_ensemble(image_path)
        print("\nImage Analysis Results:")
        image_result_dict = serialize_ensemble_result(image_result)
        print(json.dumps(image_result_dict, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
