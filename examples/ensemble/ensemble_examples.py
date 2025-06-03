"""
Comprehensive examples of using the MultiMind Ensemble system.
"""

import asyncio
import json
from typing import Dict, List, Any
from pathlib import Path
import logging

from multimind import Router, TaskType
from multimind.ensemble import AdvancedEnsemble, EnsembleMethod
from multimind.core.provider import GenerationResult, EmbeddingResult, ImageAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleExamples:
    def __init__(self):
        """Initialize the ensemble examples."""
        self.router = Router()
        self.ensemble = AdvancedEnsemble(self.router)
        
    async def run_text_generation_ensemble(
        self,
        prompt: str,
        providers: List[str] = ["openai", "anthropic", "ollama"]
    ) -> Dict[str, Any]:
        """Run text generation ensemble with multiple providers."""
        logger.info(f"Running text generation ensemble for prompt: {prompt}")
        
        # Get results from all providers
        results = await asyncio.gather(*[
            self.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=provider,
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "mistral"
            )
            for provider in providers
        ])
        
        # Try different ensemble methods
        ensemble_results = {}
        
        # 1. Weighted Voting
        weighted_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.TEXT_GENERATION,
            weights={
                "openai": 0.4,
                "anthropic": 0.4,
                "ollama": 0.2
            }
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
        providers: List[str] = ["openai", "huggingface"]
    ) -> Dict[str, Any]:
        """Run embedding ensemble with multiple providers."""
        logger.info(f"Running embedding ensemble for text: {text[:100]}...")
        
        # Get embeddings from all providers
        results = await asyncio.gather(*[
            self.router.route(
                TaskType.EMBEDDINGS,
                text,
                provider=provider,
                model="text-embedding-ada-002" if provider == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
            )
            for provider in providers
        ])
        
        # Combine embeddings using weighted voting
        combined_result = await self.ensemble.combine_results(
            results=results,
            method=EnsembleMethod.WEIGHTED_VOTING,
            task_type=TaskType.EMBEDDINGS,
            weights={
                "openai": 0.6,
                "huggingface": 0.4
            }
        )
        
        return combined_result
    
    async def run_image_analysis_ensemble(
        self,
        image_path: str,
        providers: List[str] = ["openai", "anthropic"]
    ) -> Dict[str, Any]:
        """Run image analysis ensemble with multiple providers."""
        logger.info(f"Running image analysis ensemble for image: {image_path}")
        
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
            for provider in providers
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
            for provider in providers
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
                model="gpt-4" if provider == "openai" else "claude-3-sonnet" if provider == "anthropic" else "codellama"
            )
            for provider in providers
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
    
    # 1. Text Generation Example
    text_result = await examples.run_text_generation_ensemble(
        "Explain the concept of ensemble learning in machine learning."
    )
    print("\nText Generation Results:")
    print(json.dumps(text_result, indent=2))
    
    # 2. Embedding Example
    embedding_result = await examples.run_embedding_ensemble(
        "This is a sample text for embedding generation."
    )
    print("\nEmbedding Results:")
    print(json.dumps(embedding_result.dict(), indent=2))
    
    # 3. QA Example
    qa_result = await examples.run_qa_ensemble(
        question="What is the capital of France?",
        context="Paris is the capital and largest city of France. It is known for its iconic Eiffel Tower and rich cultural heritage."
    )
    print("\nQA Results:")
    print(json.dumps(qa_result.dict(), indent=2))
    
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
    print(json.dumps(code_review_result.dict(), indent=2))
    
    # 5. Image Analysis Example (if image path is provided)
    image_path = "path/to/your/image.jpg"  # Replace with actual image path
    if Path(image_path).exists():
        image_result = await examples.run_image_analysis_ensemble(image_path)
        print("\nImage Analysis Results:")
        print(json.dumps(image_result.dict(), indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 