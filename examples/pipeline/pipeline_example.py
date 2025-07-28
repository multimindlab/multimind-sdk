"""
Example demonstrating how to use the pipeline system.
"""

import asyncio
import os
from typing import List, Dict, Any
from multimind import Router, TaskType, TaskConfig, RoutingStrategy, Pipeline, PipelineBuilder
from multimind.core.provider import ProviderConfig
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
            }
        }
    )
    
    embeddings_config = TaskConfig(
        preferred_providers=["openai"],
        fallback_providers=[],
        routing_strategy=RoutingStrategy.COST_BASED
    )
    
    image_analysis_config = TaskConfig(
        preferred_providers=["openai", "claude"],
        fallback_providers=[],
        routing_strategy=RoutingStrategy.ENSEMBLE
    )
    
    router.configure_task(TaskType.TEXT_GENERATION, text_generation_config)
    router.configure_task(TaskType.EMBEDDINGS, embeddings_config)
    router.configure_task(TaskType.IMAGE_ANALYSIS, image_analysis_config)
    
    builder = PipelineBuilder(router)
    
    # Example 1: Content Generation Pipeline
    print("\nExample 1: Content Generation Pipeline")
    content_pipeline = builder.content_generation()
    result = await content_pipeline.run({
        "topic": "The Future of Artificial Intelligence",
        "target_audience": "technical professionals",
        "length": "1000 words"
    })
    print(f"Generated Content: {result.result}")
    
    # Example 2: Data Analysis Pipeline
    print("\nExample 2: Data Analysis Pipeline")
    data_pipeline = builder.data_analysis()
    result = await data_pipeline.run({
        "data": {
            "sales": [100, 150, 200, 180, 250],
            "months": ["Jan", "Feb", "Mar", "Apr", "May"]
        },
        "analysis_type": "trend_analysis"
    })
    print(f"Analysis Result: {result.result}")
    
    # Example 3: Multi-modal QA Pipeline
    print("\nExample 3: Multi-modal QA Pipeline")
    multi_modal_pipeline = builder.multi_modal_qa()
    # Note: In a real example, you would provide an actual image
    image_data = b"dummy_image_data"  # Replace with actual image data
    result = await multi_modal_pipeline.run({
        "image": image_data,
        "question": "What is happening in this image and how does it relate to quantum computing?"
    })
    print(f"Multi-modal QA Result: {result.result}")
    
    # Example 4: Code Generation Pipeline
    print("\nExample 4: Code Generation Pipeline")
    code_pipeline = builder.code_generation()
    result = await code_pipeline.run({
        "task": "Create a Python function to implement a binary search tree",
        "requirements": ["Must be well-documented", "Include unit tests"]
    })
    print(f"Generated Code: {result.result}")
    
    # Example 5: Sentiment Analysis Pipeline
    print("\nExample 5: Sentiment Analysis Pipeline")
    sentiment_pipeline = builder.sentiment_analysis()
    result = await sentiment_pipeline.run({
        "text": "The new product features are impressive, but the user interface could be more intuitive. Customer support is excellent though.",
        "aspects": ["features", "interface", "support"]
    })
    print(f"Sentiment Analysis Result: {result.result}")
    
    # Example 6: Document Processing Pipeline
    print("\nExample 6: Document Processing Pipeline")
    doc_pipeline = builder.document_processing()
    result = await doc_pipeline.run({
        "document": """
        Apple Inc. announced its new iPhone 15 Pro today. The device features a titanium frame,
        A17 Pro chip, and improved camera system. CEO Tim Cook stated that this represents
        their most significant iPhone upgrade in years. The company also revealed new Apple Watch
        models with enhanced health monitoring capabilities.
        """,
        "extract_entities": True
    })
    print(f"Document Processing Result: {result.result}")
    
    # Example 7: Translation Pipeline
    print("\nExample 7: Translation Pipeline")
    translation_pipeline = builder.translation_pipeline()
    result = await translation_pipeline.run({
        "text": "The quick brown fox jumps over the lazy dog.",
        "target_language": "Spanish",
        "preserve_style": True
    })
    print(f"Translation Result: {result.result}")
    
    # Example 8: Research Assistant Pipeline
    print("\nExample 8: Research Assistant Pipeline")
    research_pipeline = builder.research_assistant()
    result = await research_pipeline.run({
        "topic": "Quantum Computing Applications in Machine Learning",
        "research_scope": "last 2 years",
        "key_aspects": ["algorithms", "hardware", "applications"]
    })
    print(f"Research Analysis Result: {result.result}")

if __name__ == "__main__":
    asyncio.run(main()) 