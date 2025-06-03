"""
API examples for different pipeline types using the MultiMind SDK.
"""

from multimind import Router, PipelineBuilder
import asyncio
import json
from typing import Dict, Any, List

async def run_qa_retrieval_pipeline(query: str) -> Dict[str, Any]:
    """Run a QA retrieval pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.qa_retrieval()
    
    result = await pipeline.run(query)
    return result

async def run_code_review_pipeline(code: str) -> Dict[str, Any]:
    """Run a code review pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.code_review()
    
    result = await pipeline.run(code)
    return result

async def run_image_analysis_pipeline(image_path: str) -> Dict[str, Any]:
    """Run an image analysis pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.image_analysis()
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    result = await pipeline.run(image_data)
    return result

async def run_text_summarization_pipeline(text: str) -> Dict[str, Any]:
    """Run a text summarization pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.text_summarization()
    
    result = await pipeline.run(text)
    return result

async def run_content_generation_pipeline(topic: str) -> Dict[str, Any]:
    """Run a content generation pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.content_generation()
    
    result = await pipeline.run(topic)
    return result

async def run_data_analysis_pipeline(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run a data analysis pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.data_analysis()
    
    result = await pipeline.run(data)
    return result

async def run_multi_modal_qa_pipeline(query: str, image_path: str) -> Dict[str, Any]:
    """Run a multi-modal QA pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.multi_modal_qa()
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    input_data = {
        "query": query,
        "image": image_data
    }
    
    result = await pipeline.run(input_data)
    return result

async def run_code_generation_pipeline(requirements: str) -> Dict[str, Any]:
    """Run a code generation pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.code_generation()
    
    result = await pipeline.run(requirements)
    return result

async def run_sentiment_analysis_pipeline(text: str) -> Dict[str, Any]:
    """Run a sentiment analysis pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.sentiment_analysis()
    
    result = await pipeline.run(text)
    return result

async def run_document_processing_pipeline(document: str) -> Dict[str, Any]:
    """Run a document processing pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.document_processing()
    
    result = await pipeline.run(document)
    return result

async def run_translation_pipeline(text: str, target_language: str) -> Dict[str, Any]:
    """Run a translation pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.translation_pipeline()
    
    input_data = {
        "text": text,
        "target_language": target_language
    }
    
    result = await pipeline.run(input_data)
    return result

async def run_research_assistant_pipeline(query: str) -> Dict[str, Any]:
    """Run a research assistant pipeline example."""
    router = Router()
    builder = PipelineBuilder(router)
    pipeline = builder.research_assistant()
    
    result = await pipeline.run(query)
    return result

# Example usage
async def main():
    # QA Retrieval Example
    qa_result = await run_qa_retrieval_pipeline("What is machine learning?")
    print("QA Retrieval Result:", json.dumps(qa_result, indent=2))
    
    # Code Review Example
    code = """
    def calculate_sum(a, b):
        return a + b
    """
    code_review_result = await run_code_review_pipeline(code)
    print("Code Review Result:", json.dumps(code_review_result, indent=2))
    
    # Text Summarization Example
    text = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data."
    summary_result = await run_text_summarization_pipeline(text)
    print("Text Summarization Result:", json.dumps(summary_result, indent=2))
    
    # Content Generation Example
    content_result = await run_content_generation_pipeline("The Future of AI")
    print("Content Generation Result:", json.dumps(content_result, indent=2))
    
    # Sentiment Analysis Example
    sentiment_result = await run_sentiment_analysis_pipeline("I absolutely love this product! It's amazing.")
    print("Sentiment Analysis Result:", json.dumps(sentiment_result, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 