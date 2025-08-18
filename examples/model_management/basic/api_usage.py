"""
API usage example of the MultiModelWrapper.
"""

import asyncio
import aiohttp
import json
from multimind import ModelFactory, MultiModelWrapper

async def run_api_examples():
    base_url = "http://localhost:8000"  # Default FastAPI port

    async with aiohttp.ClientSession() as session:
        # Example 1: Generate text
        print("Example 1: Generate text")
        generate_data = {
            "prompt": "Explain quantum computing in simple terms.",
            "primary_model": "openai",
            "fallback_models": ["claude"],
            "model_weights": {"openai": 0.7, "claude": 0.3},
            "temperature": 0.7,
            "max_tokens": 150
        }
        async with session.post(f"{base_url}/generate", json=generate_data) as response:
            result = await response.json()
            print("Generated text:", result["response"])

        # Example 2: Chat completion
        print("\nExample 2: Chat completion")
        chat_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What are the benefits of using multiple AI models?"}
            ],
            "primary_model": "openai",
            "fallback_models": ["claude"],
            "temperature": 0.7
        }
        async with session.post(f"{base_url}/chat", json=chat_data) as response:
            result = await response.json()
            print("Chat response:", result["response"])

        # Example 3: Generate embeddings
        print("\nExample 3: Generate embeddings")
        embeddings_data = {
            "text": "This is a test sentence for embeddings.",
            "primary_model": "openai",
            "fallback_models": ["claude"]
        }
        async with session.post(f"{base_url}/embeddings", json=embeddings_data) as response:
            result = await response.json()
            print("Embeddings shape:", len(result["embeddings"]))

        # Example 4: Health check
        print("\nExample 4: Health check")
        async with session.get(f"{base_url}/health") as response:
            result = await response.json()
            print("Health status:", result["status"])

if __name__ == "__main__":
    asyncio.run(run_api_examples()) 