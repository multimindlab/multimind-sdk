"""
Example: Using HuggingFace models locally without API keys

This example demonstrates how to use HuggingFace models locally
without requiring an API key. The model will be downloaded and
cached on first use.

Requirements:
    pip install transformers torch

Note: You can use any public HuggingFace model. Recommended models for testing:
    - gpt2 (124M params, very fast)
    - distilgpt2 (82M params, fastest)
    - microsoft/DialoGPT-small (117M params, good for conversations)
"""

import asyncio
import os
import sys

# Add parent directory to path to import multimind
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from multimind.gateway.models import HuggingFaceHandler
from multimind.core.config import ModelConfig


async def main():
    """Example of using local HuggingFace model"""
    
    print("=" * 60)
    print("Local HuggingFace Model Example")
    print("=" * 60)
    print()
    
    # Option 1: Use with default config (no API key needed)
    print("Option 1: Using default GPT-2 model (no API key)")
    print("-" * 60)
    
    config = ModelConfig(
        api_key=None,  # No API key = use local model
        model_name="gpt2",  # Small model, good for testing
        temperature=0.7,
        max_tokens=100
    )
    
    handler = HuggingFaceHandler(config)
    
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: {prompt}")
    
    response = await handler.generate(prompt)
    print(f"Response: {response.content}")
    print()
    
    # Option 2: Chat interface
    print("Option 2: Using chat interface")
    print("-" * 60)
    
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    chat_response = await handler.chat(messages)
    print(f"User: {messages[0]['content']}")
    print(f"Assistant: {chat_response.content}")
    print()
    
    # Option 3: Use a different model
    print("Option 3: Using DistilGPT-2 (even smaller model)")
    print("-" * 60)
    
    config2 = ModelConfig(
        api_key=None,
        model_name="distilgpt2",  # Smaller than GPT-2
        temperature=0.8,
        max_tokens=80
    )
    
    handler2 = HuggingFaceHandler(config2)
    response2 = await handler2.generate("Python is")
    print(f"Prompt: Python is")
    print(f"Response: {response2.content}")
    print()
    
    print("=" * 60)
    print("Note: Models are cached in ~/.cache/huggingface/")
    print("First run will download the model, subsequent runs are faster.")
    print("=" * 60)


if __name__ == "__main__":
    # Check if transformers is available
    try:
        import transformers
        import torch
        print(f"✓ transformers version: {transformers.__version__}")
        print(f"✓ torch version: {torch.__version__}")
        print()
    except ImportError:
        print("ERROR: transformers and torch are required!")
        print("Install with: pip install transformers torch")
        sys.exit(1)
    
    # Run the example
    asyncio.run(main())


