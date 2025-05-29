"""
Example demonstrating chat completion using the Multimind SDK.
"""

import asyncio
import os
from multimind.core.router import ModelRouter
from multimind.models.openai import OpenAIModel

async def main():
    # Initialize the model router
    router = ModelRouter()
    
    # Create and register an OpenAI model
    openai_model = OpenAIModel(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    router.register_model("gpt-3.5-turbo", openai_model)
    
    # Set up a fallback chain
    router.set_fallback_chain(["gpt-3.5-turbo"])
    
    # Example chat messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Generate a chat completion
    response = await router.chat(messages)
    print(f"Assistant: {response}")
    
    # Example of streaming chat completion
    print("\nStreaming response:")
    async for chunk in openai_model.chat_stream(messages):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main()) 