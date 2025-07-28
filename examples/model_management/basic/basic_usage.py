"""
Basic usage example of the MultiModelWrapper.
"""

import asyncio
from multimind import ModelFactory, MultiModelWrapper

async def main():
    # Initialize the model factory
    factory = ModelFactory()

    # Create a multi-model wrapper with OpenAI as primary and Claude as fallback
    multi_model = MultiModelWrapper(
        model_factory=factory,
        primary_model="openai",
        fallback_models=["claude"],
        model_weights={
            "openai": 0.7,  # Higher weight for OpenAI
            "claude": 0.3   # Lower weight for Claude
        }
    )

    # Example 1: Simple text generation
    prompt = "Explain quantum computing in simple terms."
    response = await multi_model.generate(
        prompt=prompt,
        temperature=0.7,
        max_tokens=150
    )
    print("Generated text:", response)

    # Example 2: Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are the benefits of using multiple AI models?"}
    ]
    chat_response = await multi_model.chat(
        messages=messages,
        temperature=0.7
    )
    print("\nChat response:", chat_response)

    # Example 3: Streaming response
    print("\nStreaming response:")
    async for chunk in multi_model.generate_stream(
        prompt="Write a short poem about AI:",
        temperature=0.8
    ):
        print(chunk, end="", flush=True)
    print()

    # Example 4: Embeddings
    text = "This is a test sentence for embeddings."
    embeddings = await multi_model.embeddings(text)
    print("\nEmbeddings shape:", len(embeddings))

if __name__ == "__main__":
    asyncio.run(main()) 