"""
Basic agent example demonstrating how to create and use agents with different models.
Includes local HuggingFace model for testing without API keys.
"""

import asyncio
import os

from dotenv import load_dotenv

from multimind import Agent, AgentMemory, CalculatorTool, ClaudeModel, MistralModel, OpenAIModel

# Try to import HuggingFaceModel
try:
    from multimind import HuggingFaceModel
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceModel = None

async def main():
    # Load environment variables
    load_dotenv()

    # Create different model instances
    models = {}
    agents = {}

    # OpenAI Model (requires API key)
    if os.getenv("OPENAI_API_KEY"):
        models["openai"] = OpenAIModel(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )

    # Claude Model (requires API key)
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"):
        models["claude"] = ClaudeModel(
            model_name="claude-3-sonnet-20240229",
            temperature=0.7
        )

    # Mistral Model (requires API key or Ollama)
    if os.getenv("MISTRAL_API_KEY"):
        models["mistral"] = MistralModel(
            model_name="mistral-medium",
            temperature=0.7
        )

    # HuggingFace Model (local, no API key required!)
    if HUGGINGFACE_AVAILABLE:
        print("Loading local HuggingFace model (no API key required)...")
        try:
            models["huggingface"] = HuggingFaceModel(
                model_name="gpt2",  # Small model, good for testing
                api_key=None,  # No API key needed for public models
                temperature=0.7
            )
            print("✓ HuggingFace model loaded successfully!")
        except Exception as e:
            print(f"⚠ Could not load HuggingFace model: {e}")
            print("   Install with: pip install transformers torch")

    if not models:
        print("No models available! Please set at least one API key or install transformers for local HuggingFace.")
        print("\nFor local testing without API keys, install:")
        print("  pip install transformers torch")
        return

    # Create memory and tools
    memory = AgentMemory(max_history=50)
    calculator = CalculatorTool()

    # Create agents with available models
    for model_name, model in models.items():
        agents[model_name] = Agent(
            model=model,
            memory=AgentMemory(max_history=50),  # Separate memory per agent
            tools=[calculator],
            system_prompt="You are a helpful AI assistant that can perform calculations."
        )

    # Example tasks
    tasks = [
        "What is 123 * 456?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming"
    ]

    # Run tasks with available agents
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print('='*60)

        for agent_name, agent in agents.items():
            print(f"\n{agent_name.upper()} Agent:")
            try:
                response = await agent.run(task)
                # Response is a dict with 'result' or 'response' key
                if isinstance(response, dict):
                    result = response.get('result', response.get('response', response))
                    print(f"Response: {result}")
                else:
                    print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")

        # Get memory from HuggingFace agent if available
        if "huggingface" in agents:
            hf_memory = agents["huggingface"].memory
            history = hf_memory.get_history(n=1)
            if history:
                print(f"\nLast interaction in HuggingFace agent memory: {history[0]}")

if __name__ == "__main__":
    asyncio.run(main())
