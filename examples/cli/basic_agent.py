"""
Basic agent example demonstrating how to create and use agents with different models.
"""

import asyncio
import os
from dotenv import load_dotenv
from multimind import (
    Agent, AgentMemory, CalculatorTool,
    OpenAIModel, ClaudeModel, MistralModel
)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Create different model instances
    openai_model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    claude_model = ClaudeModel(
        model="claude-3-sonnet-20240229",
        temperature=0.7
    )
    
    mistral_model = MistralModel(
        model="mistral-medium",
        temperature=0.7
    )
    
    # Create memory and tools
    memory = AgentMemory(max_history=50)
    calculator = CalculatorTool()
    
    # Create agents with different models
    openai_agent = Agent(
        model=openai_model,
        memory=memory,
        tools=[calculator],
        system_prompt="You are a helpful AI assistant that can perform calculations."
    )
    
    claude_agent = Agent(
        model=claude_model,
        memory=memory,
        tools=[calculator],
        system_prompt="You are a helpful AI assistant that can perform calculations."
    )
    
    mistral_agent = Agent(
        model=mistral_model,
        memory=memory,
        tools=[calculator],
        system_prompt="You are a helpful AI assistant that can perform calculations."
    )
    
    # Example tasks
    tasks = [
        "What is 123 * 456?",
        "Explain quantum computing in simple terms",
        "Write a haiku about programming"
    ]
    
    # Run tasks with different agents
    for task in tasks:
        print(f"\nTask: {task}")
        
        print("\nOpenAI Agent:")
        response = await openai_agent.run(task)
        print(f"Response: {response}")
        
        print("\nClaude Agent:")
        response = await claude_agent.run(task)
        print(f"Response: {response}")
        
        print("\nMistral Agent:")
        response = await mistral_agent.run(task)
        print(f"Response: {response}")
        
        # Get agent memory
        history = memory.get_history(n=1)
        print(f"\nLast interaction in memory: {history[0]}")

if __name__ == "__main__":
    asyncio.run(main()) 