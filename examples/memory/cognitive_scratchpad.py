"""
Example of a cognitive scratchpad system using CognitiveScratchpadMemory.
This example demonstrates how to use MultiMind's cognitive scratchpad capabilities
for step-by-step reasoning and problem-solving.
"""

import asyncio
from typing import Dict, Any, List
from multimind import MultiMind
from multimind.memory import CognitiveScratchpadMemory
from multimind.models import OllamaLLM

async def main():
    # Initialize the LLM
    llm = OllamaLLM(model="mistral")
    
    # Initialize CognitiveScratchpadMemory
    memory = CognitiveScratchpadMemory(
        llm=llm,
        max_steps=100,
        max_chains=10,
        chain_depth=5,
        enable_analysis=True,
        analysis_interval=3600,  # 1 hour
        storage_path="cognitive_scratchpad.json"
    )
    
    # Initialize MultiMind with cognitive scratchpad memory
    mm = MultiMind(
        llm=llm,
        memory=memory,
        system_prompt="You are a problem-solving system that uses step-by-step reasoning."
    )
    
    # Example problem-solving scenarios
    scenarios = [
        {
            "problem": "Solve the equation: 2x + 5 = 13",
            "expected_steps": [
                "Isolate the variable term",
                "Subtract 5 from both sides",
                "Divide both sides by 2",
                "Verify the solution"
            ]
        },
        {
            "problem": "Analyze the impact of climate change on agriculture",
            "expected_steps": [
                "Identify key climate factors",
                "Analyze effects on crop yields",
                "Consider adaptation strategies",
                "Evaluate economic impact"
            ]
        },
        {
            "problem": "Design a simple machine learning model for image classification",
            "expected_steps": [
                "Choose a model architecture",
                "Prepare the dataset",
                "Define the training process",
                "Implement evaluation metrics"
            ]
        }
    ]
    
    # Process each scenario
    for scenario in scenarios:
        print(f"\nProblem: {scenario['problem']}")
        
        # Start reasoning process
        memory.start_reasoning_chain(scenario['problem'])
        
        # Process each expected step
        for step in scenario['expected_steps']:
            print(f"\nStep: {step}")
            response = await mm.chat(f"Let's {step.lower()}")
            print(f"Reasoning: {response}")
            
            # Add step to memory
            memory.add_reasoning_step(
                step=step,
                reasoning=response,
                confidence=0.8
            )
        
        # Complete the reasoning chain
        memory.complete_reasoning_chain()
        
        # Get reasoning statistics
        stats = memory.get_reasoning_stats()
        print("\nReasoning Statistics:")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Average confidence: {stats['average_confidence']}")
        print(f"Chain depth: {stats['chain_depth']}")
        
        # Get reasoning analysis
        analysis = memory.get_reasoning_analysis()
        if analysis:
            print("\nReasoning Analysis:")
            for insight in analysis:
                print(f"- {insight}")
        
        # Get step dependencies
        dependencies = memory.get_step_dependencies()
        if dependencies:
            print("\nStep Dependencies:")
            for dep in dependencies:
                print(f"- {dep}")
        
        # Get reasoning suggestions
        suggestions = memory.get_reasoning_suggestions()
        if suggestions:
            print("\nReasoning Suggestions:")
            for suggestion in suggestions:
                print(f"- {suggestion}")

if __name__ == "__main__":
    asyncio.run(main()) 