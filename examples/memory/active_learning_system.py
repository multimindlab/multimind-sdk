"""
Example of an active learning system using ActiveLearningMemory.
This example demonstrates how to use MultiMind's active learning capabilities
to improve responses based on user feedback.
"""

import asyncio
from typing import Dict, Any
from multimind import MultiMind
from multimind.memory import ActiveLearningMemory
from multimind.models import OllamaLLM

async def main():
    # Initialize the LLM
    llm = OllamaLLM(model="mistral")
    
    # Initialize ActiveLearningMemory
    memory = ActiveLearningMemory(
        llm=llm,
        max_items=1000,
        max_feedback=100,
        feedback_threshold=0.7,
        enable_reinforcement=True,
        reinforcement_interval=3600,  # 1 hour
        enable_analysis=True,
        analysis_interval=3600,  # 1 hour
        storage_path="active_learning_memory.json"
    )
    
    # Initialize MultiMind with active learning memory
    mm = MultiMind(
        llm=llm,
        memory=memory,
        system_prompt="You are an AI assistant that learns from user feedback."
    )
    
    # Example learning scenarios
    scenarios = [
        {
            "query": "What is machine learning?",
            "feedback": "The explanation was too technical. Please simplify.",
            "expected_improvement": "Use simpler language and examples"
        },
        {
            "query": "How does a neural network work?",
            "feedback": "Good explanation, but could use more real-world examples.",
            "expected_improvement": "Add practical examples"
        },
        {
            "query": "What are the applications of AI?",
            "feedback": "The response was too brief. Need more details.",
            "expected_improvement": "Provide more comprehensive coverage"
        }
    ]
    
    # Simulate learning process
    for scenario in scenarios:
        print(f"\nQuery: {scenario['query']}")
        response = await mm.chat(scenario['query'])
        print(f"Initial Response: {response}")
        
        # Simulate user feedback
        print(f"\nUser Feedback: {scenario['feedback']}")
        memory.track_feedback(
            query=scenario['query'],
            response=response,
            feedback=scenario['feedback'],
            expected_improvement=scenario['expected_improvement']
        )
        
        # Get learning statistics
        stats = memory.get_active_learning_stats()
        print("\nLearning Statistics:")
        print(f"Total items: {stats['total_items']}")
        print(f"Feedback count: {stats['feedback_count']}")
        print(f"Reinforcement data: {stats['reinforcement_data']}")
        
        # Get learning suggestions
        suggestions = memory.get_active_learning_suggestions()
        if suggestions:
            print("\nLearning Suggestions:")
            for suggestion in suggestions:
                print(f"- {suggestion}")
        
        # Analyze feedback patterns
        patterns = memory.analyze_feedback_patterns()
        if patterns:
            print("\nFeedback Patterns:")
            for pattern in patterns:
                print(f"- {pattern}")

if __name__ == "__main__":
    asyncio.run(main()) 