"""
Example of an event tracking system using EventSourcedMemory.
This example demonstrates how to use MultiMind's event sourcing capabilities
to track and analyze sequences of events.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import EventSourcedMemory
from multimind.models import OllamaLLM

async def main():
    # Initialize the LLM
    llm = OllamaLLM(model="mistral")
    
    # Initialize EventSourcedMemory
    memory = EventSourcedMemory(
        llm=llm,
        max_events=1000,
        max_snapshots=100,
        snapshot_interval=3600,  # 1 hour
        enable_causality=True,
        enable_patterns=True,
        pattern_interval=3600,  # 1 hour
        storage_path="event_sourced.json"
    )
    
    # Initialize MultiMind with event sourced memory
    mm = MultiMind(
        llm=llm,
        memory=memory,
        system_prompt="You are an event tracking system that analyzes sequences of events."
    )
    
    # Example event sequence (user interaction flow)
    events = [
        {
            "type": "user_action",
            "action": "login",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"user_id": "user1", "device": "web"}
        },
        {
            "type": "user_action",
            "action": "search",
            "query": "machine learning",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"user_id": "user1", "device": "web"}
        },
        {
            "type": "system_action",
            "action": "display_results",
            "results_count": 10,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"user_id": "user1", "device": "web"}
        },
        {
            "type": "user_action",
            "action": "click_result",
            "result_id": "result1",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"user_id": "user1", "device": "web"}
        },
        {
            "type": "user_action",
            "action": "logout",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"user_id": "user1", "device": "web"}
        }
    ]
    
    # Add events to memory
    for event in events:
        memory.add_event(event)
    
    # Example queries
    queries = [
        "What was the sequence of user actions?",
        "What patterns can you identify in the user behavior?",
        "What was the cause of the user's search action?"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        response = await mm.chat(query)
        print(f"Response: {response}")
        
        # Get event statistics
        stats = memory.get_event_stats()
        print("\nEvent Statistics:")
        print(f"Total events: {stats['total_events']}")
        print(f"Event types: {stats['event_types']}")
        print(f"Time span: {stats['time_span']}")
        
        # Get event patterns
        patterns = memory.get_event_patterns()
        if patterns:
            print("\nEvent Patterns:")
            for pattern in patterns:
                print(f"- {pattern}")
        
        # Get causality analysis
        causality = memory.get_causality_analysis()
        if causality:
            print("\nCausality Analysis:")
            for cause_effect in causality:
                print(f"- {cause_effect}")
        
        # Get event timeline
        timeline = memory.get_event_timeline()
        if timeline:
            print("\nEvent Timeline:")
            for event in timeline:
                print(f"- {event['timestamp']}: {event['type']} - {event['action']}")

if __name__ == "__main__":
    asyncio.run(main()) 