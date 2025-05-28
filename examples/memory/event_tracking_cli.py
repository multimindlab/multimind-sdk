"""
CLI version of the event tracking system, featuring complex use cases.
This example demonstrates advanced event tracking capabilities in an interactive CLI environment.
"""

import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import EventSourcedMemory
from multimind.models import OllamaLLM

class EventTrackingCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "event_sourced.json"):
        self.llm = OllamaLLM(model=model)
        self.memory = EventSourcedMemory(
            llm=self.llm,
            max_events=1000,
            max_snapshots=100,
            snapshot_interval=3600,  # 1 hour
            enable_causality=True,
            enable_patterns=True,
            pattern_interval=3600,  # 1 hour
            storage_path=storage_path
        )
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="You are an event tracking system that analyzes sequences of events."
        )
        self.current_session = None
        self.current_context = {}

    async def process_command(self, command: str) -> None:
        """Process special commands."""
        if command.startswith("/"):
            cmd = command[1:].lower()
            if cmd == "add":
                await self.add_event()
            elif cmd == "query":
                await self.query_events()
            elif cmd == "stats":
                self.show_event_stats()
            elif cmd == "session":
                self.show_current_session()
            elif cmd == "pattern":
                await self.analyze_patterns()
            elif cmd == "cause":
                await self.analyze_causality()
            elif cmd == "timeline":
                self.show_timeline()
            elif cmd == "export":
                self.export_events()
            elif cmd == "import":
                self.import_events()
            elif cmd == "clear":
                self.clear_events()
            elif cmd == "help":
                self.show_help()
            elif cmd == "exit":
                print("Goodbye!")
                exit(0)
            else:
                print("Unknown command. Type /help for available commands.")
        else:
            await self.process_query(command)

    async def add_event(self) -> None:
        """Add new event to the system."""
        print("\nAdding new event:")
        event_type = input("Event type: ").strip()
        description = input("Description: ").strip()
        metadata = {}
        
        # Add metadata fields
        while True:
            key = input("Metadata key (or empty to finish): ").strip()
            if not key:
                break
            value = input(f"Value for {key}: ").strip()
            metadata[key] = value
        
        self.memory.add_event(
            event_type=event_type,
            description=description,
            metadata=metadata
        )
        print("Event added successfully!")

    async def query_events(self) -> None:
        """Query events in the system."""
        query = input("\nEnter your query: ").strip()
        response = await self.mm.chat(query)
        print(f"\nResponse: {response}")
        
        # Get related events
        related = self.memory.get_related_events(query)
        if related:
            print("\nRelated Events:")
            for event in related:
                print(f"- {event}")

    async def process_query(self, query: str) -> None:
        """Process a natural language query."""
        response = await self.mm.chat(query)
        print(f"\nResponse: {response}")
        
        # Update current session
        self.update_session(query)

    def update_session(self, query: str) -> None:
        """Update current event session."""
        session = self.memory.extract_session(query)
        if session:
            self.current_session = session
            self.current_context["session"] = self.current_session
            self.current_context["last_updated"] = datetime.now().isoformat()

    def show_event_stats(self) -> None:
        """Display event statistics."""
        stats = self.memory.get_event_stats()
        print("\nEvent Statistics:")
        print(f"Total events: {stats['total_events']}")
        print(f"Total sessions: {stats['total_sessions']}")
        print(f"Average events per session: {stats['average_events_per_session']}")
        
        # Show event type distribution
        types = self.memory.get_event_type_distribution()
        if types:
            print("\nEvent Type Distribution:")
            for type_, count in types.items():
                print(f"- {type_}: {count} events")

    def show_current_session(self) -> None:
        """Display current event session."""
        if self.current_session:
            print(f"\nCurrent Session: {self.current_session}")
            events = self.memory.get_session_events(self.current_session)
            if events:
                print("\nSession Events:")
                for event in events:
                    print(f"- {event}")
        else:
            print("\nNo current session identified.")

    async def analyze_patterns(self) -> None:
        """Analyze event patterns."""
        query = input("\nEnter pattern analysis query: ").strip()
        patterns = self.memory.get_event_patterns(query)
        if patterns:
            print("\nEvent Patterns:")
            for pattern in patterns:
                print(f"- {pattern}")
        else:
            print("No patterns found.")

    async def analyze_causality(self) -> None:
        """Analyze event causality."""
        query = input("\nEnter causality analysis query: ").strip()
        causality = self.memory.get_causality_analysis(query)
        if causality:
            print("\nCausality Analysis:")
            for cause in causality:
                print(f"- {cause}")
        else:
            print("No causality found.")

    def show_timeline(self) -> None:
        """Display event timeline."""
        timeline = self.memory.get_event_timeline()
        if timeline:
            print("\nEvent Timeline:")
            for event in timeline:
                print(f"- {event}")
        else:
            print("No events in timeline.")

    def export_events(self) -> None:
        """Export events to file."""
        filename = input("\nEnter export filename: ").strip()
        self.memory.export_events(filename)
        print(f"Events exported to {filename}")

    def import_events(self) -> None:
        """Import events from file."""
        filename = input("\nEnter import filename: ").strip()
        self.memory.import_events(filename)
        print(f"Events imported from {filename}")

    def clear_events(self) -> None:
        """Clear all events."""
        self.memory.clear()
        self.current_session = None
        self.current_context = {}
        print("\nEvents cleared.")

    def show_help(self) -> None:
        """Display available commands."""
        print("\nAvailable Commands:")
        print("/add - Add new event")
        print("/query - Query events")
        print("/stats - Show event statistics")
        print("/session - Show current session")
        print("/pattern - Analyze event patterns")
        print("/cause - Analyze event causality")
        print("/timeline - Show event timeline")
        print("/export - Export events")
        print("/import - Import events")
        print("/clear - Clear all events")
        print("/help - Show this help message")
        print("/exit - Exit the program")

async def main():
    parser = argparse.ArgumentParser(description="MultiMind Event Tracking CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="event_sourced.json", help="Event storage path")
    args = parser.parse_args()

    et = EventTrackingCLI(model=args.model, storage_path=args.storage)
    
    print("Welcome to MultiMind Event Tracking CLI!")
    print("Type /help for available commands.")
    print("Type /exit to quit.")
    
    while True:
        try:
            command = input("\nCommand: ").strip()
            if command:
                await et.process_command(command)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 