"""
Software Development CLI Example
This example demonstrates how to use MultiMind's memory capabilities for software development applications,
combining knowledge graph memory for code knowledge and event-sourced memory for development tracking.
"""

import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import (
    KnowledgeGraphMemory,
    EventSourcedMemory,
    HybridMemory
)
from multimind.models import OllamaLLM

class SoftwareDevelopmentCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "software_development.json"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize knowledge graph memory for code knowledge
        self.code_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_knowledge.json"
        )
        
        # Initialize event-sourced memory for development tracking
        self.dev_tracking = EventSourcedMemory(
            llm=self.llm,
            max_events=10000,
            max_snapshots=1000,
            snapshot_interval=3600,  # 1 hour
            enable_causality=True,
            enable_patterns=True,
            pattern_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_events.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.code_knowledge,
                self.dev_tracking
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path=storage_path
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a software development assistant that helps with code knowledge management and development tracking.
            You maintain a knowledge graph of code information and track development events and patterns.
            Always emphasize that you are an AI assistant and not a replacement for software development expertise."""
        )

    async def process_command(self, command: str, args: List[str]) -> None:
        """Process CLI commands."""
        if command == "/add_knowledge":
            if len(args) < 4:
                print("Usage: /add_knowledge <subject> <predicate> <object> <confidence>")
                return
            subject, predicate, object_, confidence = args[0], args[1], args[2], float(args[3])
            await self.add_code_knowledge(subject, predicate, object_, confidence)
            print(f"Added knowledge: {subject} {predicate} {object_} (confidence: {confidence})")
        
        elif command == "/add_event":
            if len(args) < 3:
                print("Usage: /add_event <type> <description> <metadata>")
                return
            event_type, description = args[0], args[1]
            metadata = eval(args[2])  # Convert string to dict
            await self.add_dev_event(event_type, description, metadata)
            print(f"Added event: {event_type} - {description}")
        
        elif command == "/query":
            if not args:
                print("Usage: /query <query_text>")
                return
            query = " ".join(args)
            await self.query_knowledge(query)
        
        elif command == "/analyze":
            if not args:
                print("Usage: /analyze <project_id>")
                return
            project_id = args[0]
            await self.analyze_development(project_id)
        
        elif command == "/stats":
            await self.show_stats()
        
        elif command == "/export":
            if len(args) < 1:
                print("Usage: /export <filename>")
                return
            filename = args[0]
            await self.export_data(filename)
        
        elif command == "/import":
            if len(args) < 1:
                print("Usage: /import <filename>")
                return
            filename = args[0]
            await self.import_data(filename)
        
        elif command == "/clear":
            await self.clear_data()
        
        elif command == "/help":
            self.show_help()
        
        else:
            print(f"Unknown command: {command}")
            self.show_help()

    async def add_code_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add code knowledge to the knowledge graph."""
        self.code_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def add_dev_event(self, event_type: str, description: str, metadata: Dict[str, Any]) -> None:
        """Add a development event."""
        self.dev_tracking.add_event(
            event_type=event_type,
            description=description,
            metadata=metadata
        )

    async def query_knowledge(self, query: str) -> None:
        """Query code knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.code_knowledge.get_related_concepts(query)
        inferences = self.code_knowledge.get_inferences(query)
        
        print(f"\nResponse: {response}")
        print("\nRelated Concepts:")
        for concept in related:
            print(f"- {concept}")
        print("\nInferences:")
        for inference in inferences:
            print(f"- {inference}")

    async def analyze_development(self, project_id: str) -> None:
        """Analyze development data and patterns."""
        patterns = self.dev_tracking.get_event_patterns(project_id)
        causality = self.dev_tracking.get_causality_analysis(project_id)
        timeline = self.dev_tracking.get_event_timeline()
        
        print("\nDevelopment Patterns:")
        for pattern in patterns:
            print(f"- {pattern}")
        
        print("\nDependency Analysis:")
        for cause in causality:
            print(f"- {cause}")
        
        print("\nTimeline:")
        for event in timeline:
            print(f"- {event}")

    async def show_stats(self) -> None:
        """Show development and knowledge statistics."""
        event_stats = self.dev_tracking.get_event_stats()
        knowledge_stats = self.code_knowledge.get_graph_stats()
        
        print("\nDevelopment Statistics:")
        for key, value in event_stats.items():
            print(f"{key}: {value}")
        
        print("\nKnowledge Graph Statistics:")
        for key, value in knowledge_stats.items():
            print(f"{key}: {value}")

    async def export_data(self, filename: str) -> None:
        """Export development and knowledge data."""
        # Implementation would depend on the storage mechanism
        print(f"Exporting data to {filename}...")

    async def import_data(self, filename: str) -> None:
        """Import development and knowledge data."""
        # Implementation would depend on the storage mechanism
        print(f"Importing data from {filename}...")

    async def clear_data(self) -> None:
        """Clear all development and knowledge data."""
        self.dev_tracking.clear()
        self.code_knowledge.clear()
        print("All data cleared.")

    def show_help(self) -> None:
        """Show available commands and their usage."""
        print("\nAvailable Commands:")
        print("/add_knowledge <subject> <predicate> <object> <confidence> - Add code knowledge")
        print("/add_event <type> <description> <metadata> - Add development event")
        print("/query <query_text> - Query code knowledge")
        print("/analyze <project_id> - Analyze development data")
        print("/stats - Show statistics")
        print("/export <filename> - Export data")
        print("/import <filename> - Import data")
        print("/clear - Clear all data")
        print("/help - Show this help message")
        print("/exit - Exit program")

async def main():
    parser = argparse.ArgumentParser(description="Software Development CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="software_development.json", help="Storage path")
    args = parser.parse_args()
    
    cli = SoftwareDevelopmentCLI(model=args.model, storage_path=args.storage)
    
    print("Software Development CLI")
    print("Type /help for available commands")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() == "/exit":
                break
            
            if user_input.startswith("/"):
                command = user_input.split()[0]
                args = user_input.split()[1:]
                await cli.process_command(command, args)
            else:
                print("Unknown command. Type /help for available commands.")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 