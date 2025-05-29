"""
Scientific Research CLI Example
This example demonstrates how to use MultiMind's memory capabilities for scientific research applications,
combining knowledge graph memory for research knowledge and event-sourced memory for experiment tracking.
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

class ScientificResearchCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "scientific_research.json"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize knowledge graph memory for research knowledge
        self.research_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_knowledge.json"
        )
        
        # Initialize event-sourced memory for experiment tracking
        self.experiment_tracking = EventSourcedMemory(
            llm=self.llm,
            max_events=10000,
            max_snapshots=1000,
            snapshot_interval=3600,  # 1 hour
            enable_causality=True,
            enable_patterns=True,
            pattern_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_experiments.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.research_knowledge,
                self.experiment_tracking
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path=storage_path
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a scientific research assistant that helps with research knowledge management and experiment tracking.
            You maintain a knowledge graph of research concepts and track experimental procedures and results.
            Always emphasize that you are an AI assistant and not a replacement for scientific expertise."""
        )

    async def process_command(self, command: str, args: List[str]) -> None:
        """Process CLI commands."""
        if command == "/add_knowledge":
            if len(args) < 4:
                print("Usage: /add_knowledge <subject> <predicate> <object> <confidence>")
                return
            subject, predicate, object_, confidence = args[0], args[1], args[2], float(args[3])
            await self.add_research_knowledge(subject, predicate, object_, confidence)
            print(f"Added knowledge: {subject} {predicate} {object_} (confidence: {confidence})")
        
        elif command == "/add_experiment":
            if len(args) < 3:
                print("Usage: /add_experiment <type> <description> <metadata>")
                return
            event_type, description = args[0], args[1]
            metadata = eval(args[2])  # Convert string to dict
            await self.add_experiment(event_type, description, metadata)
            print(f"Added experiment: {event_type} - {description}")
        
        elif command == "/query":
            if not args:
                print("Usage: /query <query_text>")
                return
            query = " ".join(args)
            await self.query_research(query)
        
        elif command == "/analyze":
            if not args:
                print("Usage: /analyze <experiment_id>")
                return
            experiment_id = args[0]
            await self.analyze_experiment(experiment_id)
        
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

    async def add_research_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add research knowledge to the knowledge graph."""
        self.research_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def add_experiment(self, event_type: str, description: str, metadata: Dict[str, Any]) -> None:
        """Add an experiment event."""
        self.experiment_tracking.add_event(
            event_type=event_type,
            description=description,
            metadata=metadata
        )

    async def query_research(self, query: str) -> None:
        """Query research knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.research_knowledge.get_related_concepts(query)
        inferences = self.research_knowledge.get_inferences(query)
        
        print(f"\nResponse: {response}")
        print("\nRelated Concepts:")
        for concept in related:
            print(f"- {concept}")
        print("\nInferences:")
        for inference in inferences:
            print(f"- {inference}")

    async def analyze_experiment(self, experiment_id: str) -> None:
        """Analyze experiment data and patterns."""
        patterns = self.experiment_tracking.get_event_patterns(experiment_id)
        causality = self.experiment_tracking.get_causality_analysis(experiment_id)
        timeline = self.experiment_tracking.get_event_timeline()
        
        print("\nExperiment Patterns:")
        for pattern in patterns:
            print(f"- {pattern}")
        
        print("\nCausality Analysis:")
        for cause in causality:
            print(f"- {cause}")
        
        print("\nTimeline:")
        for event in timeline:
            print(f"- {event}")

    async def show_stats(self) -> None:
        """Show research and experiment statistics."""
        knowledge_stats = self.research_knowledge.get_graph_stats()
        experiment_stats = self.experiment_tracking.get_event_stats()
        
        print("\nKnowledge Graph Statistics:")
        for key, value in knowledge_stats.items():
            print(f"{key}: {value}")
        
        print("\nExperiment Statistics:")
        for key, value in experiment_stats.items():
            print(f"{key}: {value}")

    async def export_data(self, filename: str) -> None:
        """Export research and experiment data."""
        # Implementation would depend on the storage mechanism
        print(f"Exporting data to {filename}...")

    async def import_data(self, filename: str) -> None:
        """Import research and experiment data."""
        # Implementation would depend on the storage mechanism
        print(f"Importing data from {filename}...")

    async def clear_data(self) -> None:
        """Clear all research and experiment data."""
        self.research_knowledge.clear()
        self.experiment_tracking.clear()
        print("All data cleared.")

    def show_help(self) -> None:
        """Show available commands and their usage."""
        print("\nAvailable Commands:")
        print("/add_knowledge <subject> <predicate> <object> <confidence> - Add research knowledge")
        print("/add_experiment <type> <description> <metadata> - Add experiment event")
        print("/query <query_text> - Query research knowledge")
        print("/analyze <experiment_id> - Analyze experiment data")
        print("/stats - Show statistics")
        print("/export <filename> - Export data")
        print("/import <filename> - Import data")
        print("/clear - Clear all data")
        print("/help - Show this help message")
        print("/exit - Exit program")

async def main():
    parser = argparse.ArgumentParser(description="Scientific Research CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="scientific_research.json", help="Storage path")
    args = parser.parse_args()
    
    cli = ScientificResearchCLI(model=args.model, storage_path=args.storage)
    
    print("Scientific Research CLI")
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