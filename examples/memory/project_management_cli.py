"""
Project Management CLI Example
This example demonstrates how to use MultiMind's memory capabilities for project management applications,
combining event-sourced memory for task tracking and knowledge graph memory for project knowledge.
"""

import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import (
    EventSourcedMemory,
    KnowledgeGraphMemory,
    HybridMemory
)
from multimind.models import OllamaLLM

class ProjectManagementCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "project_management.json"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize event-sourced memory for task tracking
        self.task_tracking = EventSourcedMemory(
            llm=self.llm,
            max_events=10000,
            max_snapshots=1000,
            snapshot_interval=3600,  # 1 hour
            enable_causality=True,
            enable_patterns=True,
            pattern_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_tasks.json"
        )
        
        # Initialize knowledge graph memory for project knowledge
        self.project_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_knowledge.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.task_tracking,
                self.project_knowledge
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path=storage_path
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a project management assistant that helps with task tracking and project knowledge management.
            You maintain a knowledge graph of project information and track task progress and dependencies.
            Always emphasize that you are an AI assistant and not a replacement for project management expertise."""
        )

    async def process_command(self, command: str, args: List[str]) -> None:
        """Process CLI commands."""
        if command == "/add_knowledge":
            if len(args) < 4:
                print("Usage: /add_knowledge <subject> <predicate> <object> <confidence>")
                return
            subject, predicate, object_, confidence = args[0], args[1], args[2], float(args[3])
            await self.add_project_knowledge(subject, predicate, object_, confidence)
            print(f"Added knowledge: {subject} {predicate} {object_} (confidence: {confidence})")
        
        elif command == "/add_task":
            if len(args) < 3:
                print("Usage: /add_task <type> <description> <metadata>")
                return
            event_type, description = args[0], args[1]
            metadata = eval(args[2])  # Convert string to dict
            await self.add_task(event_type, description, metadata)
            print(f"Added task: {event_type} - {description}")
        
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
            await self.analyze_project(project_id)
        
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

    async def add_project_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add project knowledge to the knowledge graph."""
        self.project_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def add_task(self, event_type: str, description: str, metadata: Dict[str, Any]) -> None:
        """Add a task event."""
        self.task_tracking.add_event(
            event_type=event_type,
            description=description,
            metadata=metadata
        )

    async def query_knowledge(self, query: str) -> None:
        """Query project knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.project_knowledge.get_related_concepts(query)
        inferences = self.project_knowledge.get_inferences(query)
        
        print(f"\nResponse: {response}")
        print("\nRelated Concepts:")
        for concept in related:
            print(f"- {concept}")
        print("\nInferences:")
        for inference in inferences:
            print(f"- {inference}")

    async def analyze_project(self, project_id: str) -> None:
        """Analyze project data and patterns."""
        patterns = self.task_tracking.get_event_patterns(project_id)
        causality = self.task_tracking.get_causality_analysis(project_id)
        timeline = self.task_tracking.get_event_timeline()
        
        print("\nProject Patterns:")
        for pattern in patterns:
            print(f"- {pattern}")
        
        print("\nDependency Analysis:")
        for cause in causality:
            print(f"- {cause}")
        
        print("\nTimeline:")
        for event in timeline:
            print(f"- {event}")

    async def show_stats(self) -> None:
        """Show project and task statistics."""
        task_stats = self.task_tracking.get_event_stats()
        knowledge_stats = self.project_knowledge.get_graph_stats()
        
        print("\nTask Statistics:")
        for key, value in task_stats.items():
            print(f"{key}: {value}")
        
        print("\nKnowledge Graph Statistics:")
        for key, value in knowledge_stats.items():
            print(f"{key}: {value}")

    async def export_data(self, filename: str) -> None:
        """Export project and task data."""
        # Implementation would depend on the storage mechanism
        print(f"Exporting data to {filename}...")

    async def import_data(self, filename: str) -> None:
        """Import project and task data."""
        # Implementation would depend on the storage mechanism
        print(f"Importing data from {filename}...")

    async def clear_data(self) -> None:
        """Clear all project and task data."""
        self.task_tracking.clear()
        self.project_knowledge.clear()
        print("All data cleared.")

    def show_help(self) -> None:
        """Show available commands and their usage."""
        print("\nAvailable Commands:")
        print("/add_knowledge <subject> <predicate> <object> <confidence> - Add project knowledge")
        print("/add_task <type> <description> <metadata> - Add task event")
        print("/query <query_text> - Query project knowledge")
        print("/analyze <project_id> - Analyze project data")
        print("/stats - Show statistics")
        print("/export <filename> - Export data")
        print("/import <filename> - Import data")
        print("/clear - Clear all data")
        print("/help - Show this help message")
        print("/exit - Exit program")

async def main():
    parser = argparse.ArgumentParser(description="Project Management CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="project_management.json", help="Storage path")
    args = parser.parse_args()
    
    cli = ProjectManagementCLI(model=args.model, storage_path=args.storage)
    
    print("Project Management CLI")
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