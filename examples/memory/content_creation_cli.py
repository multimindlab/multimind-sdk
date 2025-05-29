"""
Content Creation CLI Example
This example demonstrates how to use MultiMind's memory capabilities for content creation applications,
combining knowledge graph memory for content knowledge and cognitive scratchpad memory for content planning.
"""

import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import (
    KnowledgeGraphMemory,
    CognitiveScratchpadMemory,
    HybridMemory
)
from multimind.models import OllamaLLM

class ContentCreationCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "content_creation.json"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize knowledge graph memory for content knowledge
        self.content_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_knowledge.json"
        )
        
        # Initialize cognitive scratchpad memory for content planning
        self.content_planning = CognitiveScratchpadMemory(
            llm=self.llm,
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            enable_dependencies=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_planning.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.content_knowledge,
                self.content_planning
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path=storage_path
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a content creation assistant that helps with content planning and knowledge management.
            You maintain a knowledge graph of content information and track content planning steps and dependencies.
            Always emphasize that you are an AI assistant and not a replacement for content creation expertise."""
        )

    async def process_command(self, command: str, args: List[str]) -> None:
        """Process CLI commands."""
        if command == "/add_knowledge":
            if len(args) < 4:
                print("Usage: /add_knowledge <subject> <predicate> <object> <confidence>")
                return
            subject, predicate, object_, confidence = args[0], args[1], args[2], float(args[3])
            await self.add_content_knowledge(subject, predicate, object_, confidence)
            print(f"Added knowledge: {subject} {predicate} {object_} (confidence: {confidence})")
        
        elif command == "/start_plan":
            if len(args) < 1:
                print("Usage: /start_plan <description>")
                return
            description = " ".join(args)
            chain_id = await self.start_content_plan(description)
            print(f"Started content plan with ID: {chain_id}")
        
        elif command == "/add_step":
            if len(args) < 2:
                print("Usage: /add_step <chain_id> <step_description>")
                return
            chain_id, description = args[0], " ".join(args[1:])
            await self.add_planning_step(chain_id, description)
            print(f"Added step to chain {chain_id}")
        
        elif command == "/query":
            if not args:
                print("Usage: /query <query_text>")
                return
            query = " ".join(args)
            await self.query_knowledge(query)
        
        elif command == "/analyze":
            if not args:
                print("Usage: /analyze <chain_id>")
                return
            chain_id = args[0]
            await self.analyze_content_plan(chain_id)
        
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

    async def add_content_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add content knowledge to the knowledge graph."""
        self.content_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def start_content_plan(self, description: str) -> str:
        """Start a new content planning chain."""
        chain_id = self.content_planning.start_chain(description)
        return chain_id

    async def add_planning_step(self, chain_id: str, description: str) -> None:
        """Add a step to the content planning chain."""
        self.content_planning.add_step(
            chain_id=chain_id,
            step_description=description
        )

    async def query_knowledge(self, query: str) -> None:
        """Query content knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.content_knowledge.get_related_concepts(query)
        inferences = self.content_knowledge.get_inferences(query)
        
        print(f"\nResponse: {response}")
        print("\nRelated Concepts:")
        for concept in related:
            print(f"- {concept}")
        print("\nInferences:")
        for inference in inferences:
            print(f"- {inference}")

    async def analyze_content_plan(self, chain_id: str) -> None:
        """Analyze content planning chain."""
        steps = self.content_planning.get_chain_steps(chain_id)
        dependencies = self.content_planning.get_chain_dependencies(chain_id)
        analysis = self.content_planning.analyze_chain(chain_id)
        
        print("\nPlanning Steps:")
        for step in steps:
            print(f"- {step}")
        
        print("\nDependencies:")
        for dep in dependencies:
            print(f"- {dep}")
        
        print("\nAnalysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")

    async def show_stats(self) -> None:
        """Show content and planning statistics."""
        knowledge_stats = self.content_knowledge.get_graph_stats()
        planning_stats = self.content_planning.get_chain_stats()
        
        print("\nKnowledge Graph Statistics:")
        for key, value in knowledge_stats.items():
            print(f"{key}: {value}")
        
        print("\nPlanning Statistics:")
        for key, value in planning_stats.items():
            print(f"{key}: {value}")

    async def export_data(self, filename: str) -> None:
        """Export content and planning data."""
        # Implementation would depend on the storage mechanism
        print(f"Exporting data to {filename}...")

    async def import_data(self, filename: str) -> None:
        """Import content and planning data."""
        # Implementation would depend on the storage mechanism
        print(f"Importing data from {filename}...")

    async def clear_data(self) -> None:
        """Clear all content and planning data."""
        self.content_knowledge.clear()
        self.content_planning.clear()
        print("All data cleared.")

    def show_help(self) -> None:
        """Show available commands and their usage."""
        print("\nAvailable Commands:")
        print("/add_knowledge <subject> <predicate> <object> <confidence> - Add content knowledge")
        print("/start_plan <description> - Start new content plan")
        print("/add_step <chain_id> <step_description> - Add planning step")
        print("/query <query_text> - Query content knowledge")
        print("/analyze <chain_id> - Analyze content plan")
        print("/stats - Show statistics")
        print("/export <filename> - Export data")
        print("/import <filename> - Import data")
        print("/clear - Clear all data")
        print("/help - Show this help message")
        print("/exit - Exit program")

async def main():
    parser = argparse.ArgumentParser(description="Content Creation CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="content_creation.json", help="Storage path")
    args = parser.parse_args()
    
    cli = ContentCreationCLI(model=args.model, storage_path=args.storage)
    
    print("Content Creation CLI")
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