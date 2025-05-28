"""
Data Analysis CLI Example
This example demonstrates how to use MultiMind's memory capabilities for data analysis applications,
combining knowledge graph memory for analysis knowledge and cognitive scratchpad memory for analysis steps.
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

class DataAnalysisCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "data_analysis.json"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize knowledge graph memory for analysis knowledge
        self.analysis_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_knowledge.json"
        )
        
        # Initialize cognitive scratchpad memory for analysis steps
        self.analysis_steps = CognitiveScratchpadMemory(
            llm=self.llm,
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            enable_dependencies=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=f"{storage_path}_steps.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.analysis_knowledge,
                self.analysis_steps
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path=storage_path
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a data analysis assistant that helps with analysis knowledge management and step tracking.
            You maintain a knowledge graph of analysis information and track analysis steps and dependencies.
            Always emphasize that you are an AI assistant and not a replacement for data analysis expertise."""
        )

    async def process_command(self, command: str, args: List[str]) -> None:
        """Process CLI commands."""
        if command == "/add_knowledge":
            if len(args) < 4:
                print("Usage: /add_knowledge <subject> <predicate> <object> <confidence>")
                return
            subject, predicate, object_, confidence = args[0], args[1], args[2], float(args[3])
            await self.add_analysis_knowledge(subject, predicate, object_, confidence)
            print(f"Added knowledge: {subject} {predicate} {object_} (confidence: {confidence})")
        
        elif command == "/start_analysis":
            if len(args) < 1:
                print("Usage: /start_analysis <description>")
                return
            description = " ".join(args)
            chain_id = await self.start_analysis(description)
            print(f"Started analysis with ID: {chain_id}")
        
        elif command == "/add_step":
            if len(args) < 2:
                print("Usage: /add_step <chain_id> <step_description>")
                return
            chain_id, description = args[0], " ".join(args[1:])
            await self.add_analysis_step(chain_id, description)
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
            await self.analyze_chain(chain_id)
        
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

    async def add_analysis_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add analysis knowledge to the knowledge graph."""
        self.analysis_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def start_analysis(self, description: str) -> str:
        """Start a new analysis chain."""
        chain_id = self.analysis_steps.start_chain(description)
        return chain_id

    async def add_analysis_step(self, chain_id: str, description: str) -> None:
        """Add a step to the analysis chain."""
        self.analysis_steps.add_step(
            chain_id=chain_id,
            step_description=description
        )

    async def query_knowledge(self, query: str) -> None:
        """Query analysis knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.analysis_knowledge.get_related_concepts(query)
        inferences = self.analysis_knowledge.get_inferences(query)
        
        print(f"\nResponse: {response}")
        print("\nRelated Concepts:")
        for concept in related:
            print(f"- {concept}")
        print("\nInferences:")
        for inference in inferences:
            print(f"- {inference}")

    async def analyze_chain(self, chain_id: str) -> None:
        """Analyze analysis chain."""
        steps = self.analysis_steps.get_chain_steps(chain_id)
        dependencies = self.analysis_steps.get_chain_dependencies(chain_id)
        analysis = self.analysis_steps.analyze_chain(chain_id)
        
        print("\nAnalysis Steps:")
        for step in steps:
            print(f"- {step}")
        
        print("\nDependencies:")
        for dep in dependencies:
            print(f"- {dep}")
        
        print("\nAnalysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")

    async def show_stats(self) -> None:
        """Show analysis and knowledge statistics."""
        knowledge_stats = self.analysis_knowledge.get_graph_stats()
        step_stats = self.analysis_steps.get_chain_stats()
        
        print("\nKnowledge Graph Statistics:")
        for key, value in knowledge_stats.items():
            print(f"{key}: {value}")
        
        print("\nAnalysis Statistics:")
        for key, value in step_stats.items():
            print(f"{key}: {value}")

    async def export_data(self, filename: str) -> None:
        """Export analysis and knowledge data."""
        # Implementation would depend on the storage mechanism
        print(f"Exporting data to {filename}...")

    async def import_data(self, filename: str) -> None:
        """Import analysis and knowledge data."""
        # Implementation would depend on the storage mechanism
        print(f"Importing data from {filename}...")

    async def clear_data(self) -> None:
        """Clear all analysis and knowledge data."""
        self.analysis_knowledge.clear()
        self.analysis_steps.clear()
        print("All data cleared.")

    def show_help(self) -> None:
        """Show available commands and their usage."""
        print("\nAvailable Commands:")
        print("/add_knowledge <subject> <predicate> <object> <confidence> - Add analysis knowledge")
        print("/start_analysis <description> - Start new analysis")
        print("/add_step <chain_id> <step_description> - Add analysis step")
        print("/query <query_text> - Query analysis knowledge")
        print("/analyze <chain_id> - Analyze analysis chain")
        print("/stats - Show statistics")
        print("/export <filename> - Export data")
        print("/import <filename> - Import data")
        print("/clear - Clear all data")
        print("/help - Show this help message")
        print("/exit - Exit program")

async def main():
    parser = argparse.ArgumentParser(description="Data Analysis CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="data_analysis.json", help="Storage path")
    args = parser.parse_args()
    
    cli = DataAnalysisCLI(model=args.model, storage_path=args.storage)
    
    print("Data Analysis CLI")
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