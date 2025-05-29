"""
CLI version of the knowledge management system, featuring complex use cases.
This example demonstrates advanced knowledge graph capabilities in an interactive CLI environment.
"""

import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import KnowledgeGraphMemory
from multimind.models import OllamaLLM

class KnowledgeManagementCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "knowledge_graph.json"):
        self.llm = OllamaLLM(model=model)
        self.memory = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=1000,
            max_edges=5000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path=storage_path
        )
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="You are a knowledge management system that maintains structured information."
        )
        self.current_domain = None
        self.current_context = {}

    async def process_command(self, command: str) -> None:
        """Process special commands."""
        if command.startswith("/"):
            cmd = command[1:].lower()
            if cmd == "add":
                await self.add_knowledge()
            elif cmd == "query":
                await self.query_knowledge()
            elif cmd == "stats":
                self.show_graph_stats()
            elif cmd == "domain":
                self.show_current_domain()
            elif cmd == "infer":
                await self.run_inference()
            elif cmd == "validate":
                await self.validate_knowledge()
            elif cmd == "export":
                self.export_knowledge()
            elif cmd == "import":
                self.import_knowledge()
            elif cmd == "clear":
                self.clear_knowledge()
            elif cmd == "help":
                self.show_help()
            elif cmd == "exit":
                print("Goodbye!")
                exit(0)
            else:
                print("Unknown command. Type /help for available commands.")
        else:
            await self.process_query(command)

    async def add_knowledge(self) -> None:
        """Add new knowledge to the graph."""
        print("\nAdding new knowledge:")
        subject = input("Subject: ").strip()
        predicate = input("Predicate: ").strip()
        object_ = input("Object: ").strip()
        confidence = float(input("Confidence (0-1): ").strip())
        
        self.memory.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )
        print("Knowledge added successfully!")

    async def query_knowledge(self) -> None:
        """Query the knowledge graph."""
        query = input("\nEnter your query: ").strip()
        response = await self.mm.chat(query)
        print(f"\nResponse: {response}")
        
        # Get related concepts
        related = self.memory.get_related_concepts(query)
        if related:
            print("\nRelated Concepts:")
            for concept in related:
                print(f"- {concept}")

    async def process_query(self, query: str) -> None:
        """Process a natural language query."""
        response = await self.mm.chat(query)
        print(f"\nResponse: {response}")
        
        # Update current domain
        self.update_domain(query)

    def update_domain(self, query: str) -> None:
        """Update current knowledge domain."""
        domain = self.memory.extract_domain(query)
        if domain:
            self.current_domain = domain
            self.current_context["domain"] = self.current_domain
            self.current_context["last_updated"] = datetime.now().isoformat()

    def show_graph_stats(self) -> None:
        """Display knowledge graph statistics."""
        stats = self.memory.get_graph_stats()
        print("\nKnowledge Graph Statistics:")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total edges: {stats['total_edges']}")
        print(f"Average confidence: {stats['average_confidence']}")
        
        # Show domain distribution
        domains = self.memory.get_domain_distribution()
        if domains:
            print("\nDomain Distribution:")
            for domain, count in domains.items():
                print(f"- {domain}: {count} concepts")

    def show_current_domain(self) -> None:
        """Display current knowledge domain."""
        if self.current_domain:
            print(f"\nCurrent Domain: {self.current_domain}")
            concepts = self.memory.get_domain_concepts(self.current_domain)
            if concepts:
                print("\nDomain Concepts:")
                for concept in concepts:
                    print(f"- {concept}")
        else:
            print("\nNo current domain identified.")

    async def run_inference(self) -> None:
        """Run inference on the knowledge graph."""
        query = input("\nEnter inference query: ").strip()
        inferences = self.memory.get_inferences(query)
        if inferences:
            print("\nInferred Knowledge:")
            for inference in inferences:
                print(f"- {inference}")
        else:
            print("No inferences found.")

    async def validate_knowledge(self) -> None:
        """Validate knowledge in the graph."""
        print("\nValidating knowledge graph...")
        validation_results = self.memory.validate_knowledge()
        if validation_results:
            print("\nValidation Results:")
            for result in validation_results:
                print(f"- {result}")
        else:
            print("No validation issues found.")

    def export_knowledge(self) -> None:
        """Export knowledge graph to file."""
        filename = input("\nEnter export filename: ").strip()
        self.memory.export_knowledge(filename)
        print(f"Knowledge exported to {filename}")

    def import_knowledge(self) -> None:
        """Import knowledge graph from file."""
        filename = input("\nEnter import filename: ").strip()
        self.memory.import_knowledge(filename)
        print(f"Knowledge imported from {filename}")

    def clear_knowledge(self) -> None:
        """Clear all knowledge."""
        self.memory.clear()
        self.current_domain = None
        self.current_context = {}
        print("\nKnowledge graph cleared.")

    def show_help(self) -> None:
        """Display available commands."""
        print("\nAvailable Commands:")
        print("/add - Add new knowledge")
        print("/query - Query the knowledge graph")
        print("/stats - Show graph statistics")
        print("/domain - Show current knowledge domain")
        print("/infer - Run inference")
        print("/validate - Validate knowledge")
        print("/export - Export knowledge graph")
        print("/import - Import knowledge graph")
        print("/clear - Clear all knowledge")
        print("/help - Show this help message")
        print("/exit - Exit the program")

async def main():
    parser = argparse.ArgumentParser(description="MultiMind Knowledge Management CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="knowledge_graph.json", help="Knowledge graph storage path")
    args = parser.parse_args()

    km = KnowledgeManagementCLI(model=args.model, storage_path=args.storage)
    
    print("Welcome to MultiMind Knowledge Management CLI!")
    print("Type /help for available commands.")
    print("Type /exit to quit.")
    
    while True:
        try:
            command = input("\nCommand: ").strip()
            if command:
                await km.process_command(command)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 