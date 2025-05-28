"""
CLI version of the cognitive scratchpad system, featuring complex use cases.
This example demonstrates advanced reasoning capabilities in an interactive CLI environment.
"""

import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import CognitiveScratchpadMemory
from multimind.models import OllamaLLM

class CognitiveScratchpadCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "cognitive_scratchpad.json"):
        self.llm = OllamaLLM(model=model)
        self.memory = CognitiveScratchpadMemory(
            llm=self.llm,
            max_steps=100,
            max_chains=10,
            chain_depth=5,
            enable_analysis=True,
            analysis_interval=3600,  # 1 hour
            storage_path=storage_path
        )
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="You are a cognitive scratchpad system that performs step-by-step reasoning."
        )
        self.current_chain = None
        self.current_context = {}

    async def process_command(self, command: str) -> None:
        """Process special commands."""
        if command.startswith("/"):
            cmd = command[1:].lower()
            if cmd == "start":
                await self.start_chain()
            elif cmd == "step":
                await self.add_step()
            elif cmd == "query":
                await self.query_reasoning()
            elif cmd == "stats":
                self.show_reasoning_stats()
            elif cmd == "chain":
                self.show_current_chain()
            elif cmd == "analyze":
                await self.analyze_reasoning()
            elif cmd == "deps":
                self.show_dependencies()
            elif cmd == "export":
                self.export_reasoning()
            elif cmd == "import":
                self.import_reasoning()
            elif cmd == "clear":
                self.clear_reasoning()
            elif cmd == "help":
                self.show_help()
            elif cmd == "exit":
                print("Goodbye!")
                exit(0)
            else:
                print("Unknown command. Type /help for available commands.")
        else:
            await self.process_query(command)

    async def start_chain(self) -> None:
        """Start a new reasoning chain."""
        problem = input("\nEnter problem statement: ").strip()
        self.current_chain = self.memory.start_chain(problem)
        print(f"Started new reasoning chain: {self.current_chain}")

    async def add_step(self) -> None:
        """Add a new reasoning step."""
        if not self.current_chain:
            print("No active reasoning chain. Use /start to create one.")
            return

        step = input("\nEnter reasoning step: ").strip()
        confidence = float(input("Confidence (0-1): ").strip())
        
        self.memory.add_step(
            chain_id=self.current_chain,
            step=step,
            confidence=confidence
        )
        print("Step added successfully!")

    async def query_reasoning(self) -> None:
        """Query reasoning chains."""
        query = input("\nEnter your query: ").strip()
        response = await self.mm.chat(query)
        print(f"\nResponse: {response}")
        
        # Get related steps
        related = self.memory.get_related_steps(query)
        if related:
            print("\nRelated Steps:")
            for step in related:
                print(f"- {step}")

    async def process_query(self, query: str) -> None:
        """Process a natural language query."""
        response = await self.mm.chat(query)
        print(f"\nResponse: {response}")
        
        # Update current chain
        self.update_chain(query)

    def update_chain(self, query: str) -> None:
        """Update current reasoning chain."""
        chain = self.memory.extract_chain(query)
        if chain:
            self.current_chain = chain
            self.current_context["chain"] = self.current_chain
            self.current_context["last_updated"] = datetime.now().isoformat()

    def show_reasoning_stats(self) -> None:
        """Display reasoning statistics."""
        stats = self.memory.get_reasoning_stats()
        print("\nReasoning Statistics:")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Total chains: {stats['total_chains']}")
        print(f"Average confidence: {stats['average_confidence']}")
        
        # Show chain distribution
        chains = self.memory.get_chain_distribution()
        if chains:
            print("\nChain Distribution:")
            for chain, count in chains.items():
                print(f"- {chain}: {count} steps")

    def show_current_chain(self) -> None:
        """Display current reasoning chain."""
        if self.current_chain:
            print(f"\nCurrent Chain: {self.current_chain}")
            steps = self.memory.get_chain_steps(self.current_chain)
            if steps:
                print("\nChain Steps:")
                for step in steps:
                    print(f"- {step}")
        else:
            print("\nNo current chain identified.")

    async def analyze_reasoning(self) -> None:
        """Analyze reasoning process."""
        query = input("\nEnter analysis query: ").strip()
        analysis = self.memory.get_reasoning_analysis(query)
        if analysis:
            print("\nReasoning Analysis:")
            for result in analysis:
                print(f"- {result}")
        else:
            print("No analysis results found.")

    def show_dependencies(self) -> None:
        """Display step dependencies."""
        if not self.current_chain:
            print("No active reasoning chain. Use /start to create one.")
            return

        deps = self.memory.get_step_dependencies(self.current_chain)
        if deps:
            print("\nStep Dependencies:")
            for step, dependencies in deps.items():
                print(f"\nStep: {step}")
                print("Depends on:")
                for dep in dependencies:
                    print(f"- {dep}")
        else:
            print("No dependencies found.")

    def export_reasoning(self) -> None:
        """Export reasoning to file."""
        filename = input("\nEnter export filename: ").strip()
        self.memory.export_reasoning(filename)
        print(f"Reasoning exported to {filename}")

    def import_reasoning(self) -> None:
        """Import reasoning from file."""
        filename = input("\nEnter import filename: ").strip()
        self.memory.import_reasoning(filename)
        print(f"Reasoning imported from {filename}")

    def clear_reasoning(self) -> None:
        """Clear all reasoning."""
        self.memory.clear()
        self.current_chain = None
        self.current_context = {}
        print("\nReasoning cleared.")

    def show_help(self) -> None:
        """Display available commands."""
        print("\nAvailable Commands:")
        print("/start - Start new reasoning chain")
        print("/step - Add reasoning step")
        print("/query - Query reasoning")
        print("/stats - Show reasoning statistics")
        print("/chain - Show current chain")
        print("/analyze - Analyze reasoning")
        print("/deps - Show step dependencies")
        print("/export - Export reasoning")
        print("/import - Import reasoning")
        print("/clear - Clear all reasoning")
        print("/help - Show this help message")
        print("/exit - Exit the program")

async def main():
    parser = argparse.ArgumentParser(description="MultiMind Cognitive Scratchpad CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="cognitive_scratchpad.json", help="Reasoning storage path")
    args = parser.parse_args()

    cs = CognitiveScratchpadCLI(model=args.model, storage_path=args.storage)
    
    print("Welcome to MultiMind Cognitive Scratchpad CLI!")
    print("Type /help for available commands.")
    print("Type /exit to quit.")
    
    while True:
        try:
            command = input("\nCommand: ").strip()
            if command:
                await cs.process_command(command)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 