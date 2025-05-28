"""
CLI version of the chatbot with memory example, featuring complex use cases.
This example demonstrates advanced memory capabilities in an interactive CLI environment.
"""

import asyncio
import argparse
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import (
    HybridMemory,
    VectorStoreMemory,
    TimeWeightedMemory,
    KnowledgeGraphMemory,
    TokenBufferMemory,
    DNCMemory
)
from multimind.models import OllamaLLM

class ChatbotCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "chatbot_memory.json"):
        self.llm = OllamaLLM(model=model)
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                VectorStoreMemory,  # For semantic search
                TimeWeightedMemory,  # For recency-based retrieval
                KnowledgeGraphMemory,  # For structured knowledge
                TokenBufferMemory,  # For recent context
                DNCMemory  # For complex reasoning
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path=storage_path
        )
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="You are a helpful AI assistant with excellent memory and reasoning capabilities."
        )
        self.conversation_history = []
        self.current_topic = None
        self.current_context = {}

    async def process_command(self, command: str) -> None:
        """Process special commands."""
        if command.startswith("/"):
            cmd = command[1:].lower()
            if cmd == "stats":
                self.show_memory_stats()
            elif cmd == "topic":
                self.show_current_topic()
            elif cmd == "context":
                self.show_current_context()
            elif cmd == "clear":
                self.clear_memory()
            elif cmd == "help":
                self.show_help()
            elif cmd == "exit":
                print("Goodbye!")
                exit(0)
            else:
                print("Unknown command. Type /help for available commands.")
        else:
            await self.process_message(command)

    async def process_message(self, message: str) -> None:
        """Process a regular message."""
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        # Get response
        response = await self.mm.chat(message)
        
        # Update conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        # Update current topic and context
        self.update_context(message, response)
        
        # Display response
        print(f"\nAssistant: {response}")

    def update_context(self, message: str, response: str) -> None:
        """Update current topic and context based on conversation."""
        # Extract potential topics from message
        topics = self.memory.extract_topics(message)
        if topics:
            self.current_topic = topics[0]
            self.current_context["topic"] = self.current_topic
            self.current_context["last_updated"] = datetime.now().isoformat()

    def show_memory_stats(self) -> None:
        """Display memory statistics."""
        stats = self.memory.get_memory_stats()
        print("\nMemory Statistics:")
        print(f"Total items: {stats['total_items']}")
        print(f"Memory types used: {stats['memory_types_used']}")
        print(f"Routing performance: {stats['routing_performance']}")
        
        # Show memory suggestions
        suggestions = self.memory.get_memory_suggestions()
        if suggestions:
            print("\nMemory Suggestions:")
            for suggestion in suggestions:
                print(f"- {suggestion}")

    def show_current_topic(self) -> None:
        """Display current conversation topic."""
        if self.current_topic:
            print(f"\nCurrent Topic: {self.current_topic}")
            related = self.memory.get_related_concepts(self.current_topic)
            if related:
                print("\nRelated Concepts:")
                for concept in related:
                    print(f"- {concept}")
        else:
            print("\nNo current topic identified.")

    def show_current_context(self) -> None:
        """Display current conversation context."""
        print("\nCurrent Context:")
        for key, value in self.current_context.items():
            print(f"{key}: {value}")

    def clear_memory(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.conversation_history = []
        self.current_topic = None
        self.current_context = {}
        print("\nMemory cleared.")

    def show_help(self) -> None:
        """Display available commands."""
        print("\nAvailable Commands:")
        print("/stats - Show memory statistics")
        print("/topic - Show current conversation topic")
        print("/context - Show current conversation context")
        print("/clear - Clear all memory")
        print("/help - Show this help message")
        print("/exit - Exit the program")

async def main():
    parser = argparse.ArgumentParser(description="MultiMind Chatbot CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--storage", default="chatbot_memory.json", help="Memory storage path")
    args = parser.parse_args()

    chatbot = ChatbotCLI(model=args.model, storage_path=args.storage)
    
    print("Welcome to MultiMind Chatbot CLI!")
    print("Type /help for available commands.")
    print("Type /exit to quit.")
    
    while True:
        try:
            message = input("\nYou: ").strip()
            if message:
                await chatbot.process_command(message)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 