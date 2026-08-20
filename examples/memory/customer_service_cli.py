"""
CLI customer-service support assistant backed by BufferMemory.

Demonstrates a minimal, working support-ticket chat loop: conversation
history is kept in a token-bounded BufferMemory (persisted to disk between
runs), each turn is tagged with a lightweight ticket category, and replies
are generated directly via the configured LLM. Uses BufferMemory.add_message/
get_messages directly rather than multimind.core.MultiMind.chat(), since
that helper expects an AgentMemory with an add_interaction() method that
BufferMemory (and every other multimind.memory.* class) does not implement.
"""

import argparse
import asyncio

from multimind.memory import BufferMemory
from multimind.models import OllamaModel

SYSTEM_PROMPT = (
    "You are a customer support agent. Be concise, polite, and ask a "
    "clarifying question when the customer's request is ambiguous."
)


class CustomerServiceCLI:
    def __init__(self, model: str = "mistral", storage_path: str = "customer_service_memory.json"):
        self.llm = OllamaModel(model_name=model)
        self.memory = BufferMemory(
            memory_key="customer_service",
            storage_path=storage_path,
            max_tokens=3000,
            strategy="sliding",
        )
        self.current_category = "general"

    async def load(self) -> None:
        await self.memory.load()

    def classify(self, text: str) -> str:
        """Very small keyword classifier for ticket category."""
        lowered = text.lower()
        if any(word in lowered for word in ("charge", "invoice", "refund", "payment", "bill")):
            return "billing"
        if any(word in lowered for word in ("error", "bug", "crash", "not working", "broken")):
            return "technical"
        if any(word in lowered for word in ("password", "login", "account", "email")):
            return "account"
        return "general"

    async def process_command(self, command: str) -> None:
        if command.startswith("/"):
            cmd = command[1:].lower()
            if cmd == "stats":
                self.show_stats()
            elif cmd == "history":
                await self.show_history()
            elif cmd == "clear":
                await self.memory.clear()
                print("\nConversation cleared.")
            elif cmd == "help":
                self.show_help()
            elif cmd == "exit":
                print("Goodbye!")
                raise SystemExit(0)
            else:
                print("Unknown command. Type /help for available commands.")
        else:
            await self.process_message(command)

    async def process_message(self, message: str) -> None:
        self.current_category = self.classify(message)
        await self.memory.add_message(
            {"role": "user", "content": message},
            metadata={"category": self.current_category},
        )

        history = await self.memory.get_messages()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        response = await self.llm.chat(messages=messages)

        await self.memory.add_message({"role": "assistant", "content": response})
        print(f"\n[{self.current_category}] Agent: {response}")

    def show_stats(self) -> None:
        stats = self.memory.get_stats()
        print("\nSession Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

    async def show_history(self) -> None:
        for entry in self.memory.get_messages_with_metadata():
            role = entry["message"].get("role", "?")
            content = entry["message"].get("content", "")
            category = entry["metadata"].get("category", "")
            tag = f" ({category})" if category else ""
            print(f"{role}{tag}: {content}")

    def show_help(self) -> None:
        print("\nAvailable Commands:")
        print("/stats   - Show buffer statistics")
        print("/history - Show full conversation with ticket categories")
        print("/clear   - Clear the conversation buffer")
        print("/help    - Show this help message")
        print("/exit    - Exit the program")


async def main() -> None:
    parser = argparse.ArgumentParser(description="MultiMind Customer Service CLI")
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument(
        "--storage", default="customer_service_memory.json", help="Memory storage path"
    )
    args = parser.parse_args()

    cli = CustomerServiceCLI(model=args.model, storage_path=args.storage)
    await cli.load()

    print("Welcome to MultiMind Customer Service CLI!")
    print("Type /help for available commands, /exit to quit.")

    while True:
        try:
            message = input("\nCustomer: ").strip()
            if message:
                await cli.process_command(message)
        except (KeyboardInterrupt, SystemExit):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
