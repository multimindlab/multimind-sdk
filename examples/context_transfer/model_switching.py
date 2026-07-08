"""Seamless model switching demo (offline, no API keys required).

A conversation starts on fake provider "nova-1" (model A), then switches to
fake provider "atlas-2" (model B) with strategy="summary". Model B answers a
question that requires knowledge from the model-A era, proving the context
survived the switch.

Run: python examples/context_transfer/model_switching.py
"""

import asyncio

from multimind.client import ModelSession


class NovaModel:
    """Fake provider A: acknowledges facts; can summarize its own transcript."""

    def __init__(self):
        self.model_name = "nova-1"

    async def chat(self, messages, **kwargs):
        last = messages[-1]["content"]
        if last.startswith("Summarize the following conversation"):
            # Derive the summary from the transcript it was actually given.
            facts = [
                line.split(":", 1)[1].strip()
                for line in last.splitlines()
                if line.startswith("user:")
            ]
            return "Key facts from the conversation: " + " | ".join(facts)
        return f"Understood, I have noted that: {last}"


class AtlasModel:
    """Fake provider B: answers only from the context it receives."""

    def __init__(self):
        self.model_name = "atlas-2"

    async def chat(self, messages, **kwargs):
        context = " ".join(m["content"] for m in messages[:-1])
        keywords = [
            w.strip("?,.!").lower()
            for w in messages[-1]["content"].split()
            if len(w.strip("?,.!")) > 3
        ]
        known = [f.strip() for f in context.replace("|", "\n").splitlines() if f.strip()]
        relevant = [fact for fact in known if any(word in fact.lower() for word in keywords)]
        if relevant:
            return "From what I know of your earlier conversation: " + "; ".join(relevant)
        return "I have no context about that."


async def main():
    model_a = NovaModel()
    model_b = AtlasModel()

    session = ModelSession(model_a)

    print("=== Conversation on model A (nova-1) ===")
    for prompt in (
        "My project is called Zephyr, a wind-turbine monitoring tool.",
        "It is written in Rust and deploys to a Raspberry Pi cluster.",
        "The launch deadline is March 15.",
    ):
        reply = await session.chat(prompt)
        print(f"user: {prompt}")
        print(f"{session.model.model_name}: {reply}\n")

    print("=== Switching to model B (atlas-2) with strategy='summary' ===")
    await session.switch(model_b, strategy="summary")
    transfer = session.transfers[-1]
    print(
        f"transfer: {transfer['from_model']} -> {transfer['to_model']} "
        f"(strategy={transfer['strategy']}, messages={transfer['message_count']})"
    )
    print(f"injected context: {session.history[0]['content']}\n")

    print("=== Model B answers using model-A era knowledge ===")
    question = "What is my project Zephyr written in, and when is the deadline?"
    answer = await session.chat(question)
    print(f"user: {question}")
    print(f"{session.model.model_name}: {answer}")

    assert "Rust" in answer and "March 15" in answer, "context was lost in the switch!"
    print("\nSuccess: knowledge survived the provider switch.")


if __name__ == "__main__":
    asyncio.run(main())
