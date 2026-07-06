"""
Self-orchestrating agent demo (offline, no API keys needed).

A scripted fake model plays the coordinator: it decomposes a task into two
spawned sub-agents ("research" and "summarize"), collects their results, and
produces a combined final answer. Prints the agent tree and audit trail.

Run: python examples/agents/self_orchestrating_agent.py
"""

import asyncio
import json
from typing import Dict, List, Optional, Union

from multimind.agents import AgentNode, AgentOrchestrator
from multimind.models.base import BaseLLM


class ScriptedModel(BaseLLM):
    """Replays a fixed script of responses, in order."""

    def __init__(self, responses: List[str]):
        super().__init__(model_name="scripted-demo")
        self.responses = list(responses)

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        return self.responses.pop(0)

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ):
        yield await self.generate(prompt)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        return await self.generate(messages[-1]["content"])

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        yield await self.chat(messages)

    async def embeddings(self, text: Union[str, List[str]], **kwargs):
        if isinstance(text, str):
            return [0.0]
        return [[0.0] for _ in text]


SCRIPT = [
    # Turn 1: coordinator spawns a research sub-agent.
    json.dumps(
        {
            "action": "spawn",
            "name": "research",
            "role": "You are a research agent. Gather key facts for the task.",
            "task": "Collect the key facts about solar panel efficiency in 2025.",
        }
    ),
    # The research sub-agent answers its task.
    json.dumps(
        {
            "action": "answer",
            "content": (
                "Commercial silicon panels reach 22-24% efficiency; "
                "perovskite tandem cells hit 33% in lab tests."
            ),
        }
    ),
    # Turn 2: coordinator spawns a summarize sub-agent.
    json.dumps(
        {
            "action": "spawn",
            "name": "summarize",
            "role": "You are a summarization agent. Write one crisp sentence.",
            "task": "Summarize the research findings in one sentence.",
        }
    ),
    # The summarize sub-agent answers its task.
    json.dumps(
        {
            "action": "answer",
            "content": (
                "Solar panels now convert up to a quarter of sunlight commercially, "
                "with lab designs nearing a third."
            ),
        }
    ),
    # Turn 3: coordinator combines both results into the final answer.
    json.dumps(
        {
            "action": "answer",
            "content": (
                "Solar efficiency update: commercial silicon panels reach 22-24%, "
                "perovskite tandems hit 33% in labs — in short, panels now convert "
                "up to a quarter of sunlight, with lab designs nearing a third."
            ),
        }
    ),
]


def print_tree(node: AgentNode, indent: int = 0) -> None:
    pad = "    " * indent
    print(f"{pad}- {node.name} (depth {node.depth})")
    print(f"{pad}  task:   {node.task}")
    print(f"{pad}  result: {node.result}")
    for child in node.children:
        print_tree(child, indent + 1)


async def main() -> None:
    audit_events: List[Dict] = []
    orchestrator = AgentOrchestrator(
        model=ScriptedModel(SCRIPT),
        max_agents=5,
        max_depth=2,
        audit_hook=audit_events.append,
    )

    task = "Give me a one-paragraph update on solar panel efficiency."
    result = await orchestrator.run(task)

    print("Task:")
    print(f"  {task}")
    print()
    print("Agent tree:")
    print_tree(result.agent_tree)
    print()
    print(f"Turns used: {result.turns_used}")
    print(f"Bounds hit: {result.bounds_hit or 'none'}")
    print()
    print("Audit trail:")
    for event in audit_events:
        print(f"  {event}")
    print()
    print("Final answer:")
    print(f"  {result.answer}")


if __name__ == "__main__":
    asyncio.run(main())
