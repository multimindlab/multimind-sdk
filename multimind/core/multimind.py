"""
MultiMind class - Main entry point for the SDK.
"""

from typing import List, Optional, Union

from ..agents.memory import AgentMemory
from ..memory.base import BaseMemory
from .base import BaseLLM


class MultiMind:
    """Main class for interacting with the MultiMind SDK."""

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[Union[AgentMemory, BaseMemory]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """Initialize MultiMind with an LLM and optional memory."""
        self.llm = llm
        self.memory = memory
        self.system_prompt = system_prompt
        self.kwargs = kwargs

    async def _history(self) -> List[dict]:
        """Prior conversation turns, available only for BaseMemory-style memories."""
        if isinstance(self.memory, BaseMemory):
            return list(await self.memory.get_messages())
        return []

    async def _remember(self, prompt: str, response: str) -> None:
        """Record a prompt/response pair using whichever memory API is present."""
        if isinstance(self.memory, BaseMemory):
            await self.memory.add_message({"role": "user", "content": prompt})
            await self.memory.add_message({"role": "assistant", "content": response})
        elif isinstance(self.memory, AgentMemory):
            self.memory.add_task(prompt)
            self.memory.add_response({"response": response})

    async def chat(
        self, message: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """Send a message and get a response."""
        messages = await self._history()
        messages.append({"role": "user", "content": message})
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        response = await self.llm.chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        await self._remember(message, response)

        return response

    async def chat_stream(
        self, message: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ):
        """Send a message and get a streaming response."""
        messages = [{"role": "user", "content": message}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        async for chunk in self.llm.chat_stream(
            messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        ):
            yield chunk

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """Generate text from a prompt."""
        response = await self.llm.generate(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        await self._remember(prompt, response)

        return response

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ):
        """Generate streaming text from a prompt."""
        async for chunk in self.llm.generate_stream(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        ):
            yield chunk

    async def get_embeddings(self, text: str, **kwargs) -> List[float]:
        """Get embeddings for text."""
        return await self.llm.embeddings(text, **kwargs)
