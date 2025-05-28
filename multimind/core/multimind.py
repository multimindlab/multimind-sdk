"""
MultiMind class - Main entry point for the SDK.
"""

from typing import Optional, List, Dict, Any
from .base import BaseLLM
from ..agents.memory import AgentMemory

class MultiMind:
    """Main class for interacting with the MultiMind SDK."""

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[AgentMemory] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Initialize MultiMind with an LLM and optional memory."""
        self.llm = llm
        self.memory = memory
        self.system_prompt = system_prompt
        self.kwargs = kwargs

    async def chat(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Send a message and get a response."""
        messages = [{"role": "user", "content": message}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        response = await self.llm.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        if self.memory:
            await self.memory.add_interaction(message, response)
        
        return response

    async def chat_stream(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Send a message and get a streaming response."""
        messages = [{"role": "user", "content": message}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        async for chunk in self.llm.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        response = await self.llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        if self.memory:
            await self.memory.add_interaction(prompt, response)
        
        return response

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate streaming text from a prompt."""
        async for chunk in self.llm.generate_stream(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk

    async def get_embeddings(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """Get embeddings for text."""
        return await self.llm.embeddings(text, **kwargs) 