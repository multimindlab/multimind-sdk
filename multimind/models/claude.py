"""
Anthropic Claude model implementation.
"""

import os
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import anthropic
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..core.exceptions import ConfigurationError
from .base import BaseLLM

class ClaudeModel(BaseLLM):
    """Anthropic Claude model implementation."""

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        # Load API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "Claude API key is not configured. "
                "Set ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable, "
                "or pass api_key explicitly."
            )
        self.client = AsyncAnthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APIError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _messages_create(self, **kwargs: Any):
        """Internal helper with retry for messages.create."""
        return await self.client.messages.create(**kwargs)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using Claude's completion API."""
        # Anthropic API requires max_tokens to be set
        if max_tokens is None:
            max_tokens = 1024  # Default value
        response = await self._messages_create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text if response.content else ""

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using Claude's completion API."""
        # Anthropic API requires max_tokens to be set
        if max_tokens is None:
            max_tokens = 1024  # Default value
        stream = await self._messages_create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.text:
                yield chunk.delta.text

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using Claude's chat API."""
        # Anthropic API requires max_tokens to be set
        if max_tokens is None:
            max_tokens = 1024  # Default value
        response = await self._messages_create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text if response.content else ""

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion using Claude's chat API."""
        # Anthropic API requires max_tokens to be set
        if max_tokens is None:
            max_tokens = 1024  # Default value
        stream = await self._messages_create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.text:
                yield chunk.delta.text

    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Claude's embeddings API."""
        raise NotImplementedError("Claude does not currently support embeddings generation")