"""
Anthropic Claude model implementation.
"""

import os
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Type, Union

import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.exceptions import ConfigurationError
from .base import BaseLLM, resolve_images


def _image_content_block(resolved: Dict[str, str]) -> Dict[str, Any]:
    """Build an Anthropic-format image content block from a resolved image."""
    if resolved["kind"] == "url":
        source: Dict[str, str] = {"type": "url", "url": resolved["url"]}
    else:
        source = {
            "type": "base64",
            "media_type": resolved["media_type"],
            "data": resolved["data"],
        }
    return {"type": "image", "source": source}


def _image_blocks(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_image_content_block(resolved) for resolved in resolve_images(images)]


class ClaudeModel(BaseLLM):
    """Anthropic Claude model implementation."""

    def __init__(
        self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None, **kwargs
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

    async def _structured_create(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """Force a tool-use call whose input schema is the Pydantic model."""
        if not (isinstance(response_format, type) and issubclass(response_format, BaseModel)):
            raise ValueError(
                "response_format for ClaudeModel must be a Pydantic BaseModel subclass"
            )
        tool_name = "structured_output"
        response = await self._messages_create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[
                {
                    "name": tool_name,
                    "description": f"Record the answer as a {response_format.__name__} object.",
                    "input_schema": response_format.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
            **kwargs,
        )
        tool_input = next(
            (
                block.input
                for block in response.content or []
                if getattr(block, "type", None) == "tool_use"
            ),
            None,
        )
        if tool_input is None:
            raw = "".join(getattr(block, "text", "") for block in response.content or [])
            raise ValueError(
                f"Claude did not return structured output for {response_format.__name__}. "
                f"Raw response: {raw}"
            )
        try:
            return response_format.model_validate(tool_input)
        except ValidationError as exc:
            raise ValueError(
                f"Failed to parse response as {response_format.__name__}: {exc}. "
                f"Raw response: {tool_input}"
            ) from exc

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[BaseModel]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[str, BaseModel]:
        """Generate text using Claude's completion API.

        ``images`` is an optional list of dicts, each with one of the keys
        ``path``, ``bytes`` or ``url`` (plus optional ``media_type``).
        """
        # Anthropic API requires max_tokens to be set
        if max_tokens is None:
            max_tokens = 1024  # Default value
        content: Any = prompt
        if images is not None:
            content = _image_blocks(images) + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        if response_format is not None:
            return await self._structured_create(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs,
            )
        response = await self._messages_create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text if response.content else ""

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
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
        response_format: Optional[Type[BaseModel]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[str, BaseModel]:
        """Generate chat completion using Claude's chat API.

        ``images`` (optional) are attached to the last user message; see
        ``generate`` for the accepted item shapes.
        """
        # Anthropic API requires max_tokens to be set
        if max_tokens is None:
            max_tokens = 1024  # Default value
        if images is not None:
            blocks = _image_blocks(images)
            messages = [dict(msg) for msg in messages]
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    msg["content"] = blocks + [{"type": "text", "text": msg["content"]}]
                    break
            else:
                raise ValueError("images require at least one user message")
        if response_format is not None:
            return await self._structured_create(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs,
            )
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
        **kwargs,
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
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Claude's embeddings API."""
        raise NotImplementedError("Claude does not currently support embeddings generation")
