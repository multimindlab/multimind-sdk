"""
OpenAI model implementation.
"""

import openai
from typing import List, Dict, Any, Optional, AsyncGenerator, Union, cast
from openai.types.chat import ChatCompletionMessageParam
from ..core.base import BaseLLM

class OpenAIModel(BaseLLM):
    """OpenAI model implementation."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        # Set pricing based on model
        if "gpt-4" in model_name:
            self.cost_per_token = 0.00003  # $0.03 per 1K tokens
        else:  # gpt-3.5-turbo
            self.cost_per_token = 0.000002  # $0.002 per 1K tokens
        self.avg_latency = 2.0  # 2 seconds average latency

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI's completion API."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using OpenAI's completion API."""
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _validate_messages(self, messages: List[Dict[str, str]]) -> List[ChatCompletionMessageParam]:
        """Convert and validate messages to OpenAI format."""
        valid_messages = []
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")
            if msg["role"] not in ("system", "user", "assistant", "function", "tool"):
                raise ValueError(f"Invalid message role: {msg['role']}")
            valid_messages.append(cast(ChatCompletionMessageParam, {
                "role": msg["role"],
                "content": msg["content"]
            }))
        return valid_messages

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using OpenAI's chat API."""
        valid_messages = self._validate_messages(messages)
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=valid_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion using OpenAI's chat API."""
        valid_messages = self._validate_messages(messages)
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=valid_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using OpenAI's embeddings API."""
        if isinstance(text, str):
            text = [text]
            
        embedding_model = kwargs.pop('model', 'text-embedding-ada-002')
        response = await self.client.embeddings.create(
            model=embedding_model,
            input=text,
            **kwargs
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if len(text) == 1 else embeddings