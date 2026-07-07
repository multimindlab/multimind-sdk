"""
OpenAI model implementation.
"""

import os
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.exceptions import ConfigurationError
from .base import BaseLLM, resolve_images


def _is_pydantic_model(response_format: Any) -> bool:
    return isinstance(response_format, type) and issubclass(response_format, BaseModel)


def _image_content_part(resolved: Dict[str, str]) -> Dict[str, Any]:
    """Build an OpenAI-format image_url content part from a resolved image."""
    if resolved["kind"] == "url":
        url = resolved["url"]
    else:
        url = f"data:{resolved['media_type']};base64,{resolved['data']}"
    return {"type": "image_url", "image_url": {"url": url}}


class OpenAIModel(BaseLLM):
    """OpenAI model implementation."""

    PROVIDER_NAME: str = "OpenAI"
    API_KEY_ENV_VARS: Tuple[str, ...] = ("OPENAI_API_KEY",)
    BASE_URL: Optional[str] = None
    DEFAULT_EMBEDDING_MODEL: Optional[str] = "text-embedding-ada-002"
    # Set to False in subclasses whose endpoint rejects image content parts.
    SUPPORTS_VISION: bool = True

    # Approximate blended per-token USD prices; prices drift, override via cost_per_token arg.
    MODEL_PRICING: Dict[str, float] = {
        "gpt-4o-mini": 0.000000375,
        "gpt-4o": 0.00000625,
        "gpt-4.1-nano": 0.00000025,
        "gpt-4.1-mini": 0.000001,
        "gpt-4.1": 0.000005,
        "gpt-4": 0.00003,
        "gpt-3.5-turbo": 0.000002,
    }
    DEFAULT_COST_PER_TOKEN: Optional[float] = 0.00001

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_per_token: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        # Load API key from environment if not provided
        if api_key is None:
            for env_var in self.API_KEY_ENV_VARS:
                api_key = os.getenv(env_var)
                if api_key:
                    break
        if not api_key:
            env_hint = " or ".join(self.API_KEY_ENV_VARS)
            raise ConfigurationError(
                f"{self.PROVIDER_NAME} API key is not configured. "
                f"Set the {env_hint} environment variable or pass api_key explicitly."
            )
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url or self.BASE_URL)
        if cost_per_token is not None:
            self.cost_per_token = cost_per_token
        else:
            self.cost_per_token = self._lookup_cost_per_token(model_name)
        self.avg_latency = 2.0  # 2 seconds average latency

    @classmethod
    def _lookup_cost_per_token(cls, model_name: str) -> Optional[float]:
        """Longest-prefix match against the per-model pricing table."""
        for prefix in sorted(cls.MODEL_PRICING, key=len, reverse=True):
            if model_name.startswith(prefix):
                return cls.MODEL_PRICING[prefix]
        return cls.DEFAULT_COST_PER_TOKEN

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIError,
                openai.APIConnectionError,
                openai.APITimeoutError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _chat_completions_create(self, **kwargs: Any):
        """Internal helper with retry for chat.completions.create."""
        return await self.client.chat.completions.create(**kwargs)

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIError,
                openai.APIConnectionError,
                openai.APITimeoutError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _embeddings_create(self, **kwargs: Any):
        """Internal helper with retry for embeddings.create."""
        return await self.client.embeddings.create(**kwargs)

    @staticmethod
    def _schema_response_format(model_cls: Type[BaseModel]) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "schema": model_cls.model_json_schema(),
            },
        }

    @staticmethod
    def _parse_structured(model_cls: Type[BaseModel], text: str) -> BaseModel:
        try:
            return model_cls.model_validate_json(text)
        except (ValidationError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse response as {model_cls.__name__}: {exc}. Raw response: {text}"
            ) from exc

    async def _create_completion(
        self,
        messages: List[Any],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Union[Type[BaseModel], Dict[str, Any]]],
        **kwargs,
    ) -> Union[str, BaseModel]:
        """Shared non-streaming completion with optional structured output."""
        if response_format is not None:
            if _is_pydantic_model(response_format):
                kwargs["response_format"] = self._schema_response_format(response_format)
            else:
                kwargs["response_format"] = response_format
        response = await self._chat_completions_create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        if _is_pydantic_model(response_format):
            return self._parse_structured(cast(Type[BaseModel], response_format), content)
        return content

    def _resolve_vision_parts(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve image inputs to content parts, guarding vision support."""
        if not self.SUPPORTS_VISION:
            raise NotImplementedError(
                f"{self.PROVIDER_NAME} does not support image inputs on its "
                "OpenAI-compatible endpoint"
            )
        return [_image_content_part(resolved) for resolved in resolve_images(images)]

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[str, BaseModel]:
        """Generate text using OpenAI's completion API.

        ``images`` is an optional list of dicts, each with one of the keys
        ``path``, ``bytes`` or ``url`` (plus optional ``media_type``).
        """
        content: Any = prompt
        if images is not None:
            content = [{"type": "text", "text": prompt}] + self._resolve_vision_parts(images)
        return await self._create_completion(
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using OpenAI's completion API."""
        stream = await self._chat_completions_create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _validate_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[ChatCompletionMessageParam]:
        """Convert and validate messages to OpenAI format."""
        valid_messages = []
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' keys")
            if msg["role"] not in ("system", "user", "assistant", "function", "tool"):
                raise ValueError(f"Invalid message role: {msg['role']}")
            valid_messages.append(
                cast(ChatCompletionMessageParam, {"role": msg["role"], "content": msg["content"]})
            )
        return valid_messages

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        images: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[str, BaseModel]:
        """Generate chat completion using OpenAI's chat API.

        ``images`` (optional) are attached to the last user message; see
        ``generate`` for the accepted item shapes.
        """
        valid_messages = self._validate_messages(messages)
        if images is not None:
            parts = self._resolve_vision_parts(images)
            for msg in reversed(valid_messages):
                if msg["role"] == "user":
                    msg["content"] = [{"type": "text", "text": msg["content"]}] + parts
                    break
            else:
                raise ValueError("images require at least one user message")
        return await self._create_completion(
            messages=valid_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion using OpenAI's chat API."""
        valid_messages = self._validate_messages(messages)
        stream = await self._chat_completions_create(
            model=self.model_name,
            messages=valid_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embeddings(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using OpenAI's embeddings API."""
        if self.DEFAULT_EMBEDDING_MODEL is None:
            raise NotImplementedError(
                f"{self.PROVIDER_NAME} does not provide an embeddings endpoint "
                "on its OpenAI-compatible API"
            )
        if isinstance(text, str):
            text = [text]

        request_kwargs = dict(kwargs)
        embedding_model = request_kwargs.pop("model", self.DEFAULT_EMBEDDING_MODEL)
        response = await self._embeddings_create(
            model=embedding_model,
            input=text,
            **request_kwargs,
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if len(text) == 1 else embeddings
