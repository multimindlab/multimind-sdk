"""
Base class for all LLM implementations.
"""

import asyncio
import base64
import mimetypes
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Coroutine
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, final

_IMAGE_SOURCE_KEYS = ("path", "bytes", "url")


def _sniff_image_media_type(data: bytes) -> Optional[str]:
    """Detect the media type of raw image bytes from magic numbers."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def resolve_image(image: Dict[str, Any]) -> Dict[str, str]:
    """Normalize one image input to a provider-agnostic form.

    ``image`` must be a dict with exactly one of the keys ``path``, ``bytes``
    or ``url`` (plus an optional ``media_type``). Returns either
    ``{"kind": "url", "url": ...}`` or
    ``{"kind": "base64", "media_type": ..., "data": <base64 str>}``.
    """
    if not isinstance(image, dict):
        raise ValueError(
            f"Each image must be a dict with one of {_IMAGE_SOURCE_KEYS}, got {type(image).__name__}"
        )
    present = [key for key in _IMAGE_SOURCE_KEYS if key in image]
    if len(present) != 1:
        raise ValueError(f"Each image must have exactly one of {_IMAGE_SOURCE_KEYS}, got {present}")
    source_key = present[0]
    media_type = image.get("media_type")

    if source_key == "url":
        return {"kind": "url", "url": str(image["url"])}

    if source_key == "path":
        path = Path(image["path"])
        data = path.read_bytes()
        media_type = media_type or mimetypes.guess_type(str(path))[0]
    else:
        data = image["bytes"]
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"'bytes' image source must be bytes, got {type(data).__name__}")
        data = bytes(data)

    media_type = media_type or _sniff_image_media_type(data)
    if not media_type:
        raise ValueError(
            "Could not determine image media type; pass an explicit 'media_type' (e.g. 'image/png')"
        )
    return {
        "kind": "base64",
        "media_type": media_type,
        "data": base64.b64encode(data).decode("ascii"),
    }


def resolve_images(images: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize a list of image inputs via :func:`resolve_image`."""
    return [resolve_image(image) for image in images]


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.cost_per_token: Optional[float] = None
        self.avg_latency: Optional[float] = None

    @abstractmethod
    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """Generate text from the model."""
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text stream from the model."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate chat completion from the model."""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion stream from the model."""
        pass

    @abstractmethod
    async def embeddings(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        pass

    @final
    def _run_sync(self, coro: Coroutine[Any, Any, Any], method_name: str) -> Any:
        """Run an async method synchronously; error out inside a running loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        coro.close()
        raise RuntimeError(
            f"{method_name}_sync() cannot be called from a running event loop; "
            f"await the async {method_name}() method instead."
        )

    @final
    def generate_sync(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> Any:
        """Synchronous wrapper around generate()."""
        return self._run_sync(
            self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs),
            "generate",
        )

    @final
    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Synchronous wrapper around chat()."""
        return self._run_sync(
            self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs),
            "chat",
        )

    @final
    def embeddings_sync(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Synchronous wrapper around embeddings()."""
        return self._run_sync(self.embeddings(text, **kwargs), "embeddings")

    async def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the cost of a request based on token usage."""
        if self.cost_per_token is None:
            return 0.0
        return (prompt_tokens + completion_tokens) * self.cost_per_token

    async def get_latency(self) -> Optional[float]:
        """Get the average latency for this model."""
        return self.avg_latency

    def get_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities for routing and selection."""
        return {
            "supported_tasks": ["text_generation", "chat", "embeddings"],
            "max_complexity": 10,
            "supported_domains": ["general"],
            "supported_languages": ["en"],
            "max_context_length": 4096,
            "model_type": "transformer",
            "supports_streaming": True,
            "supports_fine_tuning": False,
        }
