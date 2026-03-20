"""
Ollama model implementation for local model running.
"""

import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import BaseLLM

class OllamaModel(BaseLLM):
    """Runner for local models using Ollama."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        # Set default cost and latency for local models
        self.cost_per_token = 0.0
        self.avg_latency = 0.1  # 100ms default latency

    async def _make_request_stream(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Make a streaming request to the Ollama API."""
        async for line in self._make_request_stream_raw(endpoint, data):
            if line:
                yield json.loads(line.decode().strip())

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _make_request_stream_raw(
        self,
        endpoint: str,
        data: Dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        """Low-level streaming request with retry for connection issues."""
        timeout = aiohttp.ClientTimeout(total=300)  # 5 min for slow local models
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/{endpoint}"
            async with session.post(url, json=data, timeout=timeout) as response:
                response.raise_for_status()
                async for line in response.content:
                    yield line

    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a regular request to the Ollama API."""
        return await self._make_request_with_retry(endpoint, data)

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    async def _make_request_with_retry(
        self,
        endpoint: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Low-level JSON request with retry for connection issues."""
        timeout = aiohttp.ClientTimeout(total=300)  # 5 min for slow local models
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/{endpoint}"
            async with session.post(url, json=data, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from the local model."""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        response = await self._make_request("api/generate", data)
        return response.get("response", "")

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text from the local model."""
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        async for chunk in self._make_request_stream("api/generate", data):
            if "response" in chunk:
                yield chunk["response"]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion from the local model."""
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        response = await self._make_request("api/chat", data)
        return response.get("message", {}).get("content", "")

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion from the local model."""
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        async for chunk in self._make_request_stream("api/chat", data):
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings from the local model."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        embeddings = []
        for t in texts:
            data = {
                "model": self.model_name,
                "prompt": t,
                **kwargs
            }
            response = await self._make_request("api/embeddings", data)
            embeddings.append(response.get("embedding", []))

        return embeddings[0] if isinstance(text, str) else embeddings


class MistralModel(OllamaModel):
    """Convenience class for Mistral models running on Ollama."""
    
    def __init__(
        self,
        model: str = "mistral",
        model_name: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        # Use model_name if provided, otherwise use model parameter
        actual_model_name = model_name if model_name is not None else model
        super().__init__(model_name=actual_model_name, base_url=base_url, **kwargs)