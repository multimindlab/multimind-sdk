"""
Local model runner for Ollama and other local model implementations.
"""

import aiohttp
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union

from .base import BaseLLM

logger = logging.getLogger(__name__)

class LocalRunner(BaseLLM):
    """Runner for local models using Ollama."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=300)  # 5 min for slow local models
        self._session: Optional[aiohttp.ClientSession] = None
        self._headers = {"Content-Type": "application/json"}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                connector=aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    ttl_dns_cache=300,
                ),
            )
        return self._session

    def _resolve_timeout(self, timeout: Optional[Union[float, aiohttp.ClientTimeout]]) -> aiohttp.ClientTimeout:
        """Normalize timeout input into aiohttp.ClientTimeout."""
        if isinstance(timeout, aiohttp.ClientTimeout):
            return timeout
        if isinstance(timeout, (int, float)):
            return aiohttp.ClientTimeout(total=float(timeout))
        return self._timeout

    async def close(self) -> None:
        """Close the reusable HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _make_request_stream(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: Optional[Union[float, aiohttp.ClientTimeout]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Make a streaming request to the Ollama API."""
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"
        request_timeout = self._resolve_timeout(timeout)
        async with session.post(url, json=data, headers=self._headers, timeout=request_timeout) as response:
            response.raise_for_status()
            buffer = ""
            async for line in response.content:
                if not line:
                    continue
                buffer += line.decode(errors="ignore")
                for candidate in buffer.splitlines():
                    chunk = candidate.strip()
                    if not chunk:
                        continue
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        # Keep buffering if we received partial JSON.
                        break
                else:
                    # All lines decoded successfully; clear the buffer.
                    buffer = ""
                    continue
                # Preserve undecodable suffix for next chunk.
                buffer = chunk

    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: Optional[Union[float, aiohttp.ClientTimeout]] = None,
    ) -> Dict[str, Any]:
        """Make a regular request to the Ollama API with retries."""
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"
        request_timeout = self._resolve_timeout(timeout)
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with session.post(url, json=data, headers=self._headers, timeout=request_timeout) as response:
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        if last_error:
            raise last_error
        raise RuntimeError("Failed to complete request for unknown reason.")

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from the local model."""
        timeout = kwargs.pop("timeout", None)
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        response = await self._make_request("api/generate", data, timeout=timeout)
        return response.get("response", "")

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text from the local model."""
        timeout = kwargs.pop("timeout", None)
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        async for chunk in self._make_request_stream("api/generate", data, timeout=timeout):
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
        timeout = kwargs.pop("timeout", None)
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        response = await self._make_request("api/chat", data, timeout=timeout)
        return response.get("message", {}).get("content", "")

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion from the local model."""
        timeout = kwargs.pop("timeout", None)
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        if max_tokens:
            data["max_tokens"] = max_tokens

        async for chunk in self._make_request_stream("api/chat", data, timeout=timeout):
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings from the local model."""
        timeout = kwargs.pop("timeout", None)
        if isinstance(text, str):
            text = [text]

        data = {
            "model": self.model_name,
            "input": text[0] if len(text) == 1 else text,
            **kwargs
        }

        response = await self._make_request("api/embeddings", data, timeout=timeout)
        embeddings = response.get("embeddings", [])
        return embeddings[0] if len(text) == 1 else embeddings

    async def get_quality(self) -> Optional[float]:
        """Get the quality score for this model."""
        return None  # Placeholder implementation