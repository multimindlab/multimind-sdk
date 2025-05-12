"""
Local model runner for Ollama and other local model implementations.
"""

import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from .base import BaseLLM

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
        
    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        stream: bool = False
    ) -> Any:
        """Make a request to the Ollama API."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/{endpoint}"
            async with session.post(url, json=data) as response:
                if stream:
                    async for line in response.content:
                        if line:
                            yield line
                else:
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
        return response["response"]
        
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
            
        async for chunk in self._make_request("api/generate", data, stream=True):
            if chunk:
                yield chunk.decode().strip()
                
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
        return response["message"]["content"]
        
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
            
        async for chunk in self._make_request("api/chat", data, stream=True):
            if chunk:
                yield chunk.decode().strip()
                
    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings from the local model."""
        if isinstance(text, str):
            text = [text]
            
        data = {
            "model": self.model_name,
            "prompt": text[0] if len(text) == 1 else text,
            **kwargs
        }
        
        response = await self._make_request("api/embeddings", data)
        embeddings = response["embeddings"]
        return embeddings[0] if len(text) == 1 else embeddings 