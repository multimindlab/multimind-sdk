"""
Anthropic Claude model implementation.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from anthropic import AsyncAnthropic
from ..core.base import BaseLLM

class ClaudeModel(BaseLLM):
    """Anthropic Claude model implementation."""
    
    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)
        
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using Claude's completion API."""
        response = await self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.content[0].text
        
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using Claude's completion API."""
        stream = await self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
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
        response = await self.client.messages.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.content[0].text
        
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion using Claude's chat API."""
        stream = await self.client.messages.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
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