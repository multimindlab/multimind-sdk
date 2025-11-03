"""
HuggingFace model implementation for local model loading.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from .base import BaseLLM

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceModel(BaseLLM):
    """HuggingFace model implementation for local loading."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "distilgpt2")
            api_key: Optional API key for gated models (None for public models)
            device: Device to use ("cpu" or "cuda"), auto-detected if None
            **kwargs: Additional arguments
        """
        super().__init__(model_name, **kwargs)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers and PyTorch are required for HuggingFace models. "
                "Install with: pip install transformers torch"
            )
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=api_key, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        
        # Set cost and latency for local models
        self.cost_per_token = 0.0  # Free for local models
        self.avg_latency = 0.5  # 500ms default latency

    def _generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text synchronously (runs in executor)."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare generation kwargs
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens or 100,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        # Only add temperature if sampling is enabled
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature
        # Add any additional kwargs (excluding inputs)
        gen_kwargs.update({k: v for k, v in kwargs.items() if k not in inputs})
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from the model."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_text,
            prompt,
            temperature,
            max_tokens,
            **kwargs
        )

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text from the model."""
        # For now, generate full text and yield it in chunks
        # TODO: Implement proper token-by-token streaming
        full_text = await self.generate(prompt, temperature, max_tokens, **kwargs)
        
        # Yield in chunks for streaming effect
        chunk_size = 10
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i:i + chunk_size]
            await asyncio.sleep(0.01)  # Small delay for streaming effect

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion from the model."""
        prompt = self._messages_to_prompt(messages)
        return await self.generate(prompt, temperature, max_tokens, **kwargs)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion from the model."""
        prompt = self._messages_to_prompt(messages)
        async for chunk in self.generate_stream(prompt, temperature, max_tokens, **kwargs):
            yield chunk

    async def embeddings(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        # Use the model's embedding layer if available
        # For now, return a placeholder - embeddings should use a dedicated embedding model
        if isinstance(text, str):
            # Return a dummy embedding vector (768 dimensions)
            return [0.0] * 768
        else:
            return [[0.0] * 768 for _ in text]

