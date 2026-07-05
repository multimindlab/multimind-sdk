"""
HuggingFace model implementation for local model loading.
"""

import asyncio
import functools
from collections.abc import AsyncGenerator
from threading import Thread
from typing import Dict, List, Optional, Union

from .base import BaseLLM

# Try to import transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceModel(BaseLLM):
    """HuggingFace model implementation for local loading."""

    def __init__(
        self, model_name: str, api_key: Optional[str] = None, device: Optional[str] = None, **kwargs
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
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
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
            generated_text = generated_text[len(prompt) :].strip()

        return generated_text

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """Generate text from the model."""
        loop = asyncio.get_running_loop()
        generate_fn = functools.partial(
            self._generate_text, prompt, temperature, max_tokens, **kwargs
        )
        return await loop.run_in_executor(None, generate_fn)

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens or 100,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update({k: v for k, v in kwargs.items() if k not in inputs})

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        generation_error: List[Exception] = []

        def _run_generation() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(**inputs, **gen_kwargs)
            except Exception as e:  # pragma: no cover - surfaced after streaming loop
                generation_error.append(e)

        generation_thread = Thread(target=_run_generation, daemon=True)
        generation_thread.start()

        loop = asyncio.get_running_loop()
        iterator = iter(streamer)

        while True:
            try:
                chunk = await loop.run_in_executor(None, next, iterator)
            except StopIteration:
                break
            if chunk:
                yield chunk

        await loop.run_in_executor(None, generation_thread.join)
        if generation_error:
            raise generation_error[0]

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
        **kwargs,
    ) -> str:
        """Generate chat completion from the model."""
        prompt = self._messages_to_prompt(messages)
        return await self.generate(prompt, temperature, max_tokens, **kwargs)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion from the model."""
        prompt = self._messages_to_prompt(messages)
        async for chunk in self.generate_stream(prompt, temperature, max_tokens, **kwargs):
            yield chunk

    def _compute_embeddings(self, texts: List[str], max_length: int = 512) -> List[List[float]]:
        """Compute embeddings using mean pooling over the last hidden state."""
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        last_hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
        masked_hidden = last_hidden * attention_mask
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = masked_hidden.sum(dim=1) / token_counts

        return pooled.cpu().tolist()

    async def embeddings(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        texts = [text] if isinstance(text, str) else text
        max_length = kwargs.get("max_length", 512)

        loop = asyncio.get_running_loop()
        compute_fn = functools.partial(self._compute_embeddings, texts, max_length)
        embeddings = await loop.run_in_executor(None, compute_fn)

        if isinstance(text, str):
            return embeddings[0]
        return embeddings
