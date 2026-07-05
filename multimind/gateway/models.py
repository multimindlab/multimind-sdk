"""
Model handlers for different AI providers in the MultiMind Gateway
"""

import asyncio
import logging
from typing import Dict, List

import anthropic
import httpx
import openai

# Try to import HuggingFace dependencies
try:
    from huggingface_hub import InferenceClient

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..core.models import ModelHandler, ModelResponse
from .config import ModelConfig, config

logger = logging.getLogger(__name__)


class OpenAIHandler(ModelHandler):
    """Handler for OpenAI models"""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self._client = openai.AsyncOpenAI(api_key=self.config.api_key)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )

            # Fix typing issues
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                content = response.choices[0].message.content
            else:
                content = ""

            # Handle None cases for token attributes
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                total_tokens = response.usage.total_tokens or 0
            else:
                prompt_tokens = completion_tokens = total_tokens = 0

            return ModelResponse(
                content=content,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)


class AnthropicHandler(ModelHandler):
    """Handler for Anthropic models"""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self._client = anthropic.AsyncAnthropic(api_key=self.config.api_key)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            # Convert messages to Anthropic format
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            response = await self._client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )

            return ModelResponse(
                content=response.content[0].text if hasattr(response.content[0], "text") else "",
                model=self.config.model_name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)


class OllamaHandler(ModelHandler):
    """Handler for Ollama models (async HTTP via httpx)."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        timeout = self.config.timeout if hasattr(self.config, "timeout") else 30
        self._client = httpx.AsyncClient(
            base_url=str(self.config.api_base).rstrip("/"),
            timeout=timeout,
        )

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            response = await self._client.post(
                "/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                },
            )
            response.raise_for_status()
            result = response.json()

            return ModelResponse(
                content=result.get("response", ""),
                model=self.config.model_name,
                finish_reason="stop" if result.get("done", True) else None,
            )
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)


class HuggingFaceHandler(ModelHandler):
    """Handler for HuggingFace models - supports both API and local loading"""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.use_local = not self.config.api_key or self.config.api_key.strip() == ""

        if self.use_local:
            # Use local transformers model
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers and PyTorch are required for local HuggingFace models. "
                    "Install with: pip install transformers torch"
                )

            logger.info(f"Loading HuggingFace model locally: {self.config.model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer and model
            hf_token = self.config.api_key if self.config.api_key else None
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, token=hf_token)

            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, token=hf_token
            )
            self.model.to(device)
            self.model.eval()
            self.device = device

            logger.info(f"HuggingFace model loaded successfully on {device}")
        else:
            # Use HuggingFace Inference API
            if not HF_HUB_AVAILABLE:
                raise ImportError(
                    "huggingface_hub is required for HuggingFace API. "
                    "Install with: pip install huggingface_hub"
                )

            logger.info(f"Using HuggingFace Inference API for: {self.config.model_name}")
            self._client = InferenceClient(model=self.config.model_name, token=self.config.api_key)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            if self.use_local:
                return await self._chat_local(messages, **kwargs)
            else:
                return await self._chat_api(messages, **kwargs)
        except Exception as e:
            logger.error(f"HuggingFace error: {str(e)}")
            raise

    async def _chat_local(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using local transformers model"""
        # Convert messages to prompt format
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            self._generate_local,
            prompt,
            kwargs.get("temperature", self.config.temperature),
            kwargs.get("max_tokens", self.config.max_tokens or 200),
        )

        return ModelResponse(content=response, model=self.config.model_name)

    def _generate_local(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using local model (runs in executor)"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (generated part)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        return generated_text

    async def _chat_api(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using HuggingFace Inference API"""
        # Convert messages to prompt format
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        response = await self._client.text_generation(
            prompt,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            return_full_text=False,
        )

        return ModelResponse(content=response, model=self.config.model_name)

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)


def get_model_handler(model_name: str) -> ModelHandler:
    """Factory function to get the appropriate model handler"""
    model_map = {
        "openai": OpenAIHandler,
        "anthropic": AnthropicHandler,
        "ollama": OllamaHandler,
        "huggingface": HuggingFaceHandler,
    }

    handler_class = model_map.get(model_name.lower())
    if not handler_class:
        raise ValueError(f"Unsupported model: {model_name}")

    model_config = config.get_model_config(model_name)
    return handler_class(model_config)
