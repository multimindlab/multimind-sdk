import logging
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Union

from multimind.core.base import BaseLLM

logger = logging.getLogger(__name__)

# Optional torch import for non-transformer LLM features
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Non-transformer LLM features will be disabled.")

# Optional transformers import
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Non-transformer LLM features will be disabled.")

import asyncio
import warnings

import yaml

from multimind.core.chat import ChatSession

# Optional peft import
try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.debug("PEFT not available. Adapter features will be disabled.")


def _scaffold_error(cls_name: str, method: str) -> NotImplementedError:
    return NotImplementedError(
        f"{cls_name}.{method} is not implemented. {cls_name} is an extension scaffold: "
        f"subclass it and implement {method}, or use MambaLLM/RWKVLLM which are fully implemented."
    )


class NonTransformerLLM(BaseLLM):
    """Scaffold: implement generate/generate_stream/chat/chat_stream/embeddings in a subclass."""

    def __init__(self, model_name: str, model_instance: Any, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = model_instance  # This can be any non-transformer model object

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """Scaffold: implement generate in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "generate")

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Scaffold: implement generate_stream in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "generate_stream")
        yield  # unreachable; keeps this an async generator

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Scaffold: implement chat in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "chat")

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Scaffold: implement chat_stream in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "chat_stream")
        yield  # unreachable; keeps this an async generator

    async def embeddings(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Scaffold: implement embeddings in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "embeddings")

    async def get_quality(self) -> Optional[float]:
        """Get the quality score for this model."""
        return None  # Placeholder implementation


# --- Advanced Non-Transformer Architectures ---


class SSM_LLM(NonTransformerLLM):
    """Scaffold for State-Space Models (S4, Mamba): implement generate in a subclass."""

    def __init__(
        self,
        model_name: str,
        model_instance: Any,
        tokenizer: Any,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, model_instance, **kwargs)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SSM_LLM. Please install torch.")

        self.tokenizer = tokenizer
        dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.model = model_instance.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.logger = logging.getLogger("SSM_LLM")
        self.pre_hooks = []
        self.post_hooks = []
        self.adapter_path = adapter_path

    def preprocess_prompt(self, prompt: str) -> str:
        for hook in self.pre_hooks:
            prompt = hook(prompt)
        return prompt

    def postprocess_output(self, output: str) -> str:
        for hook in self.post_hooks:
            output = hook(output)
        return output

    def add_pre_hook(self, hook):
        self.pre_hooks.append(hook)

    def add_post_hook(self, hook):
        self.post_hooks.append(hook)

    def load_adapter(self, adapter_path: str):
        # Implement adapter loading for your SSM model if supported
        self.adapter_path = adapter_path

    def unload_adapter(self):
        # Implement adapter unloading for your SSM model if supported
        self.adapter_path = None

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        # Example: batch process prompts (user must implement details for their SSM)
        return [
            await self.generate(p, temperature=temperature, max_tokens=max_tokens, **kwargs)
            for p in prompts
        ]

    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        # Avoid `asyncio.run()` (nested event loops). We are already inside async code,
        # so directly gather coroutines on the current event loop.
        return await asyncio.gather(*[self.generate(p, **kwargs) for p in prompts])

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Scaffold: implement generate_stream in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "generate_stream")
        yield  # unreachable; keeps this an async generator

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Scaffold: implement chat_stream in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "chat_stream")
        yield  # unreachable; keeps this an async generator

    def new_chat_session(self, persona: Optional[str] = None, max_history: int = 10) -> ChatSession:
        return ChatSession(persona=persona, max_history=max_history)

    def log_metric(self, name: str, value: float):
        self.logger.info(f"Metric: {name} = {value}")

    def log_generation(self, prompt: str, output: str):
        self.logger.info(f"Prompt: {prompt}\nOutput: {output}")

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """Scaffold: implement generate in a subclass."""
        raise _scaffold_error(self.__class__.__name__, "generate")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        session: Optional[ChatSession] = None,
        **kwargs,
    ) -> str:
        if session is not None:
            for m in messages:
                session.add_message(m["role"], m["content"])
            prompt = session.get_prompt()
        else:
            prompt = "\n".join([m["content"] for m in messages])
        return await self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)


class MLPOnlyLLM(NonTransformerLLM):
    """Scaffold for MLP-Only models (HyperMixer, gMLP, MLP-Mixer): implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class DiffusionTextLLM(NonTransformerLLM):
    """Scaffold for diffusion text models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class MoELLMMixin(NonTransformerLLM):
    """Scaffold for Mixture-of-Experts models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class PerceiverLLM(NonTransformerLLM):
    """Scaffold for Perceiver/Perceiver IO models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


# --- Advanced Sequence Model Wrappers ---


class MegaS4LLM(NonTransformerLLM):
    """Scaffold for Mega-S4 models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class LiquidS4LLM(NonTransformerLLM):
    """Scaffold for Liquid-S4 models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class S4DLLM(NonTransformerLLM):
    """Scaffold for S4D (diagonal S4) models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class S4NDLLM(NonTransformerLLM):
    """Scaffold for S4ND (non-diagonal S4) models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class DSSLLM(NonTransformerLLM):
    """Scaffold for DSS (Diagonal State Space) models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class GSSLLM(NonTransformerLLM):
    """Scaffold for GSS (General State Space) models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class ChatSession:
    """
    Advanced chat session for memory, context window, persona/system prompt.
    """

    def __init__(self, persona: Optional[str] = None, max_history: int = 10):
        self.persona = persona
        self.max_history = max_history
        self.history = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_prompt(self):
        prompt = (self.persona + "\n") if self.persona else ""
        prompt += "\n".join([m["content"] for m in self.history])
        return prompt


class MambaLLM(NonTransformerLLM):
    """
    Advanced wrapper for HuggingFace/state-spaces Mamba models with all advanced features.
    """

    def __init__(
        self,
        model_name: str = "state-spaces/mamba-130m",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, None, **kwargs)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MambaLLM. Please install torch.")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required for MambaLLM. Please install transformers.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        )
        if adapter_path and PEFT_AVAILABLE:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except Exception:
                pass
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.logger = logging.getLogger("MambaLLM")
        self.pre_hooks = []
        self.post_hooks = []
        self.adapter_path = adapter_path

    # --- Pre/Post-processing hooks ---
    def preprocess_prompt(self, prompt: str) -> str:
        for hook in self.pre_hooks:
            prompt = hook(prompt)
        return prompt

    def postprocess_output(self, output: str) -> str:
        for hook in self.post_hooks:
            output = hook(output)
        return output

    def add_pre_hook(self, hook):
        self.pre_hooks.append(hook)

    def add_post_hook(self, hook):
        self.post_hooks.append(hook)

    # --- Adapter hot-swapping ---
    def load_adapter(self, adapter_path: str):
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.adapter_path = adapter_path

    def unload_adapter(self):
        if self.adapter_path:
            # Reload base model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.adapter_path = None

    # --- Batch generation ---
    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        gen_kwargs = {"temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # --- Async/parallel batch generation ---
    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        # Avoid `asyncio.run()` (nested event loops). We are already inside async code,
        # so directly gather coroutines on the current event loop.
        return await asyncio.gather(*[self.generate(p, **kwargs) for p in prompts])

    # --- Streaming generation ---
    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        prompt = self.preprocess_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {"temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            self.model.generate(**inputs, streamer=streamer, **gen_kwargs)
        # TextStreamer yields to stdout, so for real streaming, use a custom streamer or yield tokens here
        # For now, just yield the full output
        output = self.model.generate(**inputs, **gen_kwargs)
        yield self.tokenizer.decode(output[0], skip_special_tokens=True)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        prompt = "\n".join([m["content"] for m in messages])
        async for chunk in self.generate_stream(
            prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        ):
            yield chunk

    # --- Advanced chat memory/history ---
    def new_chat_session(self, persona: Optional[str] = None, max_history: int = 10) -> ChatSession:
        return ChatSession(persona=persona, max_history=max_history)

    # --- Precision management (fp16/bf16) and distributed ---
    # For multi-GPU/distributed, recommend using accelerate/deepspeed externally
    # Example: pass device_map="auto" and torch_dtype="float16" to __init__

    # --- Evaluation/logging hooks ---
    def log_metric(self, name: str, value: float):
        self.logger.info(f"Metric: {name} = {value}")

    def log_generation(self, prompt: str, output: str):
        self.logger.info(f"Prompt: {prompt}\nOutput: {output}")

    # --- Config-driven instantiation ---
    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    # --- Override generate to use hooks and logging ---
    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        prompt = self.preprocess_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {"temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        result = self.postprocess_output(result)
        self.log_generation(prompt, result)
        return result

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        session: Optional[ChatSession] = None,
        **kwargs,
    ) -> str:
        if session is not None:
            for m in messages:
                session.add_message(m["role"], m["content"])
            prompt = session.get_prompt()
        else:
            prompt = "\n".join([m["content"] for m in messages])
        return await self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)


class MoEMambaLLM(NonTransformerLLM):
    """Scaffold for MoE-Mamba models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class H3LLM(NonTransformerLLM):
    """Scaffold for H3 (Hyena Hybrid) models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class RetNetLLM(NonTransformerLLM):
    """Scaffold for RetNet (Retentive Network) models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class RWKVLLM(NonTransformerLLM):
    """
    Advanced wrapper for BlinkDL/rwkv-4-pile-169m (HuggingFace) with all advanced features.
    """

    def __init__(
        self,
        model_name: str = "BlinkDL/rwkv-4-pile-169m",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, None, **kwargs)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RWKVLLM. Please install torch.")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required for RWKVLLM. Please install transformers.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        )
        if adapter_path and PEFT_AVAILABLE:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except Exception:
                pass
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.logger = logging.getLogger("RWKVLLM")
        self.pre_hooks = []
        self.post_hooks = []
        self.adapter_path = adapter_path

    def preprocess_prompt(self, prompt: str) -> str:
        for hook in self.pre_hooks:
            prompt = hook(prompt)
        return prompt

    def postprocess_output(self, output: str) -> str:
        for hook in self.post_hooks:
            output = hook(output)
        return output

    def add_pre_hook(self, hook):
        self.pre_hooks.append(hook)

    def add_post_hook(self, hook):
        self.post_hooks.append(hook)

    def load_adapter(self, adapter_path: str):
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.adapter_path = adapter_path

    def unload_adapter(self):
        if self.adapter_path:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.adapter_path = None

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        gen_kwargs = {"temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        # Avoid `asyncio.run()` (nested event loops). We are already inside async code,
        # so directly gather coroutines on the current event loop.
        return await asyncio.gather(*[self.generate(p, **kwargs) for p in prompts])

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        prompt = self.preprocess_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {"temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            self.model.generate(**inputs, streamer=streamer, **gen_kwargs)
        output = self.model.generate(**inputs, **gen_kwargs)
        yield self.tokenizer.decode(output[0], skip_special_tokens=True)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        prompt = "\n".join([m["content"] for m in messages])
        async for chunk in self.generate_stream(
            prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        ):
            yield chunk

    def new_chat_session(self, persona: Optional[str] = None, max_history: int = 10) -> ChatSession:
        return ChatSession(persona=persona, max_history=max_history)

    def log_metric(self, name: str, value: float):
        self.logger.info(f"Metric: {name} = {value}")

    def log_generation(self, prompt: str, output: str):
        self.logger.info(f"Prompt: {prompt}\nOutput: {output}")

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        prompt = self.preprocess_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {"temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        result = self.postprocess_output(result)
        self.log_generation(prompt, result)
        return result

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        session: Optional[ChatSession] = None,
        **kwargs,
    ) -> str:
        if session is not None:
            for m in messages:
                session.add_message(m["role"], m["content"])
            prompt = session.get_prompt()
        else:
            prompt = "\n".join([m["content"] for m in messages])
        return await self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)


class SE3HyenaLLM(NonTransformerLLM):
    """Scaffold for SE(3)-Hyena models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class TopologicalNNLLM(NonTransformerLLM):
    """Scaffold for topological deep learning models: implement generate in a subclass."""

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


class CustomRNNLLM(NonTransformerLLM):
    """
    Advanced template for custom RNN/MLP models (PyTorch/Keras) with all advanced features.
    """

    def __init__(
        self,
        model_instance: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        **kwargs,
    ):
        super().__init__("custom-rnn", model_instance, **kwargs)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CustomRNNLLM. Please install torch.")

        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_instance.to(self.device)
        self.model.eval()
        self.logger = logging.getLogger("CustomRNNLLM")
        self.pre_hooks = []
        self.post_hooks = []
        self.adapter_path = None

    def preprocess_prompt(self, prompt: str) -> str:
        for hook in self.pre_hooks:
            prompt = hook(prompt)
        return prompt

    def postprocess_output(self, output: str) -> str:
        for hook in self.post_hooks:
            output = hook(output)
        return output

    def add_pre_hook(self, hook):
        self.pre_hooks.append(hook)

    def add_post_hook(self, hook):
        self.post_hooks.append(hook)

    def load_adapter(self, adapter_path: str):
        # Implement adapter loading for your RNN/MLP model if supported
        self.adapter_path = adapter_path

    def unload_adapter(self):
        # Implement adapter unloading for your RNN/MLP model if supported
        self.adapter_path = None

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        return [
            await self.generate(p, temperature=temperature, max_tokens=max_tokens, **kwargs)
            for p in prompts
        ]

    async def generate_batch_async(self, prompts: List[str], **kwargs) -> List[str]:
        # Avoid `asyncio.run()` (nested event loops). We are already inside async code,
        # so directly gather coroutines on the current event loop.
        return await asyncio.gather(*[self.generate(p, **kwargs) for p in prompts])

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        yield await self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        prompt = "\n".join([m["content"] for m in messages])
        async for chunk in self.generate_stream(
            prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        ):
            yield chunk

    def new_chat_session(self, persona: Optional[str] = None, max_history: int = 10) -> ChatSession:
        return ChatSession(persona=persona, max_history=max_history)

    def log_metric(self, name: str, value: float):
        self.logger.info(f"Metric: {name} = {value}")

    def log_generation(self, prompt: str, output: str):
        self.logger.info(f"Prompt: {prompt}\nOutput: {output}")

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        prompt = self.preprocess_prompt(prompt)
        # Assume model_instance has a 'generate' method and tokenizer has 'encode' and 'decode'
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # For demonstration, use model's generate or forward method
        with torch.no_grad():
            if hasattr(self.model, "generate"):
                output_ids = self.model.generate(
                    input_ids, max_length=max_tokens or 64, temperature=temperature
                )
            else:
                output_ids = self.model(input_ids)
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        result = self.postprocess_output(output)
        self.log_generation(prompt, result)
        return result

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        session: Optional[ChatSession] = None,
        **kwargs,
    ) -> str:
        # Concatenate messages for prompt
        if session is not None:
            for m in messages:
                session.add_message(m["role"], m["content"])
            prompt = session.get_prompt()
        else:
            prompt = "\n".join([m["content"] for m in messages])
        return await self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)


# --- Adapter management for per-user/session/tool injection ---
class AdapterManager:
    """
    Manages adapters per user/session/tool. Used by advanced LLM wrappers for dynamic LoRA/PEFT injection.
    """

    def __init__(self):
        self.adapters = {}  # key -> adapter_path

    def set_adapter(self, key, adapter_path):
        self.adapters[key] = adapter_path

    def get_adapter(self, key):
        return self.adapters.get(key)

    def remove_adapter(self, key):
        if key in self.adapters:
            del self.adapters[key]


# Patch advanced LLMs to support per-user/session/tool adapter injection
for _LLM in [MambaLLM, H3LLM, RWKVLLM, SSM_LLM, CustomRNNLLM]:
    _LLM.adapter_manager = AdapterManager()

    def load_adapter_for(self, key, adapter_path):
        self.adapter_manager.set_adapter(key, adapter_path)

    def unload_adapter_for(self, key):
        self.adapter_manager.remove_adapter(key)

    def get_active_adapter(self, key):
        return self.adapter_manager.get_adapter(key)

    _LLM.load_adapter_for = load_adapter_for
    _LLM.unload_adapter_for = unload_adapter_for
    _LLM.get_active_adapter = get_active_adapter
    # Patch generate/chat to use adapter if set for key
    orig_generate = _LLM.generate

    async def generate_with_adapter(self, prompt, *args, adapter_key=None, **kwargs):
        adapter_path = self.get_active_adapter(adapter_key) if adapter_key else None
        if adapter_path:
            try:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError:
                warnings.warn("peft is not installed; skipping adapter loading.")
        return await orig_generate(self, prompt, *args, **kwargs)

    _LLM.generate = generate_with_adapter
    orig_chat = _LLM.chat

    async def chat_with_adapter(self, messages, *args, adapter_key=None, **kwargs):
        adapter_path = self.get_active_adapter(adapter_key) if adapter_key else None
        if adapter_path:
            try:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError:
                warnings.warn("peft is not installed; skipping adapter loading.")
        return await orig_chat(self, messages, *args, **kwargs)

    _LLM.chat = chat_with_adapter

# --- Advanced/Optional Features (TODO Stubs) ---


class QLoRALLM(NonTransformerLLM):
    """
    QLoRA wrapper: reloads the base HuggingFace model in 4-bit (bitsandbytes NF4)
    and attaches LoRA adapters via peft. Requires the [finetune-gpu] extra
    (torch, transformers, peft, bitsandbytes); raises NotImplementedError without it.
    """

    def __init__(
        self,
        base_llm,
        *args,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(base_llm.model_name, *args, **kwargs)
        self.base_llm = base_llm
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.peft_model = None

    def _require_deps(self):
        try:
            import bitsandbytes  # noqa: F401
            import peft
            import torch
            import transformers
        except ImportError as e:
            raise NotImplementedError(
                "QLoRALLM requires torch, transformers, peft, and bitsandbytes. "
                "Install with: pip install 'multimind-sdk[finetune-gpu]'"
            ) from e
        return torch, transformers, peft

    def apply_qlora(self, **load_kwargs):
        """Reload the base model 4-bit quantized and wrap it with LoRA adapters."""
        torch, transformers, peft = self._require_deps()
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=load_kwargs.pop("device_map", "auto"),
            **load_kwargs,
        )
        model = peft.prepare_model_for_kbit_training(model)
        lora_config = peft.LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = peft.get_peft_model(model, lora_config)
        return self.peft_model

    def _get_tokenizer(self):
        tokenizer = getattr(self.base_llm, "tokenizer", None)
        if tokenizer is None:
            raise NotImplementedError(
                "QLoRALLM requires a base LLM exposing a HuggingFace tokenizer "
                "(e.g. MambaLLM or RWKVLLM)."
            )
        return tokenizer

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        torch, _, _ = self._require_deps()
        if self.peft_model is None:
            self.apply_qlora()
        tokenizer = self._get_tokenizer()
        inputs = tokenizer(prompt, return_tensors="pt").to(self.peft_model.device)
        gen_kwargs = {"temperature": temperature}
        if max_tokens:
            gen_kwargs["max_new_tokens"] = max_tokens
        with torch.no_grad():
            output = self.peft_model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)


class CompacterLLM(NonTransformerLLM):
    """
    Scaffold for Compacter parameter-efficient tuning: implement generate in a subclass.
    Note: peft does not ship a Compacter config (Compacter lives in the separate
    `adapters` library), so this stays a fail-honest scaffold.
    """

    def __init__(self, base_llm, *args, **kwargs):
        super().__init__(base_llm.model_name, *args, **kwargs)
        self.base_llm = base_llm

    async def generate(self, prompt: str, **kwargs) -> str:
        raise _scaffold_error(self.__class__.__name__, "generate")


# TODO: Model merging capabilities
# TODO: Advanced quantization support
# TODO: GPU acceleration and distributed processing
# TODO: Advanced CLI/API features (streaming, profiles, chat session switching)
# TODO: Vector store migration/optimization tools
