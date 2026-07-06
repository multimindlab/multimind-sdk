"""
Models package for MultiMind SDK.
"""

from .base import BaseLLM
from .claude import ClaudeModel
from .deepseek import DeepSeekModel
from .factory import ModelFactory
from .gemini import GeminiModel
from .groq import GroqModel
from .mistral import MistralAIModel
from .multi_model import MultiModelWrapper
from .ollama import MistralModel, OllamaModel
from .openai import OpenAIModel

# Try to import HuggingFace model
try:
    from .huggingface import HuggingFaceModel

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceModel = None

__all__ = [
    "BaseLLM",
    "ModelFactory",
    "OpenAIModel",
    "ClaudeModel",
    "OllamaModel",
    "MistralModel",
    "GroqModel",
    "MistralAIModel",
    "GeminiModel",
    "DeepSeekModel",
    "MultiModelWrapper",
]

if HUGGINGFACE_AVAILABLE:
    __all__.append("HuggingFaceModel")
