"""
Advanced Model Adapters for Context Transfer

Provides comprehensive model-specific adapters for formatting conversation context
for different LLM providers across the entire AI ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json


class ModelAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.supported_formats = ["text", "markdown", "json"]
        self.max_context_length = 8000  # Default token limit
    
    @abstractmethod
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        """
        Format conversation context for this specific model.
        
        Args:
            summary: Conversation summary
            source_model: Source model name
            **kwargs: Additional formatting options
            
        Returns:
            Formatted prompt for this model
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the base system prompt for this model.
        
        Returns:
            System prompt string
        """
        pass
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata and capabilities."""
        return {
            "name": self.model_name,
            "supported_formats": self.supported_formats,
            "max_context_length": self.max_context_length,
            "supports_code": True,
            "supports_images": False,
            "supports_tools": False
        }


class DeepSeekAdapter(ModelAdapter):
    """Advanced adapter for DeepSeek models."""
    
    def __init__(self):
        super().__init__("DeepSeek")
        self.max_context_length = 32000
        self.supports_code = True
        self.supports_tools = True
    
    def get_system_prompt(self) -> str:
        return "You are DeepSeek, an advanced AI assistant with expertise in coding, reasoning, and problem-solving."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_code_context = kwargs.get('include_code_context', True)
        include_reasoning = kwargs.get('include_reasoning', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue helping the user from where they left off. Maintain the context and provide helpful, accurate responses."""

        if include_code_context:
            prompt += "\n\nIf the conversation involves code, maintain the same programming language and style."
        
        if include_reasoning:
            prompt += "\n\nProvide clear reasoning for your responses when appropriate."
        
        return prompt


class ClaudeAdapter(ModelAdapter):
    """Advanced adapter for Claude models."""
    
    def __init__(self):
        super().__init__("Claude")
        self.max_context_length = 200000
        self.supports_code = True
        self.supports_tools = True
    
    def get_system_prompt(self) -> str:
        return "You are Claude, an AI assistant by Anthropic, designed to be helpful, harmless, and honest."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_safety = kwargs.get('include_safety', True)
        include_ethics = kwargs.get('include_ethics', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue assisting the user from where they left off. Maintain the conversation context and provide thoughtful, helpful responses."""

        if include_safety:
            prompt += "\n\nAlways prioritize safety and ethical considerations in your responses."
        
        if include_ethics:
            prompt += "\n\nIf the conversation involves potentially harmful content, provide guidance on safer alternatives."
        
        return prompt


class ChatGPTAdapter(ModelAdapter):
    """Advanced adapter for ChatGPT models."""
    
    def __init__(self):
        super().__init__("ChatGPT")
        self.max_context_length = 128000
        self.supports_code = True
        self.supports_tools = True
    
    def get_system_prompt(self) -> str:
        return "You are ChatGPT, an AI assistant by OpenAI, designed to help with a wide range of tasks."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_creativity = kwargs.get('include_creativity', True)
        include_examples = kwargs.get('include_examples', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue helping the user from where they left off. Maintain the context and provide helpful responses."""

        if include_creativity:
            prompt += "\n\nFeel free to be creative and provide innovative solutions when appropriate."
        
        if include_examples:
            prompt += "\n\nWhen helpful, provide concrete examples to illustrate your points."
        
        return prompt


class GeminiAdapter(ModelAdapter):
    """Advanced adapter for Gemini models."""
    
    def __init__(self):
        super().__init__("Gemini")
        self.max_context_length = 1000000
        self.supports_code = True
        self.supports_images = True
        self.supports_tools = True
    
    def get_system_prompt(self) -> str:
        return "You are Gemini, an AI assistant by Google, capable of understanding and generating text, code, and images."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_multimodal = kwargs.get('include_multimodal', True)
        include_web_search = kwargs.get('include_web_search', False)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue helping the user from where they left off. Maintain the context and provide accurate, helpful responses."""

        if include_multimodal:
            prompt += "\n\nYou can handle text, code, and image content as needed."
        
        if include_web_search:
            prompt += "\n\nIf current information is needed, you can search the web for the latest data."
        
        return prompt


class MistralAdapter(ModelAdapter):
    """Advanced adapter for Mistral models."""
    
    def __init__(self):
        super().__init__("Mistral")
        self.max_context_length = 32000
        self.supports_code = True
        self.supports_tools = True
    
    def get_system_prompt(self) -> str:
        return "You are Mistral, an AI assistant designed for reasoning, coding, and problem-solving."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_reasoning = kwargs.get('include_reasoning', True)
        include_step_by_step = kwargs.get('include_step_by_step', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue assisting the user from where they left off. Maintain the conversation context and provide helpful responses."""

        if include_reasoning:
            prompt += "\n\nProvide clear reasoning and step-by-step explanations when solving complex problems."
        
        if include_step_by_step:
            prompt += "\n\nBreak down complex tasks into manageable steps when appropriate."
        
        return prompt


class LlamaAdapter(ModelAdapter):
    """Advanced adapter for Llama models."""
    
    def __init__(self):
        super().__init__("Llama")
        self.max_context_length = 4096
        self.supports_code = True
    
    def get_system_prompt(self) -> str:
        return "You are Llama, an AI assistant designed to be helpful and informative."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_simplicity = kwargs.get('include_simplicity', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue helping the user from where they left off. Maintain the context and provide helpful responses."""

        if include_simplicity:
            prompt += "\n\nKeep responses clear and straightforward."
        
        return prompt


class CohereAdapter(ModelAdapter):
    """Advanced adapter for Cohere models."""
    
    def __init__(self):
        super().__init__("Cohere")
        self.max_context_length = 2048
        self.supports_code = True
    
    def get_system_prompt(self) -> str:
        return "You are Cohere, an AI assistant focused on natural language understanding and generation."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_natural = kwargs.get('include_natural', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue helping the user from where they left off. Maintain the context and provide natural, helpful responses."""

        if include_natural:
            prompt += "\n\nUse natural, conversational language in your responses."
        
        return prompt


class AnthropicClaudeAdapter(ModelAdapter):
    """Advanced adapter for Anthropic's Claude models."""
    
    def __init__(self):
        super().__init__("AnthropicClaude")
        self.max_context_length = 200000
        self.supports_code = True
        self.supports_tools = True
    
    def get_system_prompt(self) -> str:
        return "You are Claude, an AI assistant by Anthropic, designed to be helpful, harmless, and honest."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_constitutional = kwargs.get('include_constitutional', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue assisting the user from where they left off. Maintain the conversation context and provide thoughtful, helpful responses."""

        if include_constitutional:
            prompt += "\n\nFollow constitutional AI principles: be helpful, harmless, and honest."
        
        return prompt


class OpenAIGPT4Adapter(ModelAdapter):
    """Advanced adapter for OpenAI GPT-4 models."""
    
    def __init__(self):
        super().__init__("OpenAIGPT4")
        self.max_context_length = 128000
        self.supports_code = True
        self.supports_tools = True
        self.supports_images = True
    
    def get_system_prompt(self) -> str:
        return "You are GPT-4, an advanced AI assistant by OpenAI, capable of understanding and generating text, code, and images."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        include_advanced_reasoning = kwargs.get('include_advanced_reasoning', True)
        include_creativity = kwargs.get('include_creativity', True)
        
        prompt = f"""{self.get_system_prompt()}

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue helping the user from where they left off. Maintain the context and provide helpful, accurate responses."""

        if include_advanced_reasoning:
            prompt += "\n\nUse advanced reasoning capabilities to provide comprehensive solutions."
        
        if include_creativity:
            prompt += "\n\nLeverage creative problem-solving when appropriate."
        
        return prompt


class AdapterFactory:
    """
    Advanced factory class for creating model adapters.
    """
    
    _adapters = {
        "deepseek": DeepSeekAdapter,
        "claude": ClaudeAdapter,
        "chatgpt": ChatGPTAdapter,
        "gemini": GeminiAdapter,
        "mistral": MistralAdapter,
        "llama": LlamaAdapter,
        "cohere": CohereAdapter,
        "anthropic_claude": AnthropicClaudeAdapter,
        "openai_gpt4": OpenAIGPT4Adapter,
        # Aliases for common names
        "gpt4": OpenAIGPT4Adapter,
        "gpt-4": OpenAIGPT4Adapter,
        "gpt3": ChatGPTAdapter,
        "gpt-3": ChatGPTAdapter,
        "claude-3": AnthropicClaudeAdapter,
        "claude-2": AnthropicClaudeAdapter,
        "claude-1": AnthropicClaudeAdapter,
    }
    
    @classmethod
    def get_adapter(cls, model_name: str) -> ModelAdapter:
        """
        Get the appropriate adapter for a model.
        
        Args:
            model_name: Name of the model (case-insensitive)
            
        Returns:
            ModelAdapter instance
            
        Raises:
            ValueError: If model is not supported
        """
        model_lower = model_name.lower().replace(" ", "_").replace("-", "_")
        
        if model_lower not in cls._adapters:
            supported = ", ".join(sorted(cls._adapters.keys()))
            raise ValueError(f"Model '{model_name}' not supported. Supported models: {supported}")
        
        return cls._adapters[model_lower]()
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model names."""
        return sorted(list(cls._adapters.keys()))
    
    @classmethod
    def get_model_capabilities(cls, model_name: str) -> Dict[str, Any]:
        """Get capabilities of a specific model."""
        adapter = cls.get_adapter(model_name)
        return adapter.get_model_metadata()
    
    @classmethod
    def list_all_capabilities(cls) -> Dict[str, Dict[str, Any]]:
        """Get capabilities of all supported models."""
        capabilities = {}
        for model_name in cls.get_supported_models():
            capabilities[model_name] = cls.get_model_capabilities(model_name)
        return capabilities 