"""
Integration adapters for fine-tuned models to work with various frameworks.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from lite_llm import LiteLLM
from superagi.agent import Agent
from superagi.tools import Tool
from semantic_kernel import Kernel, KernelFunction
from crewai import Agent as CrewAgent
from crewai import Task
import logging
from ..fine_tuning import (
    MultiTaskUniPELTPlusTuner,
    OptimizedMultiTaskTuner,
    DistilledMultiTaskTuner,
    TaskConfig,
    TaskType,
    UniPELTPlusMethod
)

logger = logging.getLogger(__name__)

class BaseModelAdapter:
    """Base class for model adapters."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "causal_lm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state as embeddings
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            
        return embeddings

class LangChainAdapter(LLM, BaseModelAdapter):
    """Adapter for LangChain integration."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "causal_lm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(model_path=model_path, model_type=model_type, device=device)
        self.kwargs = kwargs
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Generate text for LangChain."""
        # Update generation parameters
        gen_kwargs = {**self.kwargs, **kwargs}
        if stop:
            gen_kwargs["stop_sequences"] = stop
            
        return self.generate(prompt, **gen_kwargs)
        
    @property
    def _llm_type(self) -> str:
        """Return LLM type."""
        return "peft_model"

class LangChainEmbeddings(Embeddings, BaseModelAdapter):
    """Adapter for LangChain embeddings."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "causal_lm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(model_path=model_path, model_type=model_type, device=device)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for documents."""
        embeddings = []
        for text in texts:
            emb = self.get_embeddings(text)
            embeddings.append(emb.cpu().numpy().tolist())
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for query."""
        emb = self.get_embeddings(text)
        return emb.cpu().numpy().tolist()

class LiteLLMAdapter(LiteLLM, BaseModelAdapter):
    """Adapter for LiteLLM integration."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "causal_lm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(model_path=model_path, model_type=model_type, device=device)
        self.kwargs = kwargs
        
    def completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion for LiteLLM."""
        # Update generation parameters
        gen_kwargs = {**self.kwargs, **kwargs}
        gen_kwargs["max_length"] = max_tokens
        
        # Generate text
        generated_text = self.generate(prompt, **gen_kwargs)
        
        # Format response
        return {
            "choices": [{
                "text": generated_text,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(self.tokenizer.encode(prompt)),
                "completion_tokens": len(self.tokenizer.encode(generated_text)),
                "total_tokens": len(self.tokenizer.encode(prompt + generated_text))
            }
        }

class SuperAGIAdapter(Agent, BaseModelAdapter):
    """Adapter for SuperAGI integration."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "causal_lm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tools: Optional[List[Tool]] = None,
        **kwargs
    ):
        super().__init__(model_path=model_path, model_type=model_type, device=device)
        self.tools = tools or []
        self.kwargs = kwargs
        
    def execute(self, task: str, **kwargs) -> str:
        """Execute task using SuperAGI agent."""
        # Format prompt with tools
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        prompt = f"""Available tools:
{tool_descriptions}

Task: {task}

Think step by step and use the available tools to complete the task.
"""
        
        # Generate response
        response = self.generate(prompt, **{**self.kwargs, **kwargs})
        
        # Parse and execute tool calls
        # Implementation depends on SuperAGI's tool execution interface
        return response

class SemanticKernelAdapter(KernelFunction, BaseModelAdapter):
    """Adapter for Semantic Kernel integration."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "causal_lm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(model_path=model_path, model_type=model_type, device=device)
        self.kwargs = kwargs
        
    def invoke(
        self,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Invoke function for Semantic Kernel."""
        # Get prompt from context
        prompt = context.get("prompt", "")
        
        # Generate response
        response = self.generate(prompt, **{**self.kwargs, **kwargs})
        
        # Update context with response
        context["response"] = response
        return context

class CrewAIAdapter(CrewAgent, BaseModelAdapter):
    """Adapter for CrewAI integration."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "causal_lm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        role: str = "AI Assistant",
        goal: str = "Help users with their tasks",
        backstory: str = "I am an AI assistant trained to help users.",
        **kwargs
    ):
        super().__init__(model_path=model_path, model_type=model_type, device=device)
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.kwargs = kwargs
        
    def execute_task(self, task: Task) -> str:
        """Execute task using CrewAI agent."""
        # Format prompt with agent context
        prompt = f"""Role: {self.role}
Goal: {self.goal}
Backstory: {self.backstory}

Task: {task.description}

Think step by step and complete the task.
"""
        
        # Generate response
        response = self.generate(prompt, **self.kwargs)
        
        # Update task with response
        task.output = response
        return response

def create_adapter(
    framework: str,
    model_path: str,
    model_type: str = "causal_lm",
    **kwargs
) -> BaseModelAdapter:
    """Factory function to create appropriate adapter."""
    adapters = {
        "langchain": LangChainAdapter,
        "langchain_embeddings": LangChainEmbeddings,
        "litellm": LiteLLMAdapter,
        "superagi": SuperAGIAdapter,
        "semantic_kernel": SemanticKernelAdapter,
        "crewai": CrewAIAdapter
    }
    
    if framework not in adapters:
        raise ValueError(f"Unsupported framework: {framework}")
        
    return adapters[framework](
        model_path=model_path,
        model_type=model_type,
        **kwargs
    ) 