"""
ModelClient Registry for MultiMindSDK
Supports dynamic loading of transformer and non-transformer models by name, class, or config.
"""
from typing import Any, Dict, Type, Optional

# Example: import wrappers and real model classes
from multimind.llm.non_transformer_llm import (
    MambaLLM, SSM_LLM, HyenaLLM, RWKVLLM, MegaS4LLM, LiquidS4LLM, S4DLLM, S4NDLLM, DSSLLM, GSSLLM, MoEMambaLLM, H3LLM, RetNetLLM, SE3HyenaLLM, TopologicalNNLLM, MLPOnlyLLM, DiffusionTextLLM, MoELLMMixin, PerceiverLLM
)

# Optionally import transformer wrappers, e.g. from multimind.llm.transformer_llm import TransformerLLM

MODEL_REGISTRY: Dict[str, Type[Any]] = {}

# Register built-in models
BUILTIN_MODELS = {
    # SSMs and advanced non-transformers
    "mamba": MambaLLM,
    "ssm": SSM_LLM,
    "hyena": HyenaLLM,
    "rwkv": RWKVLLM,
    "mega-s4": MegaS4LLM,
    "liquid-s4": LiquidS4LLM,
    "s4d": S4DLLM,
    "s4nd": S4NDLLM,
    "dss": DSSLLM,
    "gss": GSSLLM,
    "moe-mamba": MoEMambaLLM,
    "h3": H3LLM,
    "retnet": RetNetLLM,
    "se3-hyena": SE3HyenaLLM,
    "topological-nn": TopologicalNNLLM,
    "mlp-only": MLPOnlyLLM,
    "diffusion-text": DiffusionTextLLM,
    "moe": MoELLMMixin,
    "perceiver": PerceiverLLM,
    # Add transformer wrappers here, e.g. "transformer": TransformerLLM
}
MODEL_REGISTRY.update(BUILTIN_MODELS)

def register_model(name: str, model_class: Type[Any]):
    """Register a new model class by name."""
    MODEL_REGISTRY[name] = model_class

def get_model_class(name: str) -> Optional[Type[Any]]:
    """Get a model class by name."""
    return MODEL_REGISTRY.get(name)

def create_model(name: str, *args, **kwargs) -> Any:
    """
    Instantiate a model by name, passing args/kwargs to the constructor.
    """
    model_class = get_model_class(name)
    if model_class is None:
        raise ValueError(f"Model '{name}' not found in registry.")
    return model_class(*args, **kwargs)

# Example config-based loading
# config = {"type": "mamba", "model_name": ..., "model_instance": ..., "tokenizer": ...}
# model = create_model(config["type"], config["model_name"], config["model_instance"], config["tokenizer"], ...)

# Example usage:
# register_model("custom-ssm", MyCustomSSMLLM)
# model = create_model("custom-ssm", ...) 