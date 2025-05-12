"""
Advanced PEFT implementations including UniPELT++ and Enhanced MAM Adapters.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Set
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    AdapterConfig,
    PromptTuningConfig,
    PrefixTuningConfig,
    IA3Config,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset as HFDataset
import logging
from enum import Enum
from .unified_peft import UniPELTMethod, UniPELTTuner, MAMAdapterTuner

logger = logging.getLogger(__name__)

class UniPELTPlusMethod(Enum):
    """Available methods for UniPELT++."""
    LORA = "lora"
    ADAPTER = "adapter"
    PROMPT = "prompt"
    PREFIX = "prefix"
    IA3 = "ia3"
    BITFIT = "bitfit"
    DIFFPRUNING = "diffpruning"
    SPARSE_ADAPTER = "sparse_adapter"
    COMPACTER = "compacter"
    HYPERLORA = "hyperlora"

class UniPELTPlusTuner(UniPELTTuner):
    """Enhanced UniPELT implementation with additional methods and features."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        methods: List[UniPELTPlusMethod],
        model_type: str = "causal_lm",
        method_configs: Optional[Dict[UniPELTPlusMethod, Dict[str, Any]]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Convert UniPELTPlusMethod to UniPELTMethod for base class
        base_methods = [UniPELTMethod(method.value) for method in methods 
                       if method.value in [m.value for m in UniPELTMethod]]
        
        super().__init__(
            base_model_name=base_model_name,
            output_dir=output_dir,
            methods=base_methods,
            model_type=model_type,
            method_configs=method_configs,
            training_args=training_args
        )
        
        self.methods = methods  # Store original methods
        self.model_config = model_config or {}
        
        # Additional method configurations
        self.method_configs.update({
            UniPELTPlusMethod.DIFFPRUNING: {
                "sparsity": 0.1,
                "mask_init": "uniform",
                "target_modules": ["q_proj", "v_proj"]
            },
            UniPELTPlusMethod.SPARSE_ADAPTER: {
                "adapter_size": 64,
                "sparsity": 0.1,
                "non_linearity": "relu",
                "target_modules": ["q_proj", "v_proj"]
            },
            UniPELTPlusMethod.COMPACTER: {
                "reduction_factor": 4,
                "phm_dim": 4,
                "phm_rule": "random",
                "target_modules": ["q_proj", "v_proj"]
            },
            UniPELTPlusMethod.HYPERLORA: {
                "r": 8,
                "hypernet_hidden_size": 256,
                "hypernet_num_layers": 2,
                "target_modules": ["q_proj", "v_proj"]
            }
        })
        
    def _prepare_model(self) -> None:
        """Prepare the model for UniPELT++ fine-tuning."""
        # Load base model with custom config
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **self.model_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Update token dimension for prompt tuning
        if UniPELTPlusMethod.PROMPT in self.methods:
            self.method_configs[UniPELTPlusMethod.PROMPT]["token_dim"] = self.model.config.hidden_size
            
        # Configure each PEFT method
        for method in self.methods:
            if method in [UniPELTMethod.LORA, UniPELTMethod.ADAPTER, 
                         UniPELTMethod.PROMPT, UniPELTMethod.PREFIX, 
                         UniPELTMethod.IA3]:
                # Use base class method for standard PEFT methods
                continue
                
            if method == UniPELTPlusMethod.DIFFPRUNING:
                self._apply_diffpruning()
            elif method == UniPELTPlusMethod.SPARSE_ADAPTER:
                self._apply_sparse_adapter()
            elif method == UniPELTPlusMethod.COMPACTER:
                self._apply_compacter()
            elif method == UniPELTPlusMethod.HYPERLORA:
                self._apply_hyperlora()
                
        # Apply base class methods
        super()._prepare_model()
        
    def _apply_diffpruning(self) -> None:
        """Apply DiffPruning to the model."""
        from .peft_methods import DiffPruningLayer
        
        for name, module in self.model.named_modules():
            if any(target in name for target in 
                  self.method_configs[UniPELTPlusMethod.DIFFPRUNING]["target_modules"]):
                if isinstance(module, nn.Linear):
                    parent_name = ".".join(name.split(".")[:-1])
                    parent = self.model.get_submodule(parent_name)
                    child_name = name.split(".")[-1]
                    
                    new_module = DiffPruningLayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        **self.method_configs[UniPELTPlusMethod.DIFFPRUNING]
                    )
                    setattr(parent, child_name, new_module)
                    
    def _apply_sparse_adapter(self) -> None:
        """Apply SparseAdapter to the model."""
        from .peft_methods import SparseAdapterLayer
        
        for name, module in self.model.named_modules():
            if any(target in name for target in 
                  self.method_configs[UniPELTPlusMethod.SPARSE_ADAPTER]["target_modules"]):
                if isinstance(module, nn.Linear):
                    parent_name = ".".join(name.split(".")[:-1])
                    parent = self.model.get_submodule(parent_name)
                    child_name = name.split(".")[-1]
                    
                    new_module = SparseAdapterLayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        **self.method_configs[UniPELTPlusMethod.SPARSE_ADAPTER]
                    )
                    setattr(parent, child_name, new_module)
                    
    def _apply_compacter(self) -> None:
        """Apply Compacter to the model."""
        from .advanced_tuning import CompacterLayer
        
        for name, module in self.model.named_modules():
            if any(target in name for target in 
                  self.method_configs[UniPELTPlusMethod.COMPACTER]["target_modules"]):
                if isinstance(module, nn.Linear):
                    parent_name = ".".join(name.split(".")[:-1])
                    parent = self.model.get_submodule(parent_name)
                    child_name = name.split(".")[-1]
                    
                    new_module = CompacterLayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        **self.method_configs[UniPELTPlusMethod.COMPACTER]
                    )
                    setattr(parent, child_name, new_module)
                    
    def _apply_hyperlora(self) -> None:
        """Apply HyperLoRA to the model."""
        from .advanced_tuning import HyperLoRATuner
        
        hyperlora = HyperLoRATuner(
            base_model_name=self.base_model_name,
            output_dir=self.output_dir,
            model_type=self.model_type,
            hyperlora_config=self.method_configs[UniPELTPlusMethod.HYPERLORA]
        )
        hyperlora._prepare_model()
        self.model = hyperlora.model

class EnhancedMAMAdapterTuner(MAMAdapterTuner):
    """Enhanced MAM implementation with additional components."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        model_type: str = "causal_lm",
        adapter_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        prompt_config: Optional[Dict[str, Any]] = None,
        prefix_config: Optional[Dict[str, Any]] = None,
        ia3_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            base_model_name=base_model_name,
            output_dir=output_dir,
            model_type=model_type,
            adapter_config=adapter_config,
            lora_config=lora_config,
            training_args=training_args
        )
        
        self.model_config = model_config or {}
        
        # Additional component configurations
        self.prompt_config = prompt_config or {
            "prompt_tuning_init": "RANDOM",
            "num_virtual_tokens": 20,
            "token_dim": 768  # Will be set automatically
        }
        
        self.prefix_config = prefix_config or {
            "num_virtual_tokens": 20,
            "encoder_hidden_size": 128,
            "encoder_num_layers": 2,
            "encoder_dropout": 0.1
        }
        
        self.ia3_config = ia3_config or {
            "target_modules": ["fc1", "fc2"],
            "feedforward_modules": ["fc1", "fc2"]
        }
        
    def _prepare_model(self) -> None:
        """Prepare the model for Enhanced MAM fine-tuning."""
        # Load base model with custom config
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **self.model_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Update token dimension for prompt tuning
        self.prompt_config["token_dim"] = self.model.config.hidden_size
        
        # Configure each component
        # 1. Adapter
        adapter_config = AdapterConfig(**self.adapter_config, 
                                     task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, adapter_config)
        
        # 2. LoRA
        lora_config = LoraConfig(**self.lora_config, 
                                task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, lora_config)
        
        # 3. Prompt Tuning
        prompt_config = PromptTuningConfig(**self.prompt_config, 
                                         task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, prompt_config)
        
        # 4. Prefix Tuning
        prefix_config = PrefixTuningConfig(**self.prefix_config, 
                                         task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, prefix_config)
        
        # 5. IA³
        ia3_config = IA3Config(**self.ia3_config, 
                              task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, ia3_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        
    def get_component_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get weights from all components."""
        if self.model is None:
            raise ValueError("No model loaded. Load or train first.")
            
        weights = {
            "adapter": {},
            "lora": {},
            "prompt": {},
            "prefix": {},
            "ia3": {}
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "adapter" in name.lower():
                    weights["adapter"][name] = param.data.clone()
                elif "lora" in name.lower():
                    weights["lora"][name] = param.data.clone()
                elif "prompt" in name.lower():
                    weights["prompt"][name] = param.data.clone()
                elif "prefix" in name.lower():
                    weights["prefix"][name] = param.data.clone()
                elif "ia3" in name.lower():
                    weights["ia3"][name] = param.data.clone()
                    
        return weights 