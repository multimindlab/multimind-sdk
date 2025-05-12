"""
Additional PEFT (Parameter-Efficient Fine-Tuning) methods implementation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
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
import math
from enum import Enum

logger = logging.getLogger(__name__)

class PEFTMethod(Enum):
    """Available PEFT methods."""
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

class DiffPruningLayer(nn.Module):
    """DiffPruning layer implementation for sparse fine-tuning."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.1,
        mask_init: str = "uniform",
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Initialize weight mask
        if mask_init == "uniform":
            self.mask = nn.Parameter(torch.rand(in_features, out_features))
        else:
            self.mask = nn.Parameter(torch.ones(in_features, out_features))
            
        # Initialize weights
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse mask."""
        # Apply mask to weights
        masked_weight = self.weight * torch.sigmoid(self.mask)
        
        # Apply sparsity
        if self.training:
            k = int(self.sparsity * self.mask.numel())
            threshold = torch.kthvalue(self.mask.view(-1), k).values
            mask = (self.mask > threshold).float()
            masked_weight = masked_weight * mask
            
        return nn.functional.linear(x, masked_weight, self.bias)

class SparseAdapterLayer(nn.Module):
    """SparseAdapter layer implementation with dynamic sparsity."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        adapter_size: int = 64,
        sparsity: float = 0.1,
        non_linearity: str = "relu",
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adapter_size = adapter_size
        self.sparsity = sparsity
        
        # Adapter layers
        self.down = nn.Linear(in_features, adapter_size)
        self.up = nn.Linear(adapter_size, out_features)
        
        # Sparsity masks
        self.down_mask = nn.Parameter(torch.ones(in_features, adapter_size))
        self.up_mask = nn.Parameter(torch.ones(adapter_size, out_features))
        
        # Non-linearity
        self.non_linearity = getattr(nn, non_linearity)()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse adapters."""
        # Apply sparsity to down projection
        k_down = int(self.sparsity * self.down_mask.numel())
        threshold_down = torch.kthvalue(self.down_mask.view(-1), k_down).values
        mask_down = (self.down_mask > threshold_down).float()
        down_weight = self.down.weight * mask_down
        
        # Apply sparsity to up projection
        k_up = int(self.sparsity * self.up_mask.numel())
        threshold_up = torch.kthvalue(self.up_mask.view(-1), k_up).values
        mask_up = (self.up_mask > threshold_up).float()
        up_weight = self.up.weight * mask_up
        
        # Forward pass
        h = nn.functional.linear(x, down_weight, self.down.bias)
        h = self.non_linearity(h)
        return nn.functional.linear(h, up_weight, self.up.bias)

class PEFTTuner:
    """Unified PEFT implementation supporting multiple methods."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        method: PEFTMethod,
        model_type: str = "causal_lm",
        method_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.method = method
        self.model_type = model_type
        
        # Default configurations for each method
        self.method_configs = {
            PEFTMethod.LORA: {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none"
            },
            PEFTMethod.ADAPTER: {
                "adapter_type": "houlsby",
                "adapter_size": 64,
                "adapter_non_linearity": "relu",
                "adapter_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            PEFTMethod.PROMPT: {
                "prompt_tuning_init": "RANDOM",
                "num_virtual_tokens": 20,
                "token_dim": 768  # Will be set automatically
            },
            PEFTMethod.PREFIX: {
                "num_virtual_tokens": 20,
                "encoder_hidden_size": 128,
                "encoder_num_layers": 2,
                "encoder_dropout": 0.1
            },
            PEFTMethod.IA3: {
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
                "feedforward_modules": ["fc1", "fc2"]
            },
            PEFTMethod.DIFFPRUNING: {
                "sparsity": 0.1,
                "mask_init": "uniform",
                "target_modules": ["q_proj", "v_proj"]
            },
            PEFTMethod.SPARSE_ADAPTER: {
                "adapter_size": 64,
                "sparsity": 0.1,
                "non_linearity": "relu",
                "target_modules": ["q_proj", "v_proj"]
            }
        }
        
        # Update method config with user provided values
        if method_config:
            self.method_configs[method].update(method_config)
            
        # Default training arguments
        self.training_args = training_args or {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-3,
            "fp16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine"
        }
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _get_model_class(self):
        """Get the appropriate model class based on model type."""
        if self.model_type == "causal_lm":
            return AutoModelForCausalLM
        elif self.model_type == "sequence_classification":
            return AutoModelForSequenceClassification
        elif self.model_type == "seq2seq":
            return AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _prepare_model(self) -> None:
        """Prepare the model for PEFT fine-tuning."""
        # Load base model and tokenizer
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Update token dimension for prompt tuning
        if self.method == PEFTMethod.PROMPT:
            self.method_configs[self.method]["token_dim"] = self.model.config.hidden_size
            
        # Configure PEFT method
        if self.method in [PEFTMethod.LORA, PEFTMethod.ADAPTER, PEFTMethod.PROMPT, 
                          PEFTMethod.PREFIX, PEFTMethod.IA3]:
            # Use PEFT library for standard methods
            if self.method == PEFTMethod.LORA:
                config = LoraConfig(**self.method_configs[self.method], 
                                  task_type=TaskType.CAUSAL_LM)
            elif self.method == PEFTMethod.ADAPTER:
                config = AdapterConfig(**self.method_configs[self.method], 
                                     task_type=TaskType.CAUSAL_LM)
            elif self.method == PEFTMethod.PROMPT:
                config = PromptTuningConfig(**self.method_configs[self.method], 
                                          task_type=TaskType.CAUSAL_LM)
            elif self.method == PEFTMethod.PREFIX:
                config = PrefixTuningConfig(**self.method_configs[self.method], 
                                          task_type=TaskType.CAUSAL_LM)
            elif self.method == PEFTMethod.IA3:
                config = IA3Config(**self.method_configs[self.method], 
                                 task_type=TaskType.CAUSAL_LM)
                
            self.model = get_peft_model(self.model, config)
            
        elif self.method in [PEFTMethod.DIFFPRUNING, PEFTMethod.SPARSE_ADAPTER]:
            # Custom implementation for advanced methods
            for name, module in self.model.named_modules():
                if any(target in name for target in self.method_configs[self.method]["target_modules"]):
                    if isinstance(module, nn.Linear):
                        parent_name = ".".join(name.split(".")[:-1])
                        parent = self.model.get_submodule(parent_name)
                        child_name = name.split(".")[-1]
                        
                        if self.method == PEFTMethod.DIFFPRUNING:
                            new_module = DiffPruningLayer(
                                in_features=module.in_features,
                                out_features=module.out_features,
                                **self.method_configs[self.method]
                            )
                        else:  # SPARSE_ADAPTER
                            new_module = SparseAdapterLayer(
                                in_features=module.in_features,
                                out_features=module.out_features,
                                **self.method_configs[self.method]
                            )
                            
                        setattr(parent, child_name, new_module)
                        
        elif self.method == PEFTMethod.BITFIT:
            # BitFit: only train bias terms
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        
    def prepare_dataset(
        self,
        texts: List[str],
        max_length: int = 512,
        **kwargs
    ) -> HFDataset:
        """Prepare dataset for training."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            
        # Create dataset
        dataset = HFDataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def train(
        self,
        train_dataset: Union[HFDataset, List[str]],
        eval_dataset: Optional[Union[HFDataset, List[str]]] = None,
        **kwargs
    ) -> None:
        """Train the model using the selected PEFT method."""
        if self.model is None:
            self._prepare_model()
            
        # Prepare datasets if raw texts are provided
        if isinstance(train_dataset, list):
            train_dataset = self.prepare_dataset(train_dataset, **kwargs)
        if isinstance(eval_dataset, list):
            eval_dataset = self.prepare_dataset(eval_dataset, **kwargs)
            
        # Create trainer
        training_args = TrainingArguments(**self.training_args)
        
        # Select appropriate data collator
        if self.model_type == "seq2seq":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info(f"Starting {self.method.value} fine-tuning...")
        self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")
        
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the fine-tuned model."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
            
        save_path = path or self.output_dir
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
        
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model."""
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")
        
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters from the model."""
        if self.model is None:
            raise ValueError("No model loaded. Load or train first.")
            
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone()
        return params 