"""
UniPELT (Unified Parameter-Efficient Language Model Tuning) and MAM (Mixture of Adapters and Methods) implementations.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    AdapterConfig,
    PromptTuningConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset as HFDataset
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class UniPELTMethod(Enum):
    """Available methods for UniPELT."""
    LORA = "lora"
    ADAPTER = "adapter"
    PROMPT = "prompt"
    PREFIX = "prefix"
    IA3 = "ia3"
    BITFIT = "bitfit"

class UniPELTTuner:
    """UniPELT implementation that combines multiple parameter-efficient methods."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        methods: List[UniPELTMethod],
        method_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.methods = methods
        
        # Default configurations for each method
        self.method_configs = method_configs or {
            "lora": {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none"
            },
            "adapter": {
                "adapter_type": "houlsby",
                "adapter_size": 64,
                "adapter_non_linearity": "relu",
                "adapter_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            "prompt": {
                "prompt_tuning_init": "RANDOM",
                "num_virtual_tokens": 20,
                "token_dim": 768  # Will be set automatically
            },
            "prefix": {
                "num_virtual_tokens": 20,
                "encoder_hidden_size": 128,
                "encoder_num_layers": 2,
                "encoder_dropout": 0.1
            },
            "ia3": {
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
                "feedforward_modules": ["fc1", "fc2"]
            }
        }
        
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
        
    def _prepare_model(self) -> None:
        """Prepare the model for UniPELT fine-tuning."""
        # Load base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
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
            
        # Update token dimension based on model
        if "prompt" in self.method_configs:
            self.method_configs["prompt"]["token_dim"] = self.model.config.hidden_size
            
        # Configure each method
        peft_configs = []
        for method in self.methods:
            if method == UniPELTMethod.LORA:
                config = LoraConfig(**self.method_configs["lora"], task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.ADAPTER:
                config = AdapterConfig(**self.method_configs["adapter"], task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.PROMPT:
                config = PromptTuningConfig(**self.method_configs["prompt"], task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.PREFIX:
                config = PrefixTuningConfig(**self.method_configs["prefix"], task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.IA3:
                config = IA3Config(**self.method_configs["ia3"], task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.BITFIT:
                # BitFit is handled separately as it doesn't use PEFT
                continue
            peft_configs.append(config)
            
        # Apply PEFT configurations
        if peft_configs:
            self.model = get_peft_model(self.model, peft_configs[0])
            for config in peft_configs[1:]:
                self.model.add_adapter(config)
                
        # Handle BitFit if selected
        if UniPELTMethod.BITFIT in self.methods:
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    param.requires_grad = True
                    
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
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
        """Train the model using UniPELT."""
        if self.model is None:
            self._prepare_model()
            
        # Prepare datasets if raw texts are provided
        if isinstance(train_dataset, list):
            train_dataset = self.prepare_dataset(train_dataset, **kwargs)
        if isinstance(eval_dataset, list):
            eval_dataset = self.prepare_dataset(eval_dataset, **kwargs)
            
        # Create trainer
        training_args = TrainingArguments(**self.training_args)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train
        logger.info(f"Starting UniPELT fine-tuning with methods: {[m.value for m in self.methods]}")
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
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")


class MAMAdapterTuner:
    """MAM (Mixture of Adapters and Methods) implementation for adaptive fine-tuning."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        adapter_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        
        # Default adapter configuration
        self.adapter_config = adapter_config or {
            "adapter_type": "houlsby",
            "adapter_size": 64,
            "adapter_non_linearity": "relu",
            "adapter_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        # Default LoRA configuration
        self.lora_config = lora_config or {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        }
        
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
        
    def _prepare_model(self) -> None:
        """Prepare the model for MAM fine-tuning."""
        # Load base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
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
            
        # Configure adapter
        adapter_config = AdapterConfig(**self.adapter_config, task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, adapter_config)
        
        # Add LoRA
        lora_config = LoraConfig(**self.lora_config, task_type=TaskType.CAUSAL_LM)
        self.model.add_adapter(lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
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
        """Train the model using MAM."""
        if self.model is None:
            self._prepare_model()
            
        # Prepare datasets if raw texts are provided
        if isinstance(train_dataset, list):
            train_dataset = self.prepare_dataset(train_dataset, **kwargs)
        if isinstance(eval_dataset, list):
            eval_dataset = self.prepare_dataset(eval_dataset, **kwargs)
            
        # Create trainer
        training_args = TrainingArguments(**self.training_args)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train
        logger.info("Starting MAM fine-tuning...")
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
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")
        
    def get_adapter_weights(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get weights from both adapter and LoRA components."""
        if self.model is None:
            raise ValueError("No model loaded. Load or train first.")
            
        adapter_weights = {}
        lora_weights = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "adapter" in name:
                    adapter_weights[name] = param.data.clone()
                elif "lora" in name:
                    lora_weights[name] = param.data.clone()
                    
        return adapter_weights, lora_weights 