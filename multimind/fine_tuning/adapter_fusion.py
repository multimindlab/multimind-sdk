"""
AdapterFusion implementation for combining multiple adapters through a fusion layer.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
    PeftType
)
import logging
from datasets import Dataset as HFDataset

import warnings

# Deprecated compatibility shim for AdapterConfig
class AdapterConfig:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AdapterConfig is deprecated. Please use LoraConfig or PeftConfig instead.",
            DeprecationWarning
        )
        self._config = LoraConfig(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._config, item)

# Deprecated compatibility shim for TaskType
class TaskType:
    def __init__(self, value=None, *args, **kwargs):
        warnings.warn(
            "TaskType is deprecated. Please use PeftType instead.",
            DeprecationWarning
        )
        if value is None:
            self._type = PeftType.LORA
        else:
            self._type = PeftType(value)

    def __getattr__(self, item):
        return getattr(self._type, item)

logger = logging.getLogger(__name__)

class AdapterFusionLayer(nn.Module):
    """AdapterFusion layer that combines multiple adapters through attention."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_adapters: int,
        adapter_size: int = 64,
        attention_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.adapter_size = adapter_size

        # Query, Key, Value projections for attention
        self.query = nn.Linear(in_features, adapter_size)
        self.key = nn.Linear(adapter_size, adapter_size)
        self.value = nn.Linear(adapter_size, adapter_size)

        # Output projection
        self.output = nn.Linear(adapter_size, out_features)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(in_features)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor, adapter_outputs: List[torch.Tensor]) -> torch.Tensor:
        # Layer normalization
        x_norm = self.layer_norm(x)

        # Project query
        query = self.query(x_norm)  # [batch_size, seq_len, adapter_size]

        # Stack adapter outputs
        adapter_outputs = torch.stack(adapter_outputs, dim=1)  # [batch_size, num_adapters, seq_len, adapter_size]

        # Project keys and values
        keys = self.key(adapter_outputs)  # [batch_size, num_adapters, seq_len, adapter_size]
        values = self.value(adapter_outputs)  # [batch_size, num_adapters, seq_len, adapter_size]

        # Compute attention scores
        attention_scores = torch.matmul(
            query.unsqueeze(1),  # [batch_size, 1, seq_len, adapter_size]
            keys.transpose(-2, -1)  # [batch_size, num_adapters, adapter_size, seq_len]
        )  # [batch_size, num_adapters, seq_len, seq_len]

        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Compute weighted sum of values
        context = torch.matmul(
            attention_probs,  # [batch_size, num_adapters, seq_len, seq_len]
            values  # [batch_size, num_adapters, seq_len, adapter_size]
        )  # [batch_size, num_adapters, seq_len, adapter_size]

        # Sum over adapters
        context = context.sum(dim=1)  # [batch_size, seq_len, adapter_size]

        # Project to output dimension
        output = self.output(context)  # [batch_size, seq_len, out_features]

        return output

class AdapterFusionTuner:
    """AdapterFusion implementation for fine-tuning."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        adapter_configs: List[Dict[str, Any]],
        fusion_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.adapter_configs = adapter_configs

        # Default fusion configuration
        self.fusion_config = fusion_config or {
            "adapter_size": 64,
            "attention_dropout": 0.1
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
        self.adapters = []

    def _prepare_model(self) -> None:
        """Prepare the model for AdapterFusion fine-tuning."""
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

        # Add adapters
        for i, config in enumerate(self.adapter_configs):
            adapter_config = LoraConfig(
                **config,
                task_type=PeftType.CAUSAL_LM
            )
            self.model.add_adapter(f"adapter_{i}", adapter_config)
            self.adapters.append(f"adapter_{i}")

        # Add fusion layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name)
                child_name = name.split(".")[-1]

                new_module = AdapterFusionLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    num_adapters=len(self.adapters),
                    **self.fusion_config
                )
                setattr(parent, child_name, new_module)

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
        """Train the model using AdapterFusion."""
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
        logger.info(f"Starting AdapterFusion fine-tuning with {len(self.adapters)} adapters")
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

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters from the model."""
        if self.model is None:
            raise ValueError("No model loaded. Load or train first.")

        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone()
        return params 

__all__ = [
    'AdapterFusionLayer',
    'AdapterFusionTuner',
    'AdapterConfig',
    'TaskType',
]