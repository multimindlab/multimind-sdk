"""
AdapterDrop implementation for dynamically dropping adapters during training.
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
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, PeftType
import logging
from datasets import Dataset as HFDataset
import random

logger = logging.getLogger(__name__)

class AdapterDropLayer(nn.Module):
    """AdapterDrop layer that dynamically drops adapters during training."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_adapters: int,
        adapter_size: int = 64,
        dropout_prob: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.adapter_size = adapter_size
        self.dropout_prob = dropout_prob

        # Initialize adapters
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, adapter_size),
                nn.ReLU(),
                nn.Linear(adapter_size, out_features)
            )
            for _ in range(num_adapters)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer normalization
        x_norm = self.layer_norm(x)

        # Initialize output
        output = torch.zeros_like(x)

        # Process through each adapter with dropout
        for i, adapter in enumerate(self.adapters):
            if self.training:
                # During training, randomly drop adapters
                if random.random() > self.dropout_prob:
                    adapter_output = adapter(x_norm)
                    output = output + adapter_output
            else:
                # During inference, use all adapters
                adapter_output = adapter(x_norm)
                output = output + adapter_output

        return output

class AdapterDropTuner:
    """AdapterDrop implementation for fine-tuning."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        num_adapters: int,
        adapter_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.num_adapters = num_adapters

        # Default adapter configuration
        self.adapter_config = adapter_config or {
            "adapter_size": 64,
            "dropout_prob": 0.1
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
        """Prepare the model for AdapterDrop fine-tuning."""
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

        # Replace linear layers with AdapterDrop layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name)
                child_name = name.split(".")[-1]

                new_module = AdapterDropLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    num_adapters=self.num_adapters,
                    **self.adapter_config
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
        """Train the model using AdapterDrop."""
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
        logger.info(f"Starting AdapterDrop fine-tuning with {self.num_adapters} adapters")
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