"""
Prefix/Prompt Pooling implementation for efficient fine-tuning.
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
    PromptTuningConfig,
    PrefixTuningConfig,
    get_peft_model,
    TaskType
)
import logging
from datasets import Dataset as HFDataset
import numpy as np

logger = logging.getLogger(__name__)

class PromptPoolingLayer(nn.Module):
    """Prompt Pooling layer that uses a pool of prompts/prefixes."""

    def __init__(
        self,
        num_virtual_tokens: int,
        token_dim: int,
        pool_size: int,
        method: str = "prompt",  # "prompt" or "prefix"
        attention_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.token_dim = token_dim
        self.pool_size = pool_size
        self.method = method

        # Initialize prompt/prefix pool
        if method == "prompt":
            self.pool = nn.Parameter(
                torch.randn(pool_size, num_virtual_tokens, token_dim)
            )
        else:  # prefix
            self.pool = nn.Parameter(
                torch.randn(pool_size, num_virtual_tokens, token_dim)
            )
            self.prefix_projection = nn.Linear(token_dim, token_dim)

        # Attention for selecting from pool
        self.query = nn.Linear(token_dim, token_dim)
        self.key = nn.Linear(token_dim, token_dim)
        self.value = nn.Linear(token_dim, token_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer normalization
        x_norm = self.layer_norm(x)

        # Project query
        query = self.query(x_norm)  # [batch_size, seq_len, token_dim]

        # Project keys and values from pool
        keys = self.key(self.pool)  # [pool_size, num_virtual_tokens, token_dim]
        values = self.value(self.pool)  # [pool_size, num_virtual_tokens, token_dim]

        # Compute attention scores
        attention_scores = torch.matmul(
            query.unsqueeze(1),  # [batch_size, 1, seq_len, token_dim]
            keys.transpose(-2, -1)  # [pool_size, token_dim, num_virtual_tokens]
        )  # [batch_size, pool_size, seq_len, num_virtual_tokens]

        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Compute weighted sum of values
        context = torch.matmul(
            attention_probs,  # [batch_size, pool_size, seq_len, num_virtual_tokens]
            values  # [pool_size, num_virtual_tokens, token_dim]
        )  # [batch_size, pool_size, seq_len, token_dim]

        # Sum over pool
        context = context.sum(dim=1)  # [batch_size, seq_len, token_dim]

        if self.method == "prefix":
            # Project prefix
            context = self.prefix_projection(context)

        return context

class PromptPoolingTuner:
    """Prompt/Prefix Pooling implementation for fine-tuning."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        method: str = "prompt",  # "prompt" or "prefix"
        pool_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.method = method

        # Default pool configuration
        self.pool_config = pool_config or {
            "num_virtual_tokens": 20,
            "pool_size": 10,
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

    def _prepare_model(self) -> None:
        """Prepare the model for Prompt/Prefix Pooling fine-tuning."""
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

        # Configure prompt/prefix tuning
        if self.method == "prompt":
            config = PromptTuningConfig(
                num_virtual_tokens=self.pool_config["num_virtual_tokens"],
                task_type=TaskType.CAUSAL_LM
            )
        else:  # prefix
            config = PrefixTuningConfig(
                num_virtual_tokens=self.pool_config["num_virtual_tokens"],
                task_type=TaskType.CAUSAL_LM
            )

        # Get the model
        self.model = get_peft_model(self.model, config)

        # Replace prompt/prefix layers with pooling layers
        for name, module in self.model.named_modules():
            if isinstance(module, (PromptTuningConfig, PrefixTuningConfig)):
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name)
                child_name = name.split(".")[-1]

                new_module = PromptPoolingLayer(
                    num_virtual_tokens=self.pool_config["num_virtual_tokens"],
                    token_dim=self.model.config.hidden_size,
                    pool_size=self.pool_config["pool_size"],
                    method=self.method,
                    **self.pool_config
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
        """Train the model using Prompt/Prefix Pooling."""
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
        logger.info(f"Starting {self.method.capitalize()} Pooling fine-tuning")
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