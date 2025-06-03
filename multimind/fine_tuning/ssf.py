"""
SSF (Scaling and Shifting Features) implementation for efficient fine-tuning.
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
import logging
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)

class SSFLayer(nn.Module):
    """SSF layer that applies scaling and shifting to features."""

    def __init__(
        self,
        hidden_size: int,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Initialize scaling and shifting parameters
        self.scale = nn.Parameter(torch.ones(hidden_size) * init_scale)
        self.shift = nn.Parameter(torch.zeros(hidden_size) * init_shift)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer normalization
        x_norm = self.layer_norm(x)

        # Apply scaling and shifting
        x_scaled = x_norm * self.scale
        x_shifted = x_scaled + self.shift

        # Apply dropout
        output = self.dropout(x_shifted)

        return output

class SSFTuner:
    """SSF implementation for fine-tuning."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        ssf_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir

        # Default SSF configuration
        self.ssf_config = ssf_config or {
            "init_scale": 1.0,
            "init_shift": 0.0,
            "dropout": 0.1
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
        """Prepare the model for SSF fine-tuning."""
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

        # Add SSF layers after each transformer layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name)
                child_name = name.split(".")[-1]

                # Create SSF layer
                ssf_layer = SSFLayer(
                    hidden_size=module.normalized_shape[0],
                    **self.ssf_config
                )

                # Insert SSF layer after LayerNorm
                setattr(parent, f"{child_name}_ssf", ssf_layer)

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
        """Train the model using SSF."""
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
        logger.info("Starting SSF fine-tuning")
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