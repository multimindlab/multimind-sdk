"""
Intrinsic SAID (Structured Adaptation in the Intrinsic Dimension) implementation.
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
import numpy as np
from scipy.linalg import svd

logger = logging.getLogger(__name__)

class IntrinsicSAIDLayer(nn.Module):
    """Intrinsic SAID layer that adapts in the intrinsic dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        intrinsic_dim: int,
        rank: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intrinsic_dim = intrinsic_dim
        self.rank = rank

        # Initialize projection matrices
        self.U = nn.Parameter(torch.randn(in_features, intrinsic_dim))
        self.V = nn.Parameter(torch.randn(intrinsic_dim, out_features))

        # Initialize low-rank adaptation
        self.A = nn.Parameter(torch.randn(intrinsic_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, intrinsic_dim))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer normalization
        x_norm = self.layer_norm(x)

        # Project to intrinsic dimension
        x_intrinsic = torch.matmul(x_norm, self.U)  # [batch_size, seq_len, intrinsic_dim]

        # Apply low-rank adaptation
        adaptation = torch.matmul(
            torch.matmul(x_intrinsic, self.A),  # [batch_size, seq_len, rank]
            self.B  # [rank, intrinsic_dim]
        )  # [batch_size, seq_len, intrinsic_dim]

        # Add adaptation
        x_intrinsic = x_intrinsic + self.dropout(adaptation)

        # Project back to output dimension
        output = torch.matmul(x_intrinsic, self.V)  # [batch_size, seq_len, out_features]

        return output

class IntrinsicSAIDTuner:
    """Intrinsic SAID implementation for fine-tuning."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        intrinsic_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir

        # Default intrinsic configuration
        self.intrinsic_config = intrinsic_config or {
            "intrinsic_dim": 64,
            "rank": 8,
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

    def _compute_intrinsic_dimension(self, weight_matrix: torch.Tensor) -> int:
        """Compute the intrinsic dimension of a weight matrix using SVD."""
        # Convert to numpy for SVD
        weight_np = weight_matrix.detach().cpu().numpy()
        
        # Compute SVD
        U, S, V = svd(weight_np)
        
        # Compute cumulative variance explained
        total_var = np.sum(S ** 2)
        cum_var = np.cumsum(S ** 2) / total_var
        
        # Find dimension that explains 95% of variance
        intrinsic_dim = np.argmax(cum_var >= 0.95) + 1
        
        return min(intrinsic_dim, self.intrinsic_config["intrinsic_dim"])

    def _prepare_model(self) -> None:
        """Prepare the model for Intrinsic SAID fine-tuning."""
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

        # Replace linear layers with Intrinsic SAID layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                parent = self.model.get_submodule(parent_name)
                child_name = name.split(".")[-1]

                # Compute intrinsic dimension for this layer
                intrinsic_dim = self._compute_intrinsic_dimension(module.weight)

                new_module = IntrinsicSAIDLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    intrinsic_dim=intrinsic_dim,
                    **self.intrinsic_config
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
        """Train the model using Intrinsic SAID."""
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
        logger.info("Starting Intrinsic SAID fine-tuning")
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