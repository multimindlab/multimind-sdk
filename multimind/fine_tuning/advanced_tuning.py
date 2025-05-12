"""
Compacter and HyperLoRA implementations for advanced parameter-efficient fine-tuning.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset as HFDataset
import logging
import math
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types for fine-tuning."""
    CAUSAL_LM = "causal_lm"
    SEQ_CLS = "sequence_classification"
    SEQ2SEQ = "seq2seq"

class CompacterLayer(nn.Module):
    """Compacter layer implementation with hypercomplex multiplication."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        reduction_factor: int = 4,
        non_linearity: str = "relu",
        phm_dim: int = 4,
        phm_rule: str = "random",
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reduction_factor = reduction_factor
        self.phm_dim = phm_dim
        self.phm_rule = phm_rule
        
        # Calculate dimensions
        self.phm_in = in_features // phm_dim
        self.phm_out = out_features // phm_dim
        
        # Initialize hypercomplex matrices
        if phm_rule == "random":
            self.phm_rule = torch.randn(phm_dim, phm_dim, phm_dim)
        else:
            self.phm_rule = torch.eye(phm_dim).unsqueeze(0).repeat(phm_dim, 1, 1)
            
        # Initialize parameters
        self.weight = nn.Parameter(torch.Tensor(phm_dim, self.phm_in, self.phm_out))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
            
        # Initialize non-linearity
        self.non_linearity = getattr(nn, non_linearity)() if non_linearity else None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hypercomplex multiplication."""
        batch_size = x.size(0)
        
        # Reshape input for hypercomplex multiplication
        x = x.view(batch_size, self.phm_dim, self.phm_in)
        
        # Perform hypercomplex multiplication
        x = torch.bmm(x, self.weight)
        x = torch.bmm(self.phm_rule, x)
        
        # Reshape output
        x = x.view(batch_size, self.out_features)
        
        if self.bias is not None:
            x = x + self.bias
            
        if self.non_linearity is not None:
            x = self.non_linearity(x)
            
        return x

class CompacterTuner:
    """Compacter implementation for efficient fine-tuning with hypercomplex layers."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        model_type: ModelType = ModelType.CAUSAL_LM,
        compacter_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.model_type = model_type
        
        # Default Compacter configuration
        self.compacter_config = compacter_config or {
            "reduction_factor": 4,
            "phm_dim": 4,
            "phm_rule": "random",
            "non_linearity": "relu",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "modules_to_save": None
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
        
    def _get_model_class(self):
        """Get the appropriate model class based on model type."""
        if self.model_type == ModelType.CAUSAL_LM:
            return AutoModelForCausalLM
        elif self.model_type == ModelType.SEQ_CLS:
            return AutoModelForSequenceClassification
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _prepare_model(self) -> None:
        """Prepare the model for Compacter fine-tuning."""
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
            
        # Replace target modules with Compacter layers
        for name, module in self.model.named_modules():
            if any(target in name for target in self.compacter_config["target_modules"]):
                if isinstance(module, nn.Linear):
                    parent_name = ".".join(name.split(".")[:-1])
                    parent = self.model.get_submodule(parent_name)
                    child_name = name.split(".")[-1]
                    
                    # Create Compacter layer
                    compacter = CompacterLayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        **self.compacter_config
                    )
                    setattr(parent, child_name, compacter)
                    
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
        """Train the model using Compacter."""
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
        logger.info("Starting Compacter fine-tuning...")
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


class HyperLoRATuner:
    """HyperLoRA implementation for efficient fine-tuning with hypernetworks."""
    
    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        model_type: ModelType = ModelType.CAUSAL_LM,
        hyperlora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.model_type = model_type
        
        # Default HyperLoRA configuration
        self.hyperlora_config = hyperlora_config or {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "hypernet_hidden_size": 256,
            "hypernet_num_layers": 2,
            "hypernet_dropout": 0.1
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
        self.hypernet = None
        
    def _get_model_class(self):
        """Get the appropriate model class based on model type."""
        if self.model_type == ModelType.CAUSAL_LM:
            return AutoModelForCausalLM
        elif self.model_type == ModelType.SEQ_CLS:
            return AutoModelForSequenceClassification
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _create_hypernet(self, input_size: int, output_size: int) -> nn.Module:
        """Create a hypernetwork for generating LoRA weights."""
        return nn.Sequential(
            nn.Linear(input_size, self.hyperlora_config["hypernet_hidden_size"]),
            nn.ReLU(),
            nn.Dropout(self.hyperlora_config["hypernet_dropout"]),
            *[
                nn.Sequential(
                    nn.Linear(self.hyperlora_config["hypernet_hidden_size"],
                             self.hyperlora_config["hypernet_hidden_size"]),
                    nn.ReLU(),
                    nn.Dropout(self.hyperlora_config["hypernet_dropout"])
                ) for _ in range(self.hyperlora_config["hypernet_num_layers"] - 1)
            ],
            nn.Linear(self.hyperlora_config["hypernet_hidden_size"], output_size)
        )
        
    def _prepare_model(self) -> None:
        """Prepare the model for HyperLoRA fine-tuning."""
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
            
        # Create hypernetwork for each target module
        self.hypernet = nn.ModuleDict()
        for name, module in self.model.named_modules():
            if any(target in name for target in self.hyperlora_config["target_modules"]):
                if isinstance(module, nn.Linear):
                    # Calculate input and output sizes for hypernetwork
                    input_size = module.in_features
                    output_size = module.out_features
                    lora_size = self.hyperlora_config["r"] * (input_size + output_size)
                    
                    # Create hypernetwork
                    self.hypernet[name] = self._create_hypernet(
                        input_size=input_size,
                        output_size=lora_size
                    )
                    
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.hyperlora_config["r"],
            lora_alpha=self.hyperlora_config["lora_alpha"],
            target_modules=self.hyperlora_config["target_modules"],
            lora_dropout=self.hyperlora_config["lora_dropout"],
            bias=self.hyperlora_config["bias"],
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA configuration
        self.model = get_peft_model(self.model, lora_config)
        
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
        """Train the model using HyperLoRA."""
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
        logger.info("Starting HyperLoRA fine-tuning...")
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
        
    def get_hypernet_weights(self) -> Dict[str, torch.Tensor]:
        """Get weights from the hypernetwork."""
        if self.hypernet is None:
            raise ValueError("No hypernetwork loaded. Load or train first.")
            
        weights = {}
        for name, module in self.hypernet.items():
            weights[name] = {param_name: param.data.clone() 
                           for param_name, param in module.named_parameters()}
        return weights 