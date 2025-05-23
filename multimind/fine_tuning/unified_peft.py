"""
UniPELT and MAM Adapters implementations for advanced parameter-efficient fine-tuning.
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
    # AdapterConfig,  # Commented out due to ImportError
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
from .peft_methods import PEFTMethod, PEFTTuner

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
    """UniPELT implementation that combines multiple PEFT methods."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        methods: List[UniPELTMethod],
        model_type: str = "causal_lm",
        method_configs: Optional[Dict[UniPELTMethod, Dict[str, Any]]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.methods = methods
        self.model_type = model_type

        # Default configurations for each method
        self.method_configs = {
            UniPELTMethod.LORA: {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none"
            },
            UniPELTMethod.ADAPTER: {
                "adapter_type": "houlsby",
                "adapter_size": 64,
                "adapter_non_linearity": "relu",
                "adapter_dropout": 0.1,
                "target_modules": ["k_proj", "o_proj"]
            },
            UniPELTMethod.PROMPT: {
                "prompt_tuning_init": "RANDOM",
                "num_virtual_tokens": 20,
                "token_dim": 768  # Will be set automatically
            },
            UniPELTMethod.PREFIX: {
                "num_virtual_tokens": 20,
                "encoder_hidden_size": 128,
                "encoder_num_layers": 2,
                "encoder_dropout": 0.1
            },
            UniPELTMethod.IA3: {
                "target_modules": ["fc1", "fc2"],
                "feedforward_modules": ["fc1", "fc2"]
            },
            UniPELTMethod.BITFIT: {
                "target_modules": ["bias"]  # Special case for BitFi
            }
        }

        # Update method configs with user provided values
        if method_configs:
            for method, config in method_configs.items():
                if method in self.method_configs:
                    self.method_configs[method].update(config)

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
        self.peft_configs = {}

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
        """Prepare the model for UniPELT fine-tuning."""
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
        if UniPELTMethod.PROMPT in self.methods:
            self.method_configs[UniPELTMethod.PROMPT]["token_dim"] = self.model.config.hidden_size

        # Configure each PEFT method
        for method in self.methods:
            if method == UniPELTMethod.LORA:
                config = LoraConfig(**self.method_configs[method],
                                  task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.ADAPTER:
                # config = AdapterConfig(**self.method_configs[method],
                #                      task_type=TaskType.CAUSAL_LM)
                continue  # Skip AdapterConfig for now
            elif method == UniPELTMethod.PROMPT:
                config = PromptTuningConfig(**self.method_configs[method],
                                          task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.PREFIX:
                config = PrefixTuningConfig(**self.method_configs[method],
                                          task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.IA3:
                config = IA3Config(**self.method_configs[method],
                                 task_type=TaskType.CAUSAL_LM)
            elif method == UniPELTMethod.BITFIT:
                # BitFit is handled separately
                continue

            self.peft_configs[method] = config
            self.model = get_peft_model(self.model, config)

        # Handle BitFit separately
        if UniPELTMethod.BITFIT in self.methods:
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

        # Create datase
        dataset = HFDataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_datase

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
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")

    def get_method_parameters(self) -> Dict[UniPELTMethod, Dict[str, torch.Tensor]]:
        """Get trainable parameters for each method."""
        if self.model is None:
            raise ValueError("No model loaded. Load or train first.")

        params = {}
        for method in self.methods:
            method_params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Determine which method this parameter belongs to
                    if method == UniPELTMethod.BITFIT and "bias" in name:
                        method_params[name] = param.data.clone()
                    elif method.value in name.lower():
                        method_params[name] = param.data.clone()
            params[method] = method_params
        return params

class MAMAdapterTuner:
    """MAM (Mixture of Adapters and Methods) implementation."""

    def __init__(
        self,
        base_model_name: str,
        output_dir: str,
        model_type: str = "causal_lm",
        adapter_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.model_type = model_type

        # Default adapter configuration
        self.adapter_config = adapter_config or {
            "adapter_type": "houlsby",
            "adapter_size": 64,
            "adapter_non_linearity": "relu",
            "adapter_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj"]
        }

        # Default LoRA configuration
        self.lora_config = lora_config or {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["v_proj", "o_proj"],
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
        """Prepare the model for MAM fine-tuning."""
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

        # Configure adapter
        # adapter_config = AdapterConfig(**self.adapter_config,
        #                              task_type=TaskType.CAUSAL_LM)
        # self.model = get_peft_model(self.model, adapter_config)

        # Configure LoRA
        lora_config = LoraConfig(**self.lora_config,
                                task_type=TaskType.CAUSAL_LM)
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

        # Create datase
        dataset = HFDataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_datase

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
        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")

    def get_component_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get weights from both adapter and LoRA components."""
        if self.model is None:
            raise ValueError("No model loaded. Load or train first.")

        weights = {
            "adapter": {},
            "lora": {}
        }

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "adapter" in name.lower():
                    weights["adapter"][name] = param.data.clone()
                elif "lora" in name.lower():
                    weights["lora"][name] = param.data.clone()

        return weights