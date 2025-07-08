"""
Fine-tuning module for MultiMind SDK.

This module provides fine-tuning capabilities for language models.
"""

# Core fine-tuning classes
from .adapter_drop import AdapterDropTuner
from .adapter_fusion import AdapterFusionTuner
from .adapter_tuning import AdapterTuner
from .lora_trainer import LoRATrainer
from .qlora_trainer import QLoraTuner
from .prompt_tuning import PromptTuner, PrefixTuner
from .peft_methods import PEFTTuner
from .unified_peft import UniPELTTuner
from .advanced_unified_peft import UniPELTPlusTuner
from .moe_tuning import MoETrainer
from .rag_fine_tuner import RAGFineTuner
from .ssf import SSFTuner
from .intrinsic_said import IntrinsicSAIDTuner
from .ia3_bitfit import IA3Tuner, BitFitTuner
from .prompt_pooling import PromptPoolingTuner
from .advanced_tuning import CompacterTuner, HyperLoRATuner
from .mam_adapter import MAMAdapterTuner
from .unified_tuning import UniPELTTuner as UnifiedUniPELTTuner, MAMAdapterTuner as UnifiedMAMAdapterTuner

# Advanced fine-tuning classes
from .adaptive_peft import AdaptiveUniPELTPlusTuner, AdaptiveEnhancedMAMTuner
from .multitask_peft import MultiTaskUniPELTPlusTuner, CrossModelUniPELTPlusTuner
from .meta_learning import MetaLearner, MultiTeacherDistillation
from .advanced_meta_learning import MAMLLearner, ReptileLearner, FewShotLearner, TransferLearner
from .advanced_optimization import BayesianOptimizer, KnowledgeDistillation, OptimizedMultiTaskTuner, DistilledMultiTaskTuner

# Unified fine-tuning components
from .unified_fine_tuner import HyperparameterTuner, AdapterModule, MoEWrapper, PromptEngineeringMixin, RAGPipeline

__all__ = [
    # Core fine-tuning
    "AdapterDropTuner",
    "AdapterFusionTuner", 
    "AdapterTuner",
    "LoRATrainer",
    "QLoraTuner",
    "PromptTuner",
    "PrefixTuner",
    "PEFTTuner",
    "UniPELTTuner",
    "UniPELTPlusTuner",
    "MoETrainer",
    "RAGFineTuner",
    "SSFTuner",
    "IntrinsicSAIDTuner",
    "IA3Tuner",
    "BitFitTuner",
    "PromptPoolingTuner",
    "CompacterTuner",
    "HyperLoRATuner",
    "MAMAdapterTuner",
    "UnifiedUniPELTTuner",
    "UnifiedMAMAdapterTuner",
    
    # Advanced fine-tuning
    "AdaptiveUniPELTPlusTuner",
    "AdaptiveEnhancedMAMTuner",
    "MultiTaskUniPELTPlusTuner",
    "CrossModelUniPELTPlusTuner",
    "MetaLearner",
    "MultiTeacherDistillation",
    "MAMLLearner",
    "ReptileLearner",
    "FewShotLearner",
    "TransferLearner",
    "BayesianOptimizer",
    "KnowledgeDistillation",
    "OptimizedMultiTaskTuner",
    "DistilledMultiTaskTuner",
    
    # Unified components
    "HyperparameterTuner",
    "AdapterModule",
    "MoEWrapper",
    "PromptEngineeringMixin",
    "RAGPipeline",
] 