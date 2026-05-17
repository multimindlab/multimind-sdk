"""Fine-tuning module for MultiMind SDK.

Provides PEFT, LoRA/QLoRA, adapters, MoE training, distillation, and friends.
Requires the ``finetune`` extras: ``pip install 'multimind-sdk[finetune]'``
(or ``[finetune-gpu]`` on Linux+CUDA for bitsandbytes-backed QLoRA).
"""

try:
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
    from .unified_tuning import (
        UniPELTTuner as UnifiedUniPELTTuner,
        MAMAdapterTuner as UnifiedMAMAdapterTuner,
    )

    from .adaptive_peft import AdaptiveUniPELTPlusTuner, AdaptiveEnhancedMAMTuner
    from .multitask_peft import MultiTaskUniPELTPlusTuner, CrossModelUniPELTPlusTuner
    from .meta_learning import MetaLearner, MultiTeacherDistillation
    from .advanced_meta_learning import (
        MAMLLearner,
        ReptileLearner,
        FewShotLearner,
        TransferLearner,
    )
    from .advanced_optimization import (
        BayesianOptimizer,
        KnowledgeDistillation,
        OptimizedMultiTaskTuner,
        DistilledMultiTaskTuner,
    )

    from .unified_fine_tuner import (
        HyperparameterTuner,
        AdapterModule,
        MoEWrapper,
        PromptEngineeringMixin,
        RAGPipeline,
    )
except ImportError as exc:  # pragma: no cover - exercised on minimal installs
    raise ImportError(
        "Fine-tuning features require additional dependencies. "
        "Install with: pip install 'multimind-sdk[finetune]' "
        "(or 'multimind-sdk[finetune-gpu]' on Linux+CUDA for bitsandbytes)."
    ) from exc

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