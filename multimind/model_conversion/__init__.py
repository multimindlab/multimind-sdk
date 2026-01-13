"""
Model conversion module for MultiMind SDK.

This module provides model conversion capabilities for different formats and optimizations.
"""

# Base classes
from .base import BaseModelConverter

# Core converters
from .huggingface import HuggingFaceConverter
from .ollama import OllamaConverter

# Try to import ONNXConverter, but handle gracefully if not available
try:
    from .onnx import ONNXConverter
    ONNX_CONVERTER_AVAILABLE = True
except ImportError:
    ONNX_CONVERTER_AVAILABLE = False
    ONNXConverter = None

# Format converters
from .formats import TensorFlowConverter, SafetensorsConverter, GGMLConverter

# Try to import ONNXRuntimeConverter, but handle gracefully if not available
try:
    from .formats import ONNXRuntimeConverter
    ONNX_RUNTIME_CONVERTER_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_CONVERTER_AVAILABLE = False
    ONNXRuntimeConverter = None

# Optimization converters
from .optimization import OptimizationConverter, AdvancedOptimization
from .quantization import QuantizationConverter, AdvancedQuantization
from .distillation import DistillationConverter, AdvancedDistillation
from .hardware import HardwareOptimizedConverter, HardwareOptimizer

# Pipeline
from .pipeline import ConversionPipeline, PipelineConverter

# Manager
from .manager import ModelConversionManager

__all__ = [
    # Base
    'BaseModelConverter',
    
    # Core converters
    'HuggingFaceConverter',
    'OllamaConverter',
]

# Conditionally add ONNX-related exports
if ONNX_CONVERTER_AVAILABLE:
    __all__.append('ONNXConverter')

__all__.extend([
    # Format converters
    'TensorFlowConverter',
    'SafetensorsConverter',
    'GGMLConverter',
    
    # Optimization converters
    'OptimizationConverter',
    'AdvancedOptimization',
    'QuantizationConverter',
    'AdvancedQuantization',
    'DistillationConverter',
    'AdvancedDistillation',
    'HardwareOptimizedConverter',
    'HardwareOptimizer',
    
    # Pipeline
    'ConversionPipeline',
    'PipelineConverter',
    
    # Manager
    'ModelConversionManager',
])

# Conditionally add ONNXRuntimeConverter
if ONNX_RUNTIME_CONVERTER_AVAILABLE:
    __all__.append('ONNXRuntimeConverter') 