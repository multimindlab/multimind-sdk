from typing import Dict, Any, Optional, List, Union, Callable
import torch
import torch.nn as nn
from .base import BaseModelConverter
from .quantization import QuantizationConverter
from .optimization import OptimizationConverter
from .distillation import DistillationConverter
from .hardware import HardwareOptimizedConverter

class ConversionPipeline:
    """Advanced model conversion pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.stages = []
        self.converters = {
            'quantization': QuantizationConverter(),
            'optimization': OptimizationConverter(),
            'distillation': DistillationConverter(),
            'hardware': HardwareOptimizedConverter()
        }
    
    def add_stage(
        self,
        stage_type: str,
        stage_config: Dict[str, Any],
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> None:
        """Add a stage to the conversion pipeline."""
        self.stages.append({
            'type': stage_type,
            'config': stage_config,
            'condition': condition
        })
    
    def execute(
        self,
        model_path: str,
        output_path: str,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute the conversion pipeline."""
        pipeline_config = pipeline_config or {}
        current_model_path = model_path
        
        for stage in self.stages:
            # Check if stage should be executed
            if stage['condition'] and not stage['condition'](pipeline_config):
                continue
            
            # Get converter for stage
            converter = self.converters.get(stage['type'])
            if not converter:
                raise ValueError(f"Unknown converter type: {stage['type']}")
            
            # Validate model
            if not converter.validate(current_model_path):
                raise ValueError(
                    f"Model validation failed for stage: {stage['type']}"
                )
            
            # Execute conversion
            stage_output_path = f"{output_path}.{stage['type']}"
            current_model_path = converter.convert(
                current_model_path,
                stage_output_path,
                stage['config']
            )
        
        # Move final model to output path
        if current_model_path != output_path:
            torch.save(torch.load(current_model_path), output_path)
        
        return output_path
    
    def get_pipeline_metadata(self, model_path: str) -> Dict[str, Any]:
        """Get metadata for the entire pipeline."""
        metadata = {}
        
        for stage in self.stages:
            converter = self.converters.get(stage['type'])
            if converter:
                stage_metadata = converter.get_metadata(model_path)
                metadata[stage['type']] = stage_metadata
        
        return metadata

class PipelineConverter(BaseModelConverter):
    """Converter with advanced pipeline capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.pipeline = ConversionPipeline(config)
    
    def convert(
        self,
        model_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert model using the advanced pipeline."""
        config = config or {}
        
        # Add stages based on config
        if config.get('enable_quantization', True):
            self.pipeline.add_stage(
                'quantization',
                config.get('quantization_config', {}),
                lambda c: c.get('enable_quantization', True)
            )
        
        if config.get('enable_optimization', True):
            self.pipeline.add_stage(
                'optimization',
                config.get('optimization_config', {}),
                lambda c: c.get('enable_optimization', True)
            )
        
        if config.get('enable_distillation', False):
            self.pipeline.add_stage(
                'distillation',
                config.get('distillation_config', {}),
                lambda c: c.get('enable_distillation', False)
            )
        
        if config.get('enable_hardware_optimization', True):
            self.pipeline.add_stage(
                'hardware',
                config.get('hardware_config', {}),
                lambda c: c.get('enable_hardware_optimization', True)
            )
        
        # Execute pipeline
        return self.pipeline.execute(model_path, output_path, config)
    
    def validate(self, model_path: str) -> bool:
        """Validate if model can be processed by the pipeline."""
        try:
            model = torch.load(model_path)
            return isinstance(model, nn.Module)
        except Exception:
            return False
    
    def get_metadata(self, model_path: str) -> Dict[str, Any]:
        """Get pipeline metadata."""
        return self.pipeline.get_pipeline_metadata(model_path) 