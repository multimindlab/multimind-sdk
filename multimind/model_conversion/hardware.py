from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from .base import BaseModelConverter

class HardwareOptimizer:
    """Hardware-specific optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def optimize_for_gpu(
        self,
        model: nn.Module,
        gpu_config: Dict[str, Any]
    ) -> nn.Module:
        """Optimize model for GPU execution."""
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            # Set CUDA device
            device_id = gpu_config.get('device_id', 0)
            torch.cuda.set_device(device_id)
            
            # Enable cuDNN benchmarking
            if gpu_config.get('enable_cudnn_benchmark', True):
                torch.backends.cudnn.benchmark = True
            
            # Enable tensor cores if available
            if gpu_config.get('enable_tensor_cores', True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable mixed precision
            if gpu_config.get('enable_mixed_precision', True):
                model = model.half()
            
            # Enable memory optimization
            if gpu_config.get('enable_memory_optimization', True):
                torch.cuda.empty_cache()
        
        return model
    
    def optimize_for_cpu(
        self,
        model: nn.Module,
        cpu_config: Dict[str, Any]
    ) -> nn.Module:
        """Optimize model for CPU execution."""
        # Enable MKL optimizations
        if cpu_config.get('enable_mkl', True):
            torch.backends.mkl.enabled = True
        
        # Enable OpenMP optimizations
        if cpu_config.get('enable_openmp', True):
            torch.set_num_threads(cpu_config.get('num_threads', 4))
        
        # Enable memory optimization
        if cpu_config.get('enable_memory_optimization', True):
            torch.cuda.empty_cache()
        
        return model
    
    def optimize_for_mobile(
        self,
        model: nn.Module,
        mobile_config: Dict[str, Any]
    ) -> nn.Module:
        """Optimize model for mobile deployment."""
        # Enable quantization
        if mobile_config.get('enable_quantization', True):
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        # Enable pruning
        if mobile_config.get('enable_pruning', True):
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.utils.prune.l1_unstructured(
                        module,
                        name='weight',
                        amount=mobile_config.get('pruning_amount', 0.3)
                    )
        
        return model
    
    def optimize_for_edge(
        self,
        model: nn.Module,
        edge_config: Dict[str, Any]
    ) -> nn.Module:
        """Optimize model for edge devices."""
        # Enable quantization
        if edge_config.get('enable_quantization', True):
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        
        # Enable pruning
        if edge_config.get('enable_pruning', True):
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.utils.prune.l1_unstructured(
                        module,
                        name='weight',
                        amount=edge_config.get('pruning_amount', 0.3)
                    )
        
        # Enable memory optimization
        if edge_config.get('enable_memory_optimization', True):
            torch.cuda.empty_cache()
        
        return model

class HardwareOptimizedConverter(BaseModelConverter):
    """Converter with hardware-specific optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.optimizer = HardwareOptimizer(config)
    
    def convert(
        self,
        model_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert model with hardware-specific optimizations."""
        config = config or {}
        model = torch.load(model_path)
        
        # Apply hardware-specific optimizations
        if config.get('target_hardware') == 'gpu':
            model = self.optimizer.optimize_for_gpu(
                model,
                config.get('gpu_config', {})
            )
        elif config.get('target_hardware') == 'cpu':
            model = self.optimizer.optimize_for_cpu(
                model,
                config.get('cpu_config', {})
            )
        elif config.get('target_hardware') == 'mobile':
            model = self.optimizer.optimize_for_mobile(
                model,
                config.get('mobile_config', {})
            )
        elif config.get('target_hardware') == 'edge':
            model = self.optimizer.optimize_for_edge(
                model,
                config.get('edge_config', {})
            )
        
        # Save optimized model
        torch.save(model, output_path)
        return output_path
    
    def validate(self, model_path: str) -> bool:
        """Validate if model can be optimized."""
        try:
            model = torch.load(model_path)
            return isinstance(model, nn.Module)
        except Exception:
            return False
    
    def get_metadata(self, model_path: str) -> Dict[str, Any]:
        """Get hardware optimization metadata."""
        model = torch.load(model_path)
        return {
            'device': str(next(model.parameters()).device),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            ) / (1024 * 1024),
            'is_quantized': any(
                isinstance(m, torch.quantized.QDynamicLinear)
                for m in model.modules()
            ),
            'is_pruned': any(
                hasattr(m, 'mask')
                for m in model.modules()
            )
        } 