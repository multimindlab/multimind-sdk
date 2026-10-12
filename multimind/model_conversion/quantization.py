from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseModelConverter

# Preferred order: fbgemm/x86 give the best accuracy on server CPUs; qnnpack
# is the only engine available on ARM (e.g. Apple Silicon). torch defaults
# torch.backends.quantized.engine to "none", so quantize_dynamic() silently
# no-ops unless an engine is selected first.
_ENGINE_PREFERENCE = ("fbgemm", "x86", "qnnpack")


def _select_quantized_engine() -> str:
    supported = set(torch.backends.quantized.supported_engines)
    for engine in _ENGINE_PREFERENCE:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(
        f"No supported torch quantized engine available (supported: {sorted(supported)})"
    )


class AdvancedQuantization:
    """Advanced quantization techniques for model conversion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def quantize_aware_training(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Quantization-aware training implementation."""
        config = config or {}
        engine = _select_quantized_engine()

        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qat_qconfig(engine)
        torch.quantization.prepare_qat(model, inplace=True)

        # Training loop for quantization
        if calibration_data is not None:
            model.train()
            for _ in range(config.get("calibration_steps", 100)):
                model(calibration_data)

        # Convert to quantized model
        model.eval()
        torch.quantization.convert(model, inplace=True)

        return model

    def per_layer_quantization(
        self, model: nn.Module, layer_configs: Dict[str, Dict[str, Any]]
    ) -> nn.Module:
        """Apply different quantization schemes to different layers."""
        engine = _select_quantized_engine()
        for layer_name, layer_config in layer_configs.items():
            layer = getattr(model, layer_name)
            if layer_config.get("quantization_type") == "dynamic":
                # quantize_dynamic() is not in-place by default; it returns a new
                # module. The return value must be reassigned onto the model,
                # otherwise the layer silently stays unquantized.
                quantized_layer = torch.quantization.quantize_dynamic(
                    layer, {torch.nn.Linear}, dtype=torch.qint8
                )
                setattr(model, layer_name, quantized_layer)
            elif layer_config.get("quantization_type") == "static":
                layer.qconfig = torch.quantization.get_default_qconfig(engine)
                torch.quantization.prepare(layer, inplace=True)
                torch.quantization.convert(layer, inplace=True)

        return model

    def custom_quantization(
        self, model: nn.Module, quantization_scheme: Dict[str, Any]
    ) -> nn.Module:
        """Apply custom quantization scheme to model."""
        # Define custom quantization parameters
        scale = quantization_scheme.get("scale", 1.0)
        zero_point = quantization_scheme.get("zero_point", 0)
        dtype = quantization_scheme.get("dtype", torch.qint8)

        # Apply custom quantization
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.FakeQuantize.with_args(
                        observer=torch.quantization.MinMaxObserver,
                        scale=scale,
                        zero_point=zero_point,
                        dtype=dtype,
                    ),
                    weight=torch.quantization.FakeQuantize.with_args(
                        observer=torch.quantization.MinMaxObserver,
                        scale=scale,
                        zero_point=zero_point,
                        dtype=dtype,
                    ),
                )

        return model


class QuantizationConverter(BaseModelConverter):
    """Converter with advanced quantization capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.quantizer = AdvancedQuantization(config)

    def convert(
        self, model_path: str, output_path: str, config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert model with advanced quantization."""
        config = config or {}
        model = torch.load(model_path)

        # Apply quantization based on config
        if config.get("quantization_type") == "aware_training":
            model = self.quantizer.quantize_aware_training(
                model, config.get("calibration_data"), config
            )
        elif config.get("quantization_type") == "per_layer":
            model = self.quantizer.per_layer_quantization(model, config.get("layer_configs", {}))
        elif config.get("quantization_type") == "custom":
            model = self.quantizer.custom_quantization(model, config.get("quantization_scheme", {}))

        # Save quantized model
        torch.save(model, output_path)
        return output_path

    def validate(self, model_path: str) -> bool:
        """Validate if model can be quantized."""
        try:
            model = torch.load(model_path)
            return isinstance(model, nn.Module)
        except Exception:
            return False

    def get_metadata(self, model_path: str) -> Dict[str, Any]:
        """Get quantization metadata."""
        model = torch.load(model_path)
        return {
            "quantization_type": getattr(model, "qconfig", None),
            "dtype": next(model.parameters()).dtype,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
