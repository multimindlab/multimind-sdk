"""Edge-case tests for the Model Compression "Beta" feature area
(``multimind.model_conversion.quantization``).

The whole ``multimind.model_conversion`` package hard-imports ``torch`` and
``transformers`` at package-init time (no lazy/optional-import guard), so it
cannot be imported at all in a torch-less environment. Every test in this
file is therefore gated behind ``pytest.importorskip`` so the file collects
cleanly here and runs for real in CI's torch-enabled job.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from multimind.model_conversion.quantization import (  # noqa: E402
    AdvancedQuantization,
    QuantizationConverter,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


class _EmptyModel(nn.Module):
    """A module with zero parameters — a degenerate quantization input."""


def test_custom_quantization_zero_scale_degenerate_input_does_not_raise():
    """scale=0 / zero_point=0 is a degenerate quantization scheme, but assembling
    the qconfig (without running calibration) must not blow up."""
    model = _TinyModel()
    quantizer = AdvancedQuantization()

    result = quantizer.custom_quantization(model, {"scale": 0.0, "zero_point": 0})

    assert result.fc.qconfig is not None


def test_per_layer_quantization_missing_layer_name_raises():
    """A malformed/mismatched layer name in the config must surface clearly,
    not be silently ignored."""
    model = _TinyModel()
    quantizer = AdvancedQuantization()

    with pytest.raises(AttributeError):
        quantizer.per_layer_quantization(model, {"does_not_exist": {"quantization_type": "dynamic"}})


def test_per_layer_quantization_dynamic_round_trip_replaces_layer_and_preserves_values():
    """Regression test: quantize_dynamic() is not in-place by default — the
    quantized module must actually be written back onto the model (previously
    the return value was silently discarded, leaving the layer unquantized).

    Whether quantize_dynamic() actually swaps the layer's *type* depends on
    which quantized backend (fbgemm/x86/qnnpack) the running torch build
    supports, so this only asserts the object identity changed (the exact
    bug being guarded against) rather than a specific resulting type.
    """
    torch.manual_seed(0)
    model = _TinyModel()
    original_layer = model.fc
    x = torch.randn(2, 4)
    original_output = model(x)

    quantizer = AdvancedQuantization()
    result = quantizer.per_layer_quantization(model, {"fc": {"quantization_type": "dynamic"}})

    # The layer object itself must have changed (i.e. the return value of
    # quantize_dynamic() was actually written back, not silently discarded).
    assert result.fc is not original_layer

    quantized_output = result.fc(x)
    assert quantized_output.shape == original_output.shape
    # Dynamic int8 quantization introduces noise but should stay in the same ballpark.
    assert torch.allclose(original_output, quantized_output, atol=0.5)


def test_quantization_converter_validate_missing_file_returns_false():
    converter = QuantizationConverter()
    with patch("torch.load", side_effect=FileNotFoundError):
        assert converter.validate("does/not/exist.pt") is False


def test_quantization_converter_validate_non_module_returns_false():
    """A loadable-but-wrong-type file (e.g. a plain dict, not an nn.Module) is a
    malformed input that validate() must reject, not crash on."""
    converter = QuantizationConverter()
    with patch("torch.load", return_value={"not": "a model"}):
        assert converter.validate("some/path.pt") is False


def test_quantization_converter_get_metadata_zero_parameters_raises():
    """A model with zero parameters is a genuinely degenerate quantization
    input: get_metadata()'s ``next(model.parameters())`` has nothing to pull
    from. This documents the current (unfixed) behavior honestly rather than
    silently swallowing it."""
    converter = QuantizationConverter()
    with patch("torch.load", return_value=_EmptyModel()):
        with pytest.raises(StopIteration):
            converter.get_metadata("some/path.pt")
