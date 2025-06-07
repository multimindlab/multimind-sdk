# Model Conversion Examples

This directory contains examples for converting models between different formats using the MultiMind SDK.

## Directory Structure

```
model_conversion/
├── examples/                  # Example scripts and notebooks
│   ├── converters/           # Individual converter examples
│   │   ├── pytorch_to_gguf.py
│   │   ├── tensorflow_to_tflite.py
│   │   ├── onnx_to_ort.py
│   │   ├── pytorch_to_safetensors.py
│   │   ├── tensorflow_to_onnx.py
│   │   └── README.md
│   ├── qwen_to_ollama.py     # Qwen model conversion example
│   └── README.md
├── docker/                    # Docker-related files
│   ├── Dockerfile            # Container image definition
│   ├── docker-compose.yml    # Service orchestration
│   └── README.md            # Docker usage instructions
├── cli/                      # Command-line interface examples
│   ├── cli_example.py       # CLI tool for model conversion
│   └── README.md           # CLI usage instructions
└── README.md                  # This file
```

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for containerized examples)
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

## Implemented Converters

### 1. PyTorch to GGUF Converter
Converts PyTorch models to GGUF format with advanced quantization options.

```python
from multimind.model_conversion import ModelConversionManager

manager = ModelConversionManager()
converted_path = manager.convert(
    model_path="path/to/model.pt",
    output_path="path/to/output.gguf",
    converter_name="pytorch_to_gguf",
    config={
        "sparsity": 0.3,
        "quantization": "q4_k_m",
        "context_length": 4096,
        "embedding_type": "float32"
    }
)
```

### 2. TensorFlow to TFLite Converter
Converts TensorFlow models to TFLite format with optimization options.

```python
converted_path = manager.convert(
    model_path="path/to/model",
    output_path="path/to/output.tflite",
    converter_name="tensorflow_to_tflite",
    config={
        "quantization": "dynamic",
        "optimizations": ["DEFAULT", "OPTIMIZE_FOR_LATENCY"],
        "supported_ops": "TFLITE_BUILTINS",
        "allow_custom_ops": True
    }
)
```

### 3. ONNX to ONNX Runtime Converter
Converts ONNX models to optimized ONNX Runtime format.

```python
converted_path = manager.convert(
    model_path="path/to/model.onnx",
    output_path="path/to/output.ort",
    converter_name="onnx_to_ort",
    config={
        "optimization_level": "all",
        "providers": ["CPUExecutionProvider", "CUDAExecutionProvider"],
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "execution_mode": "parallel",
        "save_as_external_data": True
    }
)
```

### 4. PyTorch to Safetensors Converter
Converts PyTorch models to Safetensors format with compression options.

```python
converted_path = manager.convert(
    model_path="path/to/model.pt",
    output_path="path/to/output.safetensors",
    converter_name="pytorch_to_safetensors",
    config={
        "compression": "lz4",
        "compression_level": 9,
        "device": "cuda",
        "metadata": {"author": "JohnDoe", "version": "1.0"}
    }
)
```

### 5. TensorFlow to ONNX Converter
Converts TensorFlow models to ONNX format with graph optimization.

```python
converted_path = manager.convert(
    model_path="path/to/model",
    output_path="path/to/output.onnx",
    converter_name="tensorflow_to_onnx",
    config={
        "opset_version": 12,
        "do_constant_folding": True,
        "verbose": True,
        "input_shape": {"input_0": [1, 3, 224, 224]},
        "output_shape": {"output_0": [1, 1000]}
    }
)
```

## Supported Formats and Conversions

### Model Format Types

1. **HuggingFace Formats**
   - PyTorch (.pt, .pth)
   - Safetensors (.safetensors)
   - TensorFlow SavedModel
   - ONNX (.onnx)

2. **Ollama Formats**
   - GGUF (.gguf)
   - GGML (.ggml)

3. **ONNX Formats**
   - Standard ONNX (.onnx)
   - ONNX Runtime Optimized

4. **TensorRT Formats**
   - TensorRT Engine (.engine)

5. **CoreML Formats**
   - CoreML Model (.mlmodel)

### Conversion Paths

1. **From HuggingFace**
   - HuggingFace → ONNX
   - HuggingFace → Ollama
   - HuggingFace → TensorRT
   - HuggingFace → CoreML

2. **From ONNX**
   - ONNX → TensorRT
   - ONNX → CoreML
   - ONNX → Ollama

3. **From Ollama**
   - Ollama → ONNX
   - Ollama → HuggingFace

## Optimization Features

### 1. Model Optimization
- Graph optimization
- Layer fusion
- Memory optimization
- Batch processing

### 2. Hardware Acceleration
- CUDA optimization
- CPU optimization
- Neural Engine support
- GPU memory optimization

### 3. Inference Optimization
- Batch inference
- Streaming inference
- Dynamic batching
- Memory efficient inference

## Best Practices

1. **Format Selection**
   - Consider deployment target
   - Evaluate hardware requirements
   - Balance size vs. performance

2. **Quantization**
   - Use appropriate precision
   - Validate accuracy impact
   - Consider calibration needs

3. **Optimization**
   - Profile performance
   - Monitor memory usage
   - Test with real workloads

4. **Validation**
   - Verify conversion accuracy
   - Test inference speed
   - Check memory usage
   - Validate on target hardware

## Adding Custom Converters

You can create and register custom converters by implementing the `BaseModelConverter` interface:

```python
from multimind.model_conversion import BaseModelConverter

class MyCustomConverter(BaseModelConverter):
    def convert(self, model_path, output_path, config=None):
        # Implement conversion logic
        pass
    
    def validate(self, model_path):
        # Implement validation logic
        pass
    
    def get_metadata(self, model_path):
        # Implement metadata extraction
        pass
```

See the examples in the `examples/converters/` directory for complete implementations of custom converters.

## License

These examples are part of the MultiMind SDK and are subject to its license terms. 