# Advanced Model Conversion Example

This example demonstrates comprehensive model conversion capabilities of the MultiMind SDK, supporting multiple input and output formats with advanced optimization options.

## Features

- Multi-format conversion support
- Parallel conversion pipeline
- Advanced optimization options
- Metadata extraction
- Format validation

## Supported Formats

### Input Formats
- PyTorch (.pt, .pth)
- TensorFlow (SavedModel)
- ONNX (.onnx)

### Output Formats
- GGUF (.gguf)
- ONNX (.onnx)
- TensorFlow Lite (.tflite)
- Safetensors (.safetensors)
- ONNX Runtime (.ort)

## Prerequisites

1. Install required packages:
```bash
pip install torch tensorflow onnx onnxruntime safetensors
```

2. For GPU support (optional):
```bash
pip install torch-cuda tensorflow-gpu
```

## Usage

Basic usage:
```bash
python advanced_conversion_example.py \
    --model-path path/to/model \
    --source-format pytorch \
    --output-dir ./converted_models
```

Advanced usage with multiple target formats:
```bash
python advanced_conversion_example.py \
    --model-path path/to/model \
    --source-format pytorch \
    --output-dir ./converted_models \
    --target-formats gguf onnx tflite safetensors
```

### Command Line Arguments

- `--model-path`: Path to the source model (required)
- `--output-dir`: Directory to save converted models (default: "./converted_models")
- `--source-format`: Source model format (required, choices: pytorch, tensorflow, onnx)
- `--target-formats`: Target formats for conversion (default: all supported formats)

## Conversion Pipeline

The example implements a comprehensive conversion pipeline:

1. **Validation**
   - Format validation
   - Metadata extraction
   - Compatibility check

2. **Optimization**
   - Model pruning
   - Dynamic quantization
   - Hardware-specific optimization

3. **Conversion**
   - Parallel format conversion
   - Format-specific optimization
   - Metadata preservation

## Example Output

```
Conversion Results:
------------------
Converted model saved to: ./converted_models/model.gguf
Converted model saved to: ./converted_models/model.onnx
Converted model saved to: ./converted_models/model.tflite
Converted model saved to: ./converted_models/model.safetensors

Model Metadata:
--------------
Model: ./converted_models/model.gguf
format: gguf
quantization: q4_k_m
size_mb: 3500.0
...

Model: ./converted_models/model.onnx
format: onnx
version: 1.12.0
ir_version: 8
...
```

## Advanced Features

### 1. Parallel Conversion
The example supports parallel conversion to multiple formats:
```python
converter = ModelConversionManager(
    source_format="pytorch",
    target_formats=["gguf", "onnx", "tflite"],
    parallel=True
)
```

### 2. Custom Optimization
Configure optimization parameters:
```python
pipeline_config = {
    "pipeline": [
        {
            "step": "optimize",
            "methods": ["pruning", "quantization"],
            "config": {
                "pruning": {"sparsity": 0.3},
                "quantization": {"method": "dynamic"}
            }
        }
    ]
}
```

### 3. Format-Specific Options
Set format-specific conversion options:
```python
converter = ModelConversionManager(
    source_format="pytorch",
    target_format="gguf",
    format_options={
        "gguf": {
            "context_length": 4096,
            "embedding_type": "float32"
        }
    }
)
```

## Best Practices

1. **Model Selection**
   - Choose appropriate source format
   - Consider target deployment environment
   - Verify model compatibility

2. **Optimization Strategy**
   - Start with dynamic quantization
   - Apply pruning for size reduction
   - Use hardware-specific optimization

3. **Quality Assurance**
   - Validate converted models
   - Test performance metrics
   - Verify accuracy

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Use model splitting
   - Enable gradient checkpointing

2. **Conversion Failures**
   - Check format compatibility
   - Verify model structure
   - Update conversion tools

3. **Performance Issues**
   - Optimize quantization
   - Apply hardware-specific tuning
   - Use appropriate format

## Contributing

To add new format support:
1. Implement format converter
2. Add format validation
3. Create test cases
4. Update documentation

## License

This example is part of the MultiMind SDK and is subject to its license terms. 