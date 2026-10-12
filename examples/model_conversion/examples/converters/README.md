# Model Converter Examples

This directory contains examples of model conversion using the MultiMind SDK. Each example demonstrates a specific conversion path with advanced options and optimizations.

## Available Converters

### 1. PyTorch to GGUF (`pytorch_to_gguf.py`)
Converts PyTorch models to GGUF format with advanced quantization options.

```bash
python pytorch_to_gguf.py \
    --model-path path/to/model.pt \
    --output-path path/to/output.gguf \
    --sparsity 0.3 \
    --quantization q4_k_m \
    --context-length 4096 \
    --embedding-type float32
```

### 2. TensorFlow to TFLite (`tensorflow_to_tflite.py`)
Converts TensorFlow models to TFLite format with optimization options.

```bash
python tensorflow_to_tflite.py \
    --model-path path/to/model \
    --output-path path/to/output.tflite \
    --quantization dynamic \
    --optimizations DEFAULT OPTIMIZE_FOR_LATENCY \
    --supported-ops TFLITE_BUILTINS \
    --allow-custom-ops
```

### 3. ONNX to ONNX Runtime (`onnx_to_ort.py`)
Converts ONNX models to optimized ONNX Runtime format.

```bash
python onnx_to_ort.py \
    --model-path path/to/model.onnx \
    --output-path path/to/output.ort \
    --optimization-level all \
    --providers CPUExecutionProvider CUDAExecutionProvider \
    --graph-optimization-level ORT_ENABLE_ALL \
    --execution-mode parallel \
    --save-as-external-data
```

### 4. PyTorch to Safetensors (`pytorch_to_safetensors.py`)
Converts PyTorch models to Safetensors format with compression options.

```bash
python pytorch_to_safetensors.py \
    --model-path path/to/model.pt \
    --output-path path/to/output.safetensors \
    --compression lz4 \
    --compression-level 9 \
    --device cuda \
    --metadata author=JohnDoe version=1.0
```

### 5. TensorFlow to ONNX (`tensorflow_to_onnx.py`)
Converts TensorFlow models to ONNX format with graph optimization.

```bash
python tensorflow_to_onnx.py \
    --model-path path/to/model \
    --output-path path/to/output.onnx \
    --opset-version 12 \
    --do-constant-folding \
    --verbose \
    --input-shape input_0=1,3,224,224 \
    --output-shape output_0=1,1000
```

## Common Features

All converters support:

1. **Validation**
   - Format validation
   - Metadata extraction
   - Compatibility checking

2. **Optimization**
   - Model pruning
   - Quantization
   - Graph optimization
   - Hardware-specific tuning

3. **Metadata**
   - Format information
   - Model statistics
   - Conversion parameters

## Best Practices

### 1. Model Preparation
- Validate source model
- Check format compatibility
- Prepare calibration data

### 2. Conversion Options
- Choose appropriate quantization
- Set optimization levels
- Configure hardware settings

### 3. Quality Assurance
- Verify converted model
- Test performance
- Check accuracy

## Advanced Usage

### Custom Optimization Pipeline
```python
pipeline_config = {
    "pipeline": [
        {
            "step": "validate",
            "checks": ["format", "metadata"]
        },
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

### Format-Specific Options
```python
format_options = {
    "gguf": {
        "context_length": 4096,
        "embedding_type": "float32"
    },
    "tflite": {
        "optimizations": ["DEFAULT"],
        "supported_ops": ["TFLITE_BUILTINS"]
    }
}
```

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

To add new converter examples:
1. Follow existing code structure
2. Include comprehensive options
3. Add error handling
4. Update documentation

## License

These examples are part of the MultiMind SDK and are subject to its license terms. 