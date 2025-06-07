# Supported Model Conversions and Formats

## Model Format Types

### 1. HuggingFace Formats
- **PyTorch (.pt, .pth)**
  - Full precision (FP32)
  - Half precision (FP16)
  - Mixed precision
- **Safetensors (.safetensors)**
  - Secure tensor storage
  - Memory efficient
  - Safe loading
- **TensorFlow SavedModel**
  - TF2.x compatible
  - Graph and weights
- **ONNX (.onnx)**
  - Optimized for inference
  - Cross-platform

### 2. Ollama Formats
- **GGUF (.gguf)**
  - Quantized formats
  - Optimized for CPU/GPU
- **GGML (.ggml)**
  - Legacy format
  - CPU optimized

### 3. ONNX Formats
- **Standard ONNX (.onnx)**
  - Opset versions 7-15
  - Dynamic shapes
  - Static shapes
- **ONNX Runtime Optimized**
  - Quantized (INT8, INT4)
  - FP16 optimized
  - Graph optimized

### 4. TensorRT Formats
- **TensorRT Engine (.engine)**
  - FP32, FP16, INT8
  - Dynamic shapes
  - Static shapes

### 5. CoreML Formats
- **CoreML Model (.mlmodel)**
  - iOS/macOS optimized
  - Neural Engine support

## Conversion Paths

### From HuggingFace
1. **HuggingFace → ONNX**
   - Full precision
   - Quantized (INT8)
   - Dynamic axes
   - Static shapes

2. **HuggingFace → Ollama**
   - GGUF format
   - Various quantization levels
   - CPU/GPU optimized

3. **HuggingFace → TensorRT**
   - FP32/FP16/INT8
   - Dynamic/Static shapes
   - GPU optimized

4. **HuggingFace → CoreML**
   - iOS/macOS deployment
   - Neural Engine optimization

### From ONNX
1. **ONNX → TensorRT**
   - Engine optimization
   - Precision calibration
   - Dynamic/Static shapes

2. **ONNX → CoreML**
   - iOS/macOS deployment
   - Neural Engine support

3. **ONNX → Ollama**
   - GGUF conversion
   - Quantization

### From Ollama
1. **Ollama → ONNX**
   - Standard ONNX format
   - Dynamic shapes

2. **Ollama → HuggingFace**
   - PyTorch format
   - Safetensors format

## Quantization Options

### Precision Levels
1. **Full Precision**
   - FP32 (32-bit float)
   - FP16 (16-bit float)
   - BF16 (Brain floating point)

2. **Quantized**
   - INT8 (8-bit integer)
   - INT4 (4-bit integer)
   - Mixed precision

### Quantization Methods
1. **Static Quantization**
   - Calibration required
   - Fixed scale/zero-point
   - Better performance

2. **Dynamic Quantization**
   - No calibration needed
   - Runtime scaling
   - More flexible

3. **Quantization-Aware Training**
   - Training with quantization
   - Better accuracy
   - More complex setup

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

## Usage Examples

### Basic Conversion
```python
# HuggingFace to ONNX
converted_path = manager.convert(
    model_path="path/to/model",
    output_path="path/to/output",
    converter_name="huggingface",
    config={
        "format": "onnx",
        "opset_version": 12,
        "quantization": "int8"
    }
)
```

### Advanced Conversion
```python
# HuggingFace to TensorRT with calibration
converted_path = manager.convert(
    model_path="path/to/model",
    output_path="path/to/output",
    converter_name="huggingface",
    config={
        "format": "tensorrt",
        "precision": "int8",
        "calibration": {
            "method": "entropy",
            "dataset": "calibration_data",
            "batch_size": 32
        },
        "optimization": {
            "graph_optimization": True,
            "layer_fusion": True,
            "memory_optimization": True
        }
    }
)
```

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