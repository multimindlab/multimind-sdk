# MultiMind SDK Model Conversion CLI

A command-line interface for converting models between different formats using the MultiMind SDK.

## Installation

The CLI is automatically installed with the MultiMind SDK:

```bash
pip install multimind-sdk
```

## Usage

Basic usage:
```bash
multimind convert --source <source_format> --target <target_format> --model-path <path> --output-dir <dir>
```

### Examples

1. Convert HuggingFace model to GGUF:
```bash
multimind convert \
    --source huggingface \
    --target gguf \
    --model-path Qwen/Qwen1.5-7B \
    --output-dir ./models \
    --quantization q4_k_m \
    --context-length 4096 \
    --validate \
    --test
```

2. Convert PyTorch model to Safetensors:
```bash
multimind convert \
    --source pytorch \
    --target safetensors \
    --model-path ./model.pt \
    --output-dir ./converted \
    --compression lz4 \
    --compression-level 9 \
    --device cuda \
    --metadata author=JohnDoe version=1.0
```

3. Convert TensorFlow model to TFLite:
```bash
multimind convert \
    --source tensorflow \
    --target tflite \
    --model-path ./model \
    --output-dir ./converted \
    --optimizations DEFAULT OPTIMIZE_FOR_LATENCY \
    --quantization int8
```

4. Convert ONNX model to ONNX Runtime:
```bash
multimind convert \
    --source onnx \
    --target ort \
    --model-path ./model.onnx \
    --output-dir ./converted \
    --optimization-level all \
    --device cuda
```

## Supported Formats

### Source Formats
- `huggingface`: HuggingFace models
- `pytorch`: PyTorch models
- `tensorflow`: TensorFlow models
- `onnx`: ONNX models
- `ollama`: Ollama models

### Target Formats
- `gguf`: GGUF format (for Ollama)
- `safetensors`: Safetensors format
- `tflite`: TensorFlow Lite format
- `ort`: ONNX Runtime format
- `onnx`: ONNX format

## Options

### Required Arguments
- `--source`: Source model format
- `--target`: Target model format
- `--model-path`: Path to source model or HuggingFace model ID
- `--output-dir`: Directory to save converted model

### Optional Arguments
- `--quantization`: Quantization method
  - For GGUF: `q4_k_m`, `q4_0`, `q5_k_m`, `q8_0`
  - For TFLite: `int8`, `fp16`
- `--compression`: Compression method for Safetensors
  - `lz4`: Fast compression
  - `zstd`: Better compression ratio
- `--compression-level`: Compression level (1-9)
- `--optimizations`: Optimization methods
  - For TFLite: `DEFAULT`, `OPTIMIZE_FOR_LATENCY`, etc.
- `--optimization-level`: ONNX Runtime optimization level
  - `basic`: Basic optimizations
  - `all`: All optimizations
  - `extreme`: Maximum optimizations
- `--device`: Device to use
  - `cpu`: CPU only
  - `cuda`: GPU acceleration
- `--context-length`: Context length for GGUF models
- `--metadata`: Additional metadata (key=value pairs)
- `--validate`: Validate model before and after conversion
- `--test`: Test converted model
- `--verbose`: Enable verbose output

## Best Practices

1. **Model Validation**
   - Always use `--validate` for production conversions
   - Check model metadata with `--verbose`

2. **Optimization**
   - Use appropriate quantization for your use case
   - Consider hardware constraints
   - Test performance impact

3. **Testing**
   - Use `--test` to verify converted models
   - Test with real-world inputs
   - Monitor performance metrics

## Troubleshooting

1. **Memory Issues**
   - Reduce batch size
   - Use CPU if GPU memory is insufficient
   - Try different quantization methods

2. **Conversion Failures**
   - Check format compatibility
   - Verify model structure
   - Update conversion tools

3. **Performance Issues**
   - Adjust optimization levels
   - Try different quantization methods
   - Monitor hardware usage

## Contributing

To add support for new formats or features:
1. Implement the converter in `multimind/model_conversion/`
2. Add format validation
3. Update the CLI interface
4. Add tests and documentation 