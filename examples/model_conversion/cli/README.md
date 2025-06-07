# CLI Examples for Model Conversion

This directory contains command-line interface examples for model conversion.

## Files

- `cli_example.py`: A comprehensive CLI tool for model conversion

## Prerequisites

- Python 3.10 or higher
- MultiMind SDK installed
- Required dependencies (see requirements.txt)

## Usage

The CLI tool provides a flexible interface for model conversion:

```bash
python cli_example.py \
    --model-path /path/to/model \
    --output-path /path/to/output \
    --converter huggingface \
    --device cuda \
    --quantization int8 \
    --validate \
    --metadata
```

## Command-line Arguments

### Required Arguments
- `--model-path`: Path to the source model
- `--output-path`: Path where the converted model should be saved
- `--converter`: Converter to use (choices: huggingface, ollama, custom_onnx)

### Optional Arguments
- `--opset-version`: ONNX opset version (for ONNX conversion)
- `--device`: Device to use (cpu or cuda)
- `--quantization`: Quantization method (int8, int4, fp16)
- `--validate`: Validate the model before and after conversion
- `--metadata`: Print model metadata

## Examples

1. Basic conversion:
```bash
python cli_example.py \
    --model-path ./models/input \
    --output-path ./models/output \
    --converter huggingface
```

2. Conversion with validation and metadata:
```bash
python cli_example.py \
    --model-path ./models/input \
    --output-path ./models/output \
    --converter custom_onnx \
    --opset-version 13 \
    --device cuda \
    --validate \
    --metadata
```

3. Conversion with quantization:
```bash
python cli_example.py \
    --model-path ./models/input \
    --output-path ./models/output \
    --converter ollama \
    --quantization int8 \
    --device cuda
```

## Error Handling

The CLI tool provides clear error messages for:
- Invalid paths
- Missing dependencies
- Conversion failures
- Validation failures

## Output

The tool provides detailed output including:
- Validation results
- Model metadata
- Conversion progress
- Error messages (if any) 