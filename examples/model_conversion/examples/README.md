# Model Conversion Examples

This directory contains examples of model conversion using the MultiMind SDK.

## Qwen to Ollama Conversion

This example demonstrates how to convert a Qwen model from HuggingFace to Ollama GGUF format.

### Prerequisites

1. Install required packages:
```bash
pip install huggingface_hub requests
```

2. Make sure Ollama is running:
```bash
ollama serve
```

### Usage

Basic usage:
```bash
python qwen_to_ollama.py
```

Advanced usage with options:
```bash
python qwen_to_ollama.py --model-name Qwen/Qwen1.5-7B-Chat --output-dir ./converted_models --quantization q4_k_m
```

### Available Options

- `--model-name`: HuggingFace model name (default: "Qwen/Qwen1.5-7B-Chat")
- `--output-dir`: Directory to save converted model (default: "./converted_models")
- `--quantization`: Quantization method (default: "q4_k_m")
  - `q4_k_m`: 4-bit quantization with k-means clustering (balanced)
  - `q4_0`: 4-bit quantization (smaller size)
  - `q5_k_m`: 5-bit quantization with k-means clustering (better quality)
  - `q8_0`: 8-bit quantization (highest quality)

### Quantization Methods

1. `q4_k_m` (Recommended)
   - 4-bit quantization
   - Uses k-means clustering for better quality
   - Good balance between size and performance
   - Suitable for most use cases

2. `q4_0`
   - Basic 4-bit quantization
   - Smallest model size
   - Lower quality than q4_k_m
   - Good for resource-constrained environments

3. `q5_k_m`
   - 5-bit quantization
   - Uses k-means clustering
   - Better quality than q4_k_m
   - Larger model size

4. `q8_0`
   - 8-bit quantization
   - Highest quality
   - Largest model size
   - Best for quality-critical applications

### Example Output

```
Downloading Qwen model...
Model downloaded successfully to: ./converted_models/Qwen1.5-7B-Chat

Validating model...
Model validation successful

Converting model to Ollama GGUF format...
Conversion successful! Model saved to: ./converted_models/Qwen1.5-7B-Chat.gguf

Testing converted model...
Model response: "I am Qwen, a large language model trained by Alibaba Cloud. I can help you with various tasks..."
```

### Testing the Conversion

The example includes a comprehensive test suite to verify the conversion quality and model performance. To run the tests:

```bash
python test_qwen_conversion.py --model-path ./converted_models/Qwen1.5-7B-Chat.gguf
```

The test suite includes:

1. Performance Tests
   - Measures generation speed (tokens per second)
   - Tests response time for various prompts
   - Evaluates memory usage

2. Accuracy Tests
   - Mathematical reasoning
   - Code generation
   - Language translation
   - Factual knowledge
   - Creative writing

3. Memory Tests
   - Model size verification
   - Format validation
   - Quantization check

Test results are saved to a JSON file (default: `test_results.json`) with detailed metrics and performance data.

### Troubleshooting

1. Model Download Issues:
   - Check your internet connection
   - Verify the model name is correct
   - Ensure you have enough disk space

2. Conversion Issues:
   - Verify you have enough RAM (at least 16GB recommended)
   - Check if the model format is supported
   - Try a different quantization method

3. Ollama API Issues:
   - Ensure Ollama is running (`ollama serve`)
   - Check if the API is accessible (`curl http://localhost:11434/api/tags`)
   - Verify the model is properly loaded in Ollama

### Performance Considerations

1. Memory Usage:
   - 4-bit quantization: ~4GB RAM
   - 8-bit quantization: ~8GB RAM
   - Add 2-4GB for conversion process

2. Conversion Time:
   - 4-bit quantization: 5-10 minutes
   - 8-bit quantization: 10-20 minutes
   - Depends on CPU and available memory

3. Inference Speed:
   - 4-bit quantization: Faster inference
   - 8-bit quantization: Slower but higher quality
   - GPU acceleration recommended for best performance 