# Qwen Model Conversion Test Suite

This test suite provides comprehensive testing for Qwen model conversions to Ollama GGUF format. It evaluates model performance, accuracy, and memory usage to ensure high-quality conversions.

## Features

- Performance benchmarking
- Accuracy validation
- Memory usage analysis
- Detailed test reporting
- JSON result export

## Prerequisites

1. Python 3.8 or higher
2. Required packages:
```bash
pip install requests
```

3. Ollama running locally:
```bash
ollama serve
```

## Usage

Basic usage:
```bash
python test_qwen_conversion.py --model-path ./converted_models/Qwen1.5-7B-Chat.gguf
```

Advanced usage with options:
```bash
python test_qwen_conversion.py \
    --model-path ./converted_models/Qwen1.5-7B-Chat.gguf \
    --output-file detailed_results.json \
    --ollama-host http://localhost:11434
```

### Command Line Arguments

- `--model-path`: Path to the converted model file (required)
- `--output-file`: JSON file to save test results (default: "test_results.json")
- `--ollama-host`: Ollama API host URL (default: "http://localhost:11434")

## Test Categories

### 1. Performance Tests

Evaluates model generation speed and efficiency:

- **Generation Speed**
  - Measures tokens per second
  - Calculates response time
  - Tracks total tokens generated

- **Test Prompts**
  - Quantum computing explanation
  - AI poetry generation
  - Climate change summary
  - Blockchain technology explanation
  - Photosynthesis description

### 2. Accuracy Tests

Validates model output quality and correctness:

- **Mathematical Reasoning**
  - Problem: "If a train travels at 60 mph for 2.5 hours, how far does it go?"
  - Expected keywords: ["150", "miles", "distance", "calculation"]

- **Code Generation**
  - Task: "Write a Python function to calculate the Fibonacci sequence"
  - Expected keywords: ["def", "fibonacci", "return", "recursive"]

- **Language Translation**
  - Input: "Translate 'Hello, how are you?' to Spanish"
  - Expected keywords: ["hola", "como", "estas"]

- **Factual Knowledge**
  - Question: "What is the capital of France?"
  - Expected keywords: ["paris", "france", "capital"]

- **Creative Writing**
  - Prompt: "Write a short story about a robot learning to paint"
  - Expected keywords: ["robot", "paint", "art", "learn"]

### 3. Memory Tests

Analyzes model resource usage:

- Model size verification
- Format validation
- Quantization check
- Memory footprint analysis

## Test Results

The test suite generates a JSON file with detailed results:

```json
{
  "performance_tests": [
    {
      "status": "success",
      "prompt": "Explain quantum computing...",
      "response": "...",
      "generation_time": 2.5,
      "tokens_per_second": 40.0,
      "total_tokens": 100
    }
  ],
  "accuracy_tests": [
    {
      "test_case": "mathematical_reasoning",
      "prompt": "If a train travels...",
      "expected_keywords": ["150", "miles"],
      "response": "...",
      "contains_keywords": true
    }
  ],
  "memory_test": {
    "status": "success",
    "model_size": 3500000000,
    "format": "gguf",
    "quantization": "q4_k_m"
  },
  "timestamp": "2024-03-14 12:00:00"
}
```

## Performance Metrics

### Generation Speed
- Tokens per second (higher is better)
- Response time (lower is better)
- Total tokens generated

### Accuracy Score
- Keyword match rate
- Response quality
- Task completion rate

### Memory Usage
- Model size
- Format compliance
- Quantization verification

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Ensure Ollama is running
   - Check API host URL
   - Verify network connectivity

2. **Model Loading Issues**
   - Verify model path
   - Check file permissions
   - Ensure correct format

3. **Performance Issues**
   - Check system resources
   - Verify quantization
   - Monitor memory usage

### Error Messages

- `ConnectionError`: Ollama API not accessible
- `FileNotFoundError`: Model file not found
- `ValueError`: Invalid model format
- `MemoryError`: Insufficient system memory

## Best Practices

1. **Test Environment**
   - Use dedicated test machine
   - Ensure sufficient resources
   - Monitor system metrics

2. **Test Execution**
   - Run tests in isolation
   - Avoid concurrent conversions
   - Monitor system resources

3. **Result Analysis**
   - Compare with baseline
   - Track performance trends
   - Document anomalies

## Contributing

To add new test cases:

1. Add test case to `accuracy_test_cases` in `run_all_tests()`
2. Update documentation
3. Verify test coverage
4. Submit pull request

## License

This test suite is part of the MultiMind SDK and is subject to its license terms. 