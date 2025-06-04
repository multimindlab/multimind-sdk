# Multi-Model Examples

This directory contains examples demonstrating how to use the `MultiModelWrapper` in various contexts.

## Examples

1. `basic_usage.py`: Basic usage examples including:
   - Text generation
   - Chat completion
   - Streaming responses
   - Embeddings generation

2. `advanced_usage.py`: Advanced usage scenarios including:
   - Complex model configuration with multiple fallbacks
   - Task-specific model selection
   - Complex chat conversations with context management
   - Batch processing with different models
   - Streaming with automatic fallback

3. `intelligent_switching.py`: Demonstrates intelligent model selection and optimization:
   - Task-specific model selection based on content type
   - Performance-based model selection using metrics
   - Automatic fallback with performance tracking
   - Complex conversations with dynamic model selection
   - Batch processing with intelligent model selection
   - Performance metrics tracking and analysis

4. `advanced_intelligent_selection.py`: Advanced intelligent model selection scenarios:
   - Content-based model selection for different types of tasks
   - Context-aware model selection in conversations
   - Performance-based dynamic weighting
   - Task-specific optimization for specialized tasks
   - Adaptive model selection based on query complexity
   - Comprehensive performance metrics analysis

5. `advanced_features.py`: Demonstrates extended MultiModelWrapper features:
   - Rate limiting and quota management
   - Cost tracking and optimization
   - Model specialty analysis
   - Advanced error handling and recovery
   - Performance optimization with different configurations
   - Comprehensive metrics and analytics
   - Caching strategies
   - Dynamic model selection based on multiple factors

6. `fine_tuning_features.py`: Demonstrates fine-tuning and advanced model management:
   - Model fine-tuning with custom training data
   - Training data preparation and validation
   - Model evaluation and metrics tracking
   - Custom model configuration creation
   - Model version comparison
   - Advanced model management
   - Performance evaluation and benchmarking
   - Model version tracking and history

7. `training_utils.py`: Demonstrates custom training data preparation and model version management:
   - Training data loading from multiple formats (JSON, CSV, JSONL)
   - Training data validation and statistics
   - Training/validation split management
   - Model version creation and tracking
   - Training history and metrics recording
   - Version comparison and analysis
   - Model metadata management
   - Performance metrics tracking

8. `advanced_training_utils.py`: Demonstrates advanced training utilities:
   - Extended data format support (YAML, XML, Parquet, Excel)
   - Data augmentation with synonyms and paraphrasing
   - Advanced data quality analysis
   - Training metrics visualization
   - Version comparison visualization
   - Confusion matrix visualization
   - Metric distribution analysis
   - Training history tracking
   - Data quality metrics visualization

9. `cli_usage.py`: Command-line interface usage including:
   - Text generation
   - Chat completion
   - Embeddings generation

10. `api_usage.py`: API interface usage including:
    - REST API text generation
    - Chat completion
    - Embeddings generation
    - Health check endpoint

## Running the Examples

1. Basic Usage:
```bash
python basic_usage.py
```

2. Advanced Usage:
```bash
python advanced_usage.py
```

3. Intelligent Switching:
```bash
python intelligent_switching.py
```

4. Advanced Intelligent Selection:
```bash
python advanced_intelligent_selection.py
```

5. Advanced Features:
```bash
python advanced_features.py
```

6. Fine-tuning Features:
```bash
python fine_tuning_features.py
```

7. Training Utilities:
```bash
python training_utils.py
```

8. Advanced Training Utilities:
```bash
python advanced_training_utils.py
```

9. CLI Usage:
```bash
python -m multimind.cli.multi_model_cli generate --primary-model openai "Your prompt"
```

10. API Usage:
```bash
# Start the API server
uvicorn multimind.api.multi_model_api:app --reload

# Then use the API client
python api_usage.py
```

## Requirements

- Python 3.8+
- Required packages:
  - fastapi
  - uvicorn
  - click
  - aiohttp
  - pydantic
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - nltk
  - pyyaml
  - openpyxl
  - pyarrow

## Environment Variables

Set the following environment variables:
- `OPENAI_API_KEY`
- `CLAUDE_API_KEY`

## Notes

- The examples demonstrate different ways to use the `MultiModelWrapper`
- Basic examples show simple usage patterns
- Advanced examples show more complex scenarios
- Intelligent switching examples demonstrate automatic model selection and optimization
- Advanced intelligent selection shows sophisticated model selection strategies
- Advanced features demonstrate extended functionality like rate limiting and cost tracking
- Fine-tuning features show how to customize and optimize models for specific tasks
- Training utilities provide tools for data preparation and model version management
- Advanced training utilities provide extended data format support, augmentation, and visualization
- CLI examples show command-line usage
- API examples show how to use the REST API interface
- The default API server runs at `http://localhost:8000`
- API documentation is available at `http://localhost:8000/docs` 