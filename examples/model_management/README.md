# Model Management Examples

This directory contains examples demonstrating how to use the MultiModelWrapper for text-based model management, routing, and optimization.

## Directory Structure

```
model_management/
├── basic/           # Basic usage examples
├── advanced/        # Advanced features
└── training/        # Training utilities
```

## Basic Examples

Located in `basic/`:
- `basic_usage.py`: Basic text generation, chat completion, and embeddings
- `cli_usage.py`: Command-line interface usage
- `api_usage.py`: API interface usage

## Advanced Examples

Located in `advanced/`:
- `advanced_usage.py`: Complex model configuration and task-specific selection
- `advanced_features.py`: Rate limiting, cost tracking, and caching
- `intelligent_switching.py`: Performance-based model selection
- `advanced_intelligent_selection.py`: Content-based model selection
- `ensemble_example.py`: Model ensemble usage

## Training Examples

Located in `training/`:
- `fine_tuning_features.py`: Model fine-tuning and customization
- `training_utils.py`: Training data preparation and validation
- `advanced_training_utils.py`: Extended data format support and augmentation

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-key"
export CLAUDE_API_KEY="your-key"
```

3. Run examples:
```bash
# Basic examples
python basic/basic_usage.py
python basic/cli_usage.py
python basic/api_usage.py

# Advanced examples
python advanced/advanced_usage.py
python advanced/intelligent_switching.py

# Training examples
python training/fine_tuning_features.py
python training/training_utils.py
```

## Notes

- All examples include error handling and logging
- Performance metrics and costs are tracked automatically
- Examples are designed to be run independently
- Each example demonstrates different aspects of model management
- Advanced examples show more complex scenarios and optimizations
- Training examples provide tools for model customization 