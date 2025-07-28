# Model Management Examples

This document provides detailed information about the model management examples in the MultiMind SDK.

## Overview

The model management examples demonstrate how to:
- Initialize and configure models
- Handle model switching and fallbacks
- Optimize costs and performance
- Implement intelligent routing
- Track metrics and costs

## Basic Examples

### Basic Usage

The `basic_usage.py` example demonstrates fundamental model operations:

```python
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

async def basic_example():
    # Initialize
    factory = ModelFactory()
    wrapper = MultiModelWrapper(
        model_factory=factory,
        primary_model="gpt-3.5-turbo",
        fallback_models=["gpt-4", "claude"]
    )
    
    async def example_model_management():
        # Generate text
        response = await wrapper.generate(
            prompt="Explain quantum computing",
            temperature=0.7
        )
        
        # Chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like?"}
        ]
        chat_response = await wrapper.chat(messages=messages)
```

### Key Features

1. **Model Initialization**
   - Factory pattern for model creation
   - Configuration management
   - Environment variable handling

2. **Text Generation**
   - Temperature control
   - Max tokens setting
   - Response formatting

3. **Chat Completion**
   - Message history management
   - Role-based interactions
   - Context preservation

## Advanced Examples

### Cost Optimization

The `cost_optimization.py` example shows how to optimize API costs:

```python
from multimind.models.advanced import CostOptimizedWrapper

async def cost_optimization_example():
    wrapper = CostOptimizedWrapper(
        model_factory=factory,
        primary_model="gpt-3.5-turbo",
        fallback_models=["gpt-4", "claude"],
        budget=0.1
    )
    
    # Generate with cost tracking
    response = await wrapper.generate(
        prompt="Complex query",
        track_cost=True
    )
    
    # Get cost metrics
    metrics = wrapper.get_cost_metrics()
```

### Features

1. **Budget Management**
   - Per-request budget limits
   - Monthly budget tracking
   - Cost alerts

2. **Model Selection**
   - Cost-based routing
   - Performance optimization
   - Fallback strategies

3. **Metrics Tracking**
   - Cost per token
   - Total usage
   - Historical data

### Intelligent Switching

The `intelligent_switching.py` example demonstrates dynamic model selection:

```python
from multimind.models.advanced import IntelligentSwitchingWrapper

async def intelligent_switching_example():
    wrapper = IntelligentSwitchingWrapper(
        model_factory=factory,
        performance_threshold=0.8
    )
    
    # Generate with automatic switching
    response = await wrapper.generate(
        prompt="Complex query",
        auto_switch=True
    )
    
    # Get performance metrics
    metrics = wrapper.get_performance_metrics()
```

### Features

1. **Performance Monitoring**
   - Latency tracking
   - Success rate monitoring
   - Quality metrics

2. **Dynamic Switching**
   - Threshold-based switching
   - Load balancing
   - Failover handling

3. **Metrics Collection**
   - Performance history
   - Model comparison
   - Usage patterns

## Testing

### Running Tests

```bash
# Run all model management tests
pytest tests/examples/model_management/

# Run specific test files
pytest tests/examples/model_management/test_basic_usage.py
pytest tests/examples/model_management/test_cost_optimization.py
```

### Test Coverage

1. **Basic Usage Tests**
   - Model initialization
   - Text generation
   - Chat completion
   - Error handling

2. **Cost Optimization Tests**
   - Budget management
   - Cost tracking
   - Model selection
   - Metrics collection

3. **Intelligent Switching Tests**
   - Performance monitoring
   - Dynamic switching
   - Metrics collection
   - Error handling

## Best Practices

1. **Model Configuration**
   - Use environment variables
   - Implement proper error handling
   - Set appropriate timeouts

2. **Cost Management**
   - Set budget limits
   - Monitor usage
   - Implement fallbacks

3. **Performance Optimization**
   - Use appropriate models
   - Implement caching
   - Monitor metrics

4. **Error Handling**
   - Implement retries
   - Use fallback models
   - Log errors properly

## Common Issues and Solutions

1. **API Rate Limits**
   ```python
   wrapper = MultiModelWrapper(
       model_factory=factory,
       rate_limit=100,  # requests per minute
       retry_count=3
   )
   ```

2. **Timeout Handling**
   ```python
   wrapper = MultiModelWrapper(
       model_factory=factory,
       timeout=30,  # seconds
       timeout_retry=True
   )
   ```

3. **Error Recovery**
   ```python
   try:
       response = await wrapper.generate(prompt)
   except ModelError:
       # Fallback to simpler model
       response = await wrapper.generate(
           prompt,
           model="gpt-3.5-turbo"
       )
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your example
4. Write tests
5. Update documentation
6. Submit a pull request

## Support

For issues and questions:
1. Open an issue on GitHub
2. Contact contact@multimind.dev