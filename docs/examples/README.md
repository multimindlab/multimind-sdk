# MultiMind SDK Examples Documentation

This documentation provides detailed information about the example implementations in the MultiMind SDK, including usage patterns, best practices, and advanced features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Model Management Examples](#model-management-examples)
3. [Multi-Modal Examples](#multi-modal-examples)
4. [Advanced Features](#advanced-features)
5. [Testing](#testing)
6. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

1. Install the SDK:
```bash
pip install multimind-sdk
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export HUGGINGFACE_API_KEY="your-key"
```

3. Install example dependencies:
```bash
pip install -r examples/requirements.txt
```

### Basic Usage

1. Model Management:
```python
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

# Initialize
factory = ModelFactory()
wrapper = MultiModelWrapper(
    model_factory=factory,
    primary_model="gpt-3.5-turbo",
    fallback_models=["gpt-4", "claude"]
)

async def example_generate_text():
    # Generate text
    response = await wrapper.generate("Explain quantum computing")
```

2. Multi-Modal Processing:
```python
async def example_multi_modal():
    from multimind.router.multi_modal_router import MultiModalRouter

    # Initialize
    router = MultiModalRouter()

    # Process multi-modal request
    request = {
        "content": {
            "image": "base64_encoded_image",
            "text": "Describe this image"
        },
        "modalities": ["image", "text"]
    }
    response = await router.process(request)
```

## Model Management Examples

### Basic Usage

1. Text Generation:
```python
# examples/model_management/basic/basic_usage.py
async def generate_text():
    wrapper = MultiModelWrapper(
        model_factory=factory,
        primary_model="gpt-3.5-turbo"
    )
    response = await wrapper.generate(
        prompt="Write a short story",
        temperature=0.7
    )
```

2. Chat Completion:
```python
# examples/model_management/basic/basic_usage.py
async def chat_completion():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like?"}
    ]
    response = await wrapper.chat(messages=messages)
```

### Advanced Features

1. Cost Optimization:
```python
# examples/model_management/advanced/cost_optimization.py
wrapper = CostOptimizedWrapper(
    model_factory=factory,
    primary_model="gpt-3.5-turbo",
    fallback_models=["gpt-4", "claude"],
    budget=0.1
)
response = await wrapper.generate("Complex query")
```

2. Intelligent Switching:
```python
# examples/model_management/advanced/intelligent_switching.py
wrapper = IntelligentSwitchingWrapper(
    model_factory=factory,
    performance_threshold=0.8
)
response = await wrapper.generate("Query")
```

## Multi-Modal Examples

### Basic Processing

1. Image Captioning:
```python
# examples/multi_modal/basic/process_request.py
async def process_image():
    request = UnifiedRequest(
        inputs=[
            ModalityInput(
                modality="image",
                content=image_data
            ),
            ModalityInput(
                modality="text",
                content="Describe this image"
            )
        ]
    )
    response = await router.process(request)
```

2. Audio Transcription:
```python
# examples/multi_modal/basic/process_request.py
async def process_audio():
    request = UnifiedRequest(
        inputs=[
            ModalityInput(
                modality="audio",
                content=audio_data
            )
        ]
    )
    response = await router.process(request)
```

### Advanced Workflows

1. Cross-Modal Retrieval:
```python
# examples/multi_modal/workflows/cross_modal_retrieval.py
workflow = CrossModalRetrievalWorkflow(
    models=router.models,
    integrations={}
)
result = await workflow.execute(request)
```

2. Multi-Modal Analysis:
```python
# examples/multi_modal/workflows/workflows.py
workflow = MultiModalAnalysisWorkflow(
    models=router.models,
    integrations={}
)
result = await workflow.execute(request)
```

## Advanced Features

### Cost Tracking

```python
from multimind.metrics.cost_tracker import CostTracker

tracker = CostTracker()
tracker.track_cost("gpt-4", 0.05)
total_cost = tracker.get_total_cost()
```

### Performance Metrics

```python
from multimind.metrics.performance import PerformanceTracker

tracker = PerformanceTracker()
tracker.track_latency("gpt-4", 0.5)
metrics = tracker.get_metrics()
```

### Model Management

```python
from multimind.models.factory import ModelFactory

factory = ModelFactory()
factory.register_model(
    name="custom-model",
    type="openai",
    config={"temperature": 0.7}
)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/examples/

# Run specific test categories
pytest tests/examples/model_management/
pytest tests/examples/multi_modal/

# Run with coverage
pytest --cov=examples tests/examples/
```

### Writing Tests

```python
@pytest.mark.asyncio
async def test_example():
    # Setup
    wrapper = MultiModelWrapper(...)
    
    # Execute
    response = await wrapper.generate("test")
    
    # Assert
    assert response is not None
    assert len(response) > 0
```

## Best Practices

1. **Error Handling**
   - Always use try-except blocks
   - Implement proper fallback strategies
   - Log errors appropriately

2. **Resource Management**
   - Set appropriate timeouts
   - Implement rate limiting
   - Monitor API usage

3. **Performance Optimization**
   - Use appropriate model sizes
   - Implement caching where possible
   - Monitor latency and costs

4. **Security**
   - Never hardcode API keys
   - Validate input data
   - Implement proper access controls

5. **Testing**
   - Write comprehensive tests
   - Use mock data for testing
   - Test error conditions

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