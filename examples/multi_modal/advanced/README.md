# Advanced Multi-Modal Examples

This directory contains advanced examples demonstrating sophisticated multi-modal processing capabilities in the MultiMind SDK.

## Examples

### 1. Cost-Optimized Processing (`cost_optimized_processing.py`)

This example demonstrates how to process multi-modal requests while optimizing costs and maintaining performance.

**Features:**
- Dynamic model selection based on cost and performance
- Budget management and tracking
- Performance monitoring and metrics
- Error handling and recovery

**Usage:**
```python
from multimind.router.multi_modal_router import MultiModalRouter
from multimind.metrics.cost_tracker import CostTracker
from multimind.metrics.performance import PerformanceTracker
from examples.multi_modal.advanced.cost_optimized_processing import (
    CostOptimizedMultiModalProcessor
)

# Initialize components
router = MultiModalRouter()
cost_tracker = CostTracker()
performance_tracker = PerformanceTracker()

# Create processor
processor = CostOptimizedMultiModalProcessor(
    router=router,
    cost_tracker=cost_tracker,
    performance_tracker=performance_tracker,
    budget=0.1
)

# Process request
result = await processor.process_request(request, optimize_cost=True)
```

### 2. Cross-Modal Retrieval (`cross_modal_retrieval.py`)

This example shows how to implement cross-modal search and analysis.

**Features:**
- Multi-modal embedding generation
- Similarity search across modalities
- Content analysis and relationship detection
- Result ranking and filtering

**Usage:**
```python
from multimind.workflows import CrossModalRetrievalWorkflow

# Initialize workflow
workflow = CrossModalRetrievalWorkflow(
    models=router.models,
    integrations={}
)

# Execute workflow
result = await workflow.execute(request)
```

## Running the Examples

1. Install dependencies:
```bash
pip install -r examples/requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export HUGGINGFACE_API_KEY="your-key"
```

3. Run an example:
```bash
python examples/multi_modal/advanced/cost_optimized_processing.py
```

## Testing

Run the tests:
```bash
pytest tests/examples/multi_modal/test_cost_optimized_processing.py
pytest tests/examples/multi_modal/test_cross_modal_retrieval.py
```

## Best Practices

1. **Cost Management**
   - Set appropriate budgets
   - Monitor usage
   - Implement fallbacks

2. **Performance Optimization**
   - Use appropriate models
   - Implement caching
   - Monitor metrics

3. **Error Handling**
   - Implement retries
   - Use fallback models
   - Log errors properly

4. **Resource Management**
   - Handle large files
   - Implement timeouts
   - Monitor memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your example
4. Write tests
5. Update documentation
6. Submit a pull request

## Support

For issues and questions:
1. Check the [documentation](https://multimind-sdk.readthedocs.io/)
2. Open an issue on GitHub
3. Contact support@multimind.ai 