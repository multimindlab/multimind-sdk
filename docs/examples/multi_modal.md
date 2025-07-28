# Multi-Modal Examples

This document provides detailed information about the multi-modal examples in the MultiMind SDK.

## Overview

The multi-modal examples demonstrate how to:
- Process multiple modalities (text, image, audio)
- Implement cross-modal retrieval
- Create multi-modal workflows
- Handle model routing and orchestration

## Basic Examples

### Process Request

The `process_request.py` example shows basic multi-modal processing:

```python
from multimind.router.multi_modal_router import MultiModalRouter
from multimind.types import UnifiedRequest, ModalityInput

async def process_request_example():
    # Initialize router
    router = MultiModalRouter()
    
    # Process image
    image_request = UnifiedRequest(
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
    image_response = await router.process(image_request)
    
    # Process audio
    audio_request = UnifiedRequest(
        inputs=[
            ModalityInput(
                modality="audio",
                content=audio_data
            )
        ]
    )
    audio_response = await router.process(audio_request)
```

### Key Features

1. **Request Processing**
   - Multiple modality support
   - Unified request format
   - Response handling

2. **Model Routing**
   - Automatic model selection
   - Fallback handling
   - Error recovery

3. **Response Formatting**
   - Structured output
   - Error handling
   - Result validation

## Advanced Examples

### Cross-Modal Retrieval

The `cross_modal_retrieval.py` example demonstrates cross-modal search and analysis:

```python
from multimind.workflows import CrossModalRetrievalWorkflow

async def cross_modal_example():
    # Initialize workflow
    workflow = CrossModalRetrievalWorkflow(
        models=router.models,
        integrations={}
    )
    
    # Execute workflow
    result = await workflow.execute(request)
    
    # Process results
    embeddings = result.get("embeddings", {})
    similarities = result.get("similarities", {})
    analysis = result.get("analysis", {})
```

### Features

1. **Embedding Generation**
   - Multi-modal embeddings
   - Vector storage
   - Similarity search

2. **Cross-Modal Analysis**
   - Content understanding
   - Relationship detection
   - Semantic search

3. **Result Processing**
   - Similarity scoring
   - Content analysis
   - Result ranking

### Multi-Modal Workflows

The `workflows.py` example shows how to create custom workflows:

```python
from multimind.workflows import MultiModalWorkflow

class CustomWorkflow(MultiModalWorkflow):
    async def execute(self, request: UnifiedRequest) -> Dict:
        # Process image
        image_result = await self.process_image(request)
        
        # Process text
        text_result = await self.process_text(request)
        
        # Combine results
        return self.combine_results(image_result, text_result)
```

### Features

1. **Workflow Creation**
   - Custom step definition
   - Error handling
   - Result combination

2. **Model Integration**
   - Multiple model support
   - Model selection
   - Result aggregation

3. **Pipeline Management**
   - Step sequencing
   - Error recovery
   - Result validation

## Testing

### Running Tests

```bash
# Run all multi-modal tests
pytest tests/examples/multi_modal/

# Run specific test files
pytest tests/examples/multi_modal/test_multi_modal.py
pytest tests/examples/multi_modal/test_cross_modal_retrieval.py
```

### Test Coverage

1. **Basic Processing Tests**
   - Request handling
   - Response formatting
   - Error handling

2. **Cross-Modal Tests**
   - Embedding generation
   - Similarity search
   - Analysis validation

3. **Workflow Tests**
   - Step execution
   - Error recovery
   - Result validation

## Best Practices

1. **Request Handling**
   - Validate input data
   - Handle missing modalities
   - Implement proper error handling

2. **Model Management**
   - Use appropriate models
   - Implement fallbacks
   - Monitor performance

3. **Result Processing**
   - Validate outputs
   - Handle errors
   - Format results

4. **Resource Management**
   - Manage memory usage
   - Handle large files
   - Implement timeouts

## Common Issues and Solutions

1. **Large File Handling**
   ```python
   async def process_large_file(file_path: str):
       # Stream large files
       async with aiofiles.open(file_path, 'rb') as f:
           content = await f.read()
           # Process in chunks
           for chunk in chunk_content(content):
               await process_chunk(chunk)
   ```

2. **Memory Management**
   ```python
   class MemoryEfficientWorkflow(MultiModalWorkflow):
       async def execute(self, request: UnifiedRequest):
           # Process in batches
           results = []
           for batch in self.batch_requests(request):
               batch_result = await self.process_batch(batch)
               results.extend(batch_result)
           return self.combine_results(results)
   ```

3. **Error Recovery**
   ```python
   async def process_with_retry(request: UnifiedRequest):
       for attempt in range(3):
           try:
               return await router.process(request)
           except ModelError:
               if attempt == 2:
                   raise
               await asyncio.sleep(1)
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