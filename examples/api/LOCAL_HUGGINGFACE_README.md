# Using HuggingFace Models Locally (No API Key Required)

This guide shows you how to use HuggingFace models locally without requiring any API keys.

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers torch
```

### 2. Use the SDK's HuggingFaceHandler

The `HuggingFaceHandler` automatically uses local models when no API key is provided:

```python
from multimind.gateway.models import HuggingFaceHandler
from multimind.core.config import ModelConfig
import asyncio

async def main():
    # No API key needed - will load locally
    config = ModelConfig(
        api_key=None,  # No API key = use local model
        model_name="gpt2",  # Small model for testing
        temperature=0.7,
        max_tokens=100
    )
    
    handler = HuggingFaceHandler(config)
    response = await handler.generate("Hello, how are you?")
    print(response.content)

asyncio.run(main())
```

### 3. Or Use Transformers Directly

For a simpler approach, use transformers directly:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Configuration

### Environment Variables

You can configure via environment variables:

```bash
# Optional - only needed if you want to use HuggingFace Inference API
export HUGGINGFACE_API_KEY=your_api_key_here

# Model name (defaults to "gpt2" if not set)
export HUGGINGFACE_MODEL_NAME=gpt2
```

**Important**: If `HUGGINGFACE_API_KEY` is not set or empty, the SDK will automatically use local transformers instead of the API.

### Programmatic Configuration

```python
from multimind.core.config import ModelConfig

# Local model (no API key)
config = ModelConfig(
    api_key=None,
    model_name="gpt2",
    temperature=0.7,
    max_tokens=100
)

# Or use HuggingFace API (requires API key)
config = ModelConfig(
    api_key="hf_xxxxxxxxxxxx",
    model_name="mistralai/Mistral-7B-Instruct-v0.2"
)
```

## Recommended Models for Testing

### Small Models (Fast, Low Memory)
- **gpt2** (124M parameters) - Fastest, good for testing
- **distilgpt2** (82M parameters) - Even smaller
- **microsoft/DialoGPT-small** (117M parameters) - Good for conversations

### Medium Models (More Memory Required)
- **microsoft/Phi-3-mini-4k-instruct** (~3.8B parameters)
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (1.1B parameters)

### Large Models (Requires Significant RAM/VRAM)
- **mistralai/Mistral-7B-Instruct-v0.2** (7B parameters)
- **meta-llama/Llama-2-7b-chat-hf** (7B parameters) - Requires HuggingFace access

## Examples

### Example 1: Simple Local Usage
See `simple_local_hf.py` for a basic example.

### Example 2: Using SDK Handler
See `local_huggingface_example.py` for examples using the SDK's HuggingFaceHandler.

## How It Works

1. **No API Key**: When `api_key` is `None` or empty, the handler uses `transformers` library to load models locally.

2. **Model Download**: On first use, the model is downloaded from HuggingFace Hub and cached in `~/.cache/huggingface/`. Subsequent runs use the cached version.

3. **Automatic Device Selection**: Models automatically use GPU if available (`cuda`), otherwise CPU.

4. **API Fallback**: If an API key is provided, it uses HuggingFace Inference API instead.

## Tips

1. **First Run**: The first time you load a model, it will download (~500MB for GPT-2). This only happens once.

2. **Memory**: Smaller models like GPT-2 work well on most machines. Larger models may require significant RAM or VRAM.

3. **Caching**: Models are cached locally, so offline usage works after the first download.

4. **Private Models**: For private/gated models, you can still pass a token:
   ```python
   config = ModelConfig(
       api_key="hf_xxxxxxxxxxxx",  # Token for gated models
       model_name="meta-llama/Llama-2-7b-chat-hf"
   )
   ```

## Troubleshooting

### ImportError: transformers not found
```bash
pip install transformers torch
```

### Out of Memory
- Use smaller models (gpt2, distilgpt2)
- Reduce `max_tokens`
- Use CPU instead of GPU (if GPU memory is limited)

### Model Download Slow
- This is normal on first run
- Models are cached for future use
- Consider using smaller models for testing

## Performance

Local models are slower than API calls but offer:
- ✅ No API costs
- ✅ No rate limits
- ✅ Works offline (after initial download)
- ✅ Full control over model and parameters
- ✅ Privacy (data stays local)


