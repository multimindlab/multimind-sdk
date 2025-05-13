# Installation Guide

This guide will help you install and set up the MultiMind SDK in your environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Basic Installation

Install the SDK using pip:

```bash
pip install multimind-sdk
```

## Development Installation

For development or to use the latest features:

```bash
# Clone the repository
git clone https://github.com/multimind-dev/multimind-sdk.git
cd multimind-sdk

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Environment Setup

1. Create a `.env` file in your project root:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Mistral API Key
MISTRAL_API_KEY=your_mistral_api_key

# Optional: Set default model
DEFAULT_MODEL=gpt-3.5-turbo
```

2. Load environment variables in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Model-Specific Setup

### OpenAI
- Get your API key from [OpenAI Platform](https://platform.openai.com)
- Set `OPENAI_API_KEY` in your environment

### Anthropic
- Get your API key from [Anthropic Console](https://console.anthropic.com)
- Set `ANTHROPIC_API_KEY` in your environment

### Mistral
- Get your API key from [Mistral AI](https://console.mistral.ai)
- Set `MISTRAL_API_KEY` in your environment

## Verification

Test your installation:

```python
from multimind import OpenAIModel

# Create a model instance
model = OpenAIModel(model="gpt-3.5-turbo")

# Test the connection
response = await model.generate("Hello, world!")
print(response)
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Verify your `.env` file exists
   - Check that environment variables are loaded
   - Confirm API keys are valid

2. **Version Conflicts**
   - Use a virtual environment
   - Check `pip list` for conflicting packages
   - Update dependencies: `pip install --upgrade -r requirements.txt`

3. **Model Access**
   - Verify your API key has access to the requested model
   - Check model availability in your region
   - Confirm your account has sufficient credits

### Getting Help

- Check the [FAQ](../docs/faq.md)
- Open an issue on [GitHub](https://github.com/multimind-dev/multimind-sdk/issues)
- Contact support at [support@multimind.dev](mailto:support@multimind.dev)

## Next Steps

- Read the [Quickstart Guide](quickstart.md)
- Explore [Configuration Options](configuration.md)
- Check out the [API Reference](api_reference/README.md) 