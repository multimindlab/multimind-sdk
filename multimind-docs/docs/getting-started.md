# Getting Started

## Prerequisites
- Python 3.8+
- pip (Python package manager)
- API keys for supported models (e.g., OpenAI, Anthropic, Mistral)

## Installation
Install MultiMind SDK from PyPI:

```
pip install multimind-sdk
```

For development or latest features:
```
git clone https://github.com/multimindlabs/multimind-sdk.git
cd multimind-sdk
pip install -e ".[dev]"
```

## Environment Setup
Set your API keys as environment variables or in a `.env` file:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
```

## Running the SDK
You can use the CLI for training, evaluation, and inference:

```
multimind train --config configs/train_config.yaml
multimind evaluate --model ./output/model --dataset data.json
multimind infer --model ./output/model --input "What is PEFT?"
```

Refer to the CLI help for all available commands:
```
multimind --help
``` 