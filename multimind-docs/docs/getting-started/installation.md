# Installation

## Basic Installation

```bash
pip install multimind-sdk
```

## Development Installation

```bash
git clone https://github.com/multimindlabs/multimind-sdk.git
cd multimind-sdk
pip install -e ".[dev]"
```

## Environment Setup

- Set your API keys in a `.env` file or as environment variables.
- Example `.env`:
  ```env
  OPENAI_API_KEY=sk-...
  ANTHROPIC_API_KEY=...
  MISTRAL_API_KEY=...
  ```

## Troubleshooting
- If you see missing dependency errors, run `pip install -e ".[dev]"` again.
- For CUDA/GPU issues, check your PyTorch install. 