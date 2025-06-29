# MultiMind CLI Interface

The MultiMind CLI provides a comprehensive command-line interface for using the MultiMind SDK. This interface allows you to perform various AI tasks using multiple providers, ensemble methods, and specialized tools.

## Installation

Make sure you have the MultiMind package installed:

```bash
pip install multimind-sdk
```

## Available Commands

### Ensemble System

#### Text Generation

Generate text using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli generate "Your prompt here" [OPTIONS]
```

Options:
- `--providers`, `-p`: List of providers to use (default: openai, anthropic, ollama)
- `--method`, `-m`: Ensemble method to use (default: weighted_voting)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli generate "Explain quantum computing" \
    --providers openai anthropic ollama \
    --method weighted_voting \
    --output result.json
```

#### Code Review

Review code using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli review path/to/your/code.py [OPTIONS]
```

Options:
- `--providers`, `-p`: List of providers to use (default: openai, anthropic, ollama)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli review my_code.py \
    --providers openai anthropic ollama \
    --output review.json
```

#### Image Analysis

Analyze images using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli analyze-image path/to/your/image.jpg [OPTIONS]
```

Options:
- `--providers`, `-p`: List of providers to use (default: openai, anthropic)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli analyze-image photo.jpg \
    --providers openai anthropic \
    --output analysis.json
```

#### Embedding Generation

Generate embeddings using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli embed "Your text here" [OPTIONS]
```

Options:
- `--providers`, `-p`: List of providers to use (default: openai, huggingface)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli embed "This is a sample text" \
    --providers openai huggingface \
    --output embeddings.json
```

### Model Wrapper

Query various LLM models directly:

```bash
python -m examples.cli.multi_model_wrapper_cli --model MODEL_NAME --prompt "Your prompt" [OPTIONS]
```

Options:
- `--model`: Model to use (choices: dynamically loaded from available models)
- `--prompt`: Your input prompt
- `--ollama-model`: Ollama model name (default: mistral)
- `--hf-model-id`: Hugging Face model ID (default: mistralai/Mistral-7B-v0.1)

Example:
```bash
python -m examples.cli.multi_model_wrapper_cli --model openai --prompt "Hello, world!"
```

### Ollama Chat Interface

Interactive chat with Ollama models:

```bash
python -m examples.cli.chat_ollama_cli [OPTIONS]
```

Options:
- `--model`: Ollama model to use (default: mistral)
- `--history`: Path to save chat history (optional)
- `--no-stream`: Disable streaming responses
- `--debug`: Enable debug logging

Special commands in chat:
- `exit`: Exit the chat
- `history`: Show chat history
- `models`: List available models
- `clear`: Clear chat history
- `pull MODEL_NAME`: Pull a new model

Example:
```bash
python -m examples.cli.chat_ollama_cli --model llama2 --history chat.log
```

### Compliance and Governance

Manage compliance and governance features:

```bash
python -m examples.cli.compliance_cli [COMMAND] [OPTIONS]
```

Available commands:

#### DSAR (Data Subject Access Request)
```bash
# Export user data
python -m examples.cli.compliance_cli dsar export --user-id USER_ID --request-id REQUEST_ID [--format json|csv]

# Erase user data
python -m examples.cli.compliance_cli dsar erase --user-id USER_ID --request-id REQUEST_ID [--verify|--no-verify]
```

#### Model Approval
```bash
python -m examples.cli.compliance_cli governance model-approve --model-id MODEL_ID --approver APPROVER_EMAIL [--metadata JSON_STRING]
```

#### Policy Management
```bash
python -m examples.cli.compliance_cli governance policy-publish --policy-file POLICY_FILE --version VERSION [--metadata JSON_STRING]
```

#### Audit Verification
```bash
python -m examples.cli.compliance_cli governance audit-verify --chain-id CHAIN_ID [--start-time ISO_TIME] [--end-time ISO_TIME]
```

### Basic Agent

Run a basic agent with different models:

```bash
python -m examples.cli.basic_agent
```

This will run example tasks using different models (OpenAI, Claude, Mistral) with a calculator tool.

### Task Runner

Run predefined task workflows:

```bash
python -m examples.cli.task_runner
```

This will execute a research workflow with multiple tasks.

### Prompt Chain

Run a code review prompt chain:

```bash
python -m examples.cli.prompt_chain
```

This will execute a chain of prompts for code review.

## Output Format

The CLI commands output JSON data with the following structure:

```json
{
    "result": "Generated text or analysis",
    "confidence": 0.95,
    "explanation": "Explanation of the ensemble decision",
    "provider_votes": {
        "provider1": "result1",
        "provider2": "result2",
        ...
    }
}
```

## Available Ensemble Methods

- `weighted_voting`: Combines results based on provider weights
- `confidence_cascade`: Uses results based on confidence thresholds
- `parallel_voting`: Combines results from all providers in parallel
- `majority_voting`: Uses the most common result among providers
- `rank_based`: Selects results based on provider ranking

## Environment Variables

The CLI uses the following environment variables (should be set in your `.env` file):

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OLLAMA_API_KEY`: Your Ollama API key (if using Ollama)
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key (if using Hugging Face)

## Error Handling

The CLI provides clear error messages for common issues:
- Missing API keys
- Invalid file paths
- Unsupported providers
- Invalid ensemble methods
- Model availability issues
- Compliance and governance errors

## Examples

See the following example files for comprehensive usage:
- `examples/ensemble/usage_examples.py`: Ensemble system examples
- `examples/cli/basic_agent.py`: Basic agent examples
- `examples/cli/task_runner.py`: Task workflow examples
- `examples/cli/prompt_chain.py`: Prompt chain examples 
