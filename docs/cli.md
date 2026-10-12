---
title: "MultiMind CLI Interface"
---

The MultiMind CLI provides a comprehensive command-line interface for using the MultiMind SDK. This interface allows you to perform various AI tasks using multiple providers, ensemble methods, and specialized tools.

## Installation

Make sure you have the MultiMind package installed:

```bash
pip install multimind-sdk
```

## The `multimind` Command

Installing the package adds a `multimind` entry point with the following command groups:

```bash
multimind --help
```

Commands: `audit`, `backend`, `chat`, `compliance`, `config`, `dashboard`, `models`, `serve`.

Two extra commands are dispatched before the Click group and therefore do not appear in `multimind --help`: `multimind convert` and `multimind context-transfer`. Both are documented under Model Conversion and Context Transfer below.

### Model Management

```bash
# List registered model providers (or local fine-tuned models with --output-dir)
multimind models list
multimind models list --output-dir ./finetuned

# Compare responses from multiple models
multimind models compare "Explain quantum computing" -m openai -m claude

# Check health of models (all, or one with -m)
multimind models health
multimind models health -m openai

# Show metrics and health status for models
multimind models metrics -m openai

# Download a pretrained or fine-tuned model
multimind models download -m bert-base-uncased

# Export a model to ONNX or TorchScript format
multimind models export -m ./finetuned/my-model -f onnx -o my-model.onnx

# Delete a local fine-tuned model
multimind models delete -m ./finetuned/my-model
```

`models metrics` reads the gateway's `ModelMonitor` (`multimind.core.monitoring`), which is an in-memory, per-process object fed by `/v1/chat` and `/v1/generate` requests. A fresh CLI process has its own empty monitor, so the command prints empty tables unless it runs inside the same process as the server; to see live gateway numbers query `GET /v1/metrics` on the running server instead.

### Chat Sessions

```bash
# Start an interactive chat session with a model (-m is required)
multimind chat start -m openai

# Send a single prompt instead of an interactive session
multimind chat start -m openai -p "Hello, world!"

# Manage sessions
multimind chat list-sessions
multimind chat save SESSION_ID
multimind chat load SESSION_ID
multimind chat delete SESSION_ID
```

### Configuration

```bash
# Show environment and configuration info
multimind config info

# View or set global CLI configuration (stored in ~/.multimind_cli_config)
multimind config manage
multimind config manage --set default_model openai
multimind config manage --get default_model

# Generate shell completion script (bash, zsh, or fish)
multimind config completion zsh
```

### Compliance

#### PII Text Scanning

Scan text or a file for PII using the ComplianceGuard detector. Exits with code 1 if any PII is found, which makes it usable in CI pipelines.

```bash
multimind compliance scan-text "Contact john@example.com or 555-123-4567"
```

Options:

- `--file`, `-f`: Scan a file instead of inline text
- `--redact`, `-r`: Also print the text redacted with this strategy (`mask`, `hash`, or `remove`)

Example:
```bash
multimind compliance scan-text -f support_ticket.txt -r mask
```

#### Monitoring, Reports, and Alerts

```bash
# Run compliance monitoring from a configuration file
multimind compliance run-compliance -c compliance_config.json -o results.json

# Generate a compliance report
multimind compliance generate-report -c compliance_config.json -o report.json

# Show the compliance dashboard for an organization
multimind compliance dashboard -o my-org -t 7d

# Show compliance alerts (filter by status/severity)
multimind compliance alerts -o my-org -s active -v high

# Configure compliance alert rules
multimind compliance configure-alerts -o my-org -c alert_rules.json

# Run a bundled compliance example (healthcare or general)
multimind compliance run-example -t healthcare -u diagnosis

# Evidence report (md or html) from the guard audit log, cost log, and an inventory scan
multimind compliance report-evidence --audit-log audit.jsonl --costs-log costs.jsonl \
    --project . --period 2026-07 --organization "Acme" --format md -o evidence.md
```

`compliance dashboard` and `compliance alerts` read the same in-process store as the `/v1/compliance/dashboard` and `/v1/compliance/alerts` routes: one `ComplianceShard` per organization, held for the life of the backend process and filled by `POST /v1/compliance/monitor` calls (`multimind/gateway/compliance_api.py`). Nothing is persisted to disk, so a CLI invocation in its own process reports zero checks and no alerts; use the routes on the running server (`multimind backend`) to see what that process recorded.

### Compliance Proxy Server

Start the OpenAI-compatible compliance proxy. Point any existing OpenAI client's `base_url` at this proxy to get PII redaction, budget enforcement, and an audit trail with zero code changes.

```bash
multimind serve --upstream openai --port 8400 \
    --strategy mask \
    --block-on ssn,credit_card \
    --budget 10.0 \
    --audit-log audit.jsonl
```

Options:

- `--host`: Bind host (default 127.0.0.1)
- `--port`, `-p`: Bind port (default 8400)
- `--upstream`: Named upstream provider: `openai`, `groq`, `mistral`, `gemini`, `deepseek`, `ollama`
- `--upstream-base-url`: Custom OpenAI-compatible upstream URL (overrides `--upstream`)
- `--strategy`: PII redaction strategy: `mask`, `hash`, or `remove` (default `mask`)
- `--block-on`: Comma-separated PII types that reject the request, e.g. `ssn,credit_card`
- `--budget`: Max session spend in USD
- `--audit-log`: Path for the JSONL audit trail
- `--costs-log`: Path for the JSONL cost log (default `costs.jsonl`)
- `--scan-output` / `--no-scan-output`: Also redact PII in model output (default on)
- `--config`: Guardrails JSON file authored via `multimind dashboard`

The proxy can also be configured through `MULTIMIND_PROXY_*` environment variables; see the [Configuration Guide](/configuration).

### Governance Dashboard

Serve a read-only local web UI over the guard audit trail, cost log, shadow-AI inventory, and a no-code guardrails editor whose output feeds `multimind serve --config`:

```bash
multimind dashboard --port 8501 --audit-log audit.jsonl --costs-log costs.jsonl \
    --project . --guardrails guardrails.json
```

See the [dashboard guide](/dashboard).

### Backend API Server

Serve every supported REST route (gateway, governance, guard proxy, RAG when installed) from one process with one Swagger UI at `/docs`:

```bash
multimind backend --host 127.0.0.1 --port 8080
```

### AI Usage Audit

#### Shadow-AI Inventory Scan

Scan a project directory for AI usage. Exits with code 1 if hardcoded AI API keys are found.

```bash
multimind audit scan ./my-project --include-env --json
```

Options:

- `--include-env`: Also report which known AI env keys are set (names only, never values)
- `--json`: Machine-readable JSON output

#### Cost Chargeback Report

Per-tag chargeback report from the session tracker or a JSONL log.

```bash
multimind audit costs --log costs.jsonl --period 2026-07
```

Options:

- `--log`: Rebuild the tracker from a persisted JSONL cost log (e.g. `costs.jsonl`)
- `--period`: Filter by ISO timestamp prefix, e.g. `2026-07`

### Model Conversion

`multimind convert` wraps `ModelConversionManager` (`multimind/model_conversion`). It is dispatched before the Click group, so it is absent from `multimind --help`, and it imports torch, so it needs the `finetune` extra:

```bash
pip install "multimind-sdk[finetune]"
```

Without torch the command exits with code 1 and a message pointing at that extra.

```bash
# HuggingFace model to GGUF
multimind convert --source huggingface --target gguf --model-path Qwen/Qwen1.5-7B --output-dir ./models

# PyTorch to Safetensors with compression
multimind convert --source pytorch --target safetensors --model-path ./model.pt --output-dir ./out --compression lz4

# ONNX to ONNX Runtime
multimind convert --source onnx --target ort --model-path ./model.onnx --output-dir ./out --optimization-level all
```

Required options:

- `--source`: Source format: `huggingface`, `pytorch`, `tensorflow`, `onnx`, or `ollama`
- `--target`: Target format: `gguf`, `safetensors`, `tflite`, `ort`, or `onnx`
- `--model-path`: Path to the source model or a HuggingFace model ID
- `--output-dir`: Directory for the converted model (created if missing)

Optional:

- `--quantization`: `q4_k_m`, `q4_0`, `q5_k_m`, `q8_0`, `int8`, or `fp16`
- `--compression`: `lz4` or `zstd` (Safetensors); `--compression-level` 1-9 (default 9)
- `--optimizations`: One or more optimization names, e.g. `DEFAULT OPTIMIZE_FOR_LATENCY` (TFLite)
- `--optimization-level`: `basic`, `all`, or `extreme` (ONNX Runtime)
- `--device`: `cpu` (default) or `cuda`
- `--context-length`: Context length for GGUF output
- `--metadata`: One or more `key=value` pairs attached to the output
- `--validate`: Validate the model before and after conversion (a failed validation exits 1)
- `--test`: Run the converted model; implemented for `gguf` only, other targets print "not implemented"
- `--verbose`: Print source and converted model metadata, and a traceback on error

### Context Transfer

`multimind context-transfer` reformats a saved conversation from one provider into a prompt for another (`multimind.context_transfer`). It is dispatched before the Click group, needs no extras, and uses argparse-style underscore options.

```bash
# List supported models with context length and capability flags
multimind context-transfer --list_models

# Capabilities of one model
multimind context-transfer --model_info deepseek

# Basic transfer
multimind context-transfer --from_model chatgpt --to_model deepseek \
    --input_file conversation.json --output_file deepseek_prompt.txt

# Detailed summary, smart extraction, JSON output
multimind context-transfer --from_model claude --to_model gemini \
    --input_file chat.json --output_file gemini_prompt.txt \
    --last_n 10 --summary_type detailed --smart_extraction \
    --output_format json --include_code_context
```

Transfer options (`--from_model`, `--to_model`, `--input_file`, and `--output_file` are all required for a transfer; the command exits 2 if one is missing):

- `--from_model`, `--to_model`: Source and target model names, e.g. `chatgpt`, `claude`, `deepseek`, `gemini`, `mistral`; see `--list_models`
- `--input_file`: Conversation history file (the output directory is created if needed)
- `--output_file`: Where the formatted prompt is written
- `--last_n`: Number of recent turns to extract (default 5)
- `--no_summary`: Skip summarization and use only the last message
- `--summary_type`: `concise` (default), `detailed`, or `structured`
- `--smart_extraction`: Importance-based context extraction
- `--output_format`: `txt` (default), `json`, or `markdown`

Formatting flags, each adding instructions to the generated prompt: `--include_code_context`, `--include_reasoning`, `--include_safety`, `--include_creativity`, `--include_examples`, `--include_step_by_step`, `--include_multimodal`, `--include_web_search`.

Information and general options:

- `--list_models`: List supported models and exit
- `--model_info MODEL`: Show one model's capabilities and exit
- `--verbose`, `-v`: Debug logging and a traceback on unexpected errors

Exit codes: 0 on success, 1 for a missing input file, unsupported model, or any other error.

## Example CLI Scripts

The repository also ships standalone example scripts under `examples/cli/` (run from a checkout of the repository, not the installed package).

### Ensemble System

#### Text Generation

Generate text using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli generate "Your prompt here" [OPTIONS]
```

Options:

- `--providers`, `-p`: Provider to use; repeat the flag for multiple (default: openai, anthropic, ollama)
- `--method`, `-m`: Ensemble method to use (default: weighted_voting)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli generate "Explain quantum computing" \
    -p openai -p anthropic -p ollama \
    --method weighted_voting \
    --output result.json
```

#### Code Review

Review code using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli review path/to/your/code.py [OPTIONS]
```

Options:

- `--providers`, `-p`: Provider to use; repeat the flag for multiple (default: openai, anthropic, ollama)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli review my_code.py \
    -p openai -p anthropic -p ollama \
    --output review.json
```

#### Image Analysis

Analyze images using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli analyze-image path/to/your/image.jpg [OPTIONS]
```

Options:

- `--providers`, `-p`: Provider to use; repeat the flag for multiple (default: openai, anthropic, ollama)
- `--prompt`, `-q`: Instruction sent to vision models (optional)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli analyze-image photo.jpg \
    -p openai -p anthropic \
    --output analysis.json
```

#### Embedding Generation

Generate embeddings using an ensemble of models:

```bash
python -m examples.cli.ensemble_cli embed "Your text here" [OPTIONS]
```

Options:

- `--providers`, `-p`: Provider to use; repeat the flag for multiple (default: openai, ollama, huggingface)
- `--model`, `-m`: Model to use (provider-specific; optional)
- `--output`, `-o`: Output file path (optional)

Example:
```bash
python -m examples.cli.ensemble_cli embed "This is a sample text" \
    -p openai -p huggingface \
    --output embeddings.json
```

### Model Wrapper

Query various LLM models directly:

```bash
python -m examples.cli.multi_model_wrapper_cli --model MODEL_NAME --prompt "Your prompt" [OPTIONS]
```

Options:

- `--list-models`: List all available models and exit
- `--model`: Model to use (choices: dynamically loaded from available models)
- `--prompt`: Your input prompt
- `--temperature`: Sampling temperature (default: 0.7)
- `--max-tokens`: Maximum tokens to generate

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

Available commands include `ingest`, `validate-output`, `monitor-anomalies`, `export-logs`, `dsar`, `model-approve`, `policy-publish`, `audit-verify`, `plugin-register`, `test-run`, `drift-check`, and `risk-override`.

#### DSAR (Data Subject Access Request)
```bash
# Export user data
python -m examples.cli.compliance_cli dsar export --user-id USER_ID --request-id REQUEST_ID [--format json|csv]

# Erase user data
python -m examples.cli.compliance_cli dsar erase --user-id USER_ID --request-id REQUEST_ID [--verify|--no-verify]
```

#### Model Approval
```bash
python -m examples.cli.compliance_cli model-approve --model-id MODEL_ID --approver APPROVER_EMAIL [--metadata JSON_STRING]
```

#### Policy Management
```bash
python -m examples.cli.compliance_cli policy-publish --policy-file POLICY_FILE --version VERSION [--metadata JSON_STRING]
```

#### Audit Verification
```bash
python -m examples.cli.compliance_cli audit-verify --chain-id CHAIN_ID [--start-time ISO_TIME] [--end-time ISO_TIME]
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

The ensemble CLI commands output JSON data with the following structure:

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
- `ANTHROPIC_API_KEY`: Your Anthropic API key (`CLAUDE_API_KEY` is also accepted)
- `MISTRAL_API_KEY`: Your Mistral API key (if using Mistral AI)
- `GROQ_API_KEY`: Your Groq API key (if using Groq)
- `GEMINI_API_KEY`: Your Gemini API key (`GOOGLE_API_KEY` is also accepted)
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `HUGGINGFACE_API_KEY`: Hugging Face token for gated models (optional)
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)

See the [Configuration Guide](/configuration) for the full list, including the `MULTIMIND_PROXY_*` variables read by `multimind serve`.

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
