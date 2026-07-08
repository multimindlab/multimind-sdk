# Quickstart: MultiMind in 5 Minutes

Install, make your first model call, guard it against PII leaks, track what it costs, and scan text for sensitive data from the command line. Every snippet below is copy-paste runnable.

Not a developer? Read the [plain-language getting started guide](getting-started-non-technical.md) instead.

## 1. Install (30 seconds)

```bash
pip install multimind-sdk
```

That is the whole core install: multi-provider models, the PII guard, cost tracking, hallucination checks, and the CLI. Extras like `[rag]` or `[gateway]` are only needed for those features — see the [installation guide](../INSTALLATION.md).

## 2. Pick a model path

**Path A — cloud provider (needs one API key).** Set the key for whichever provider you have:

```bash
export OPENAI_API_KEY=sk-...      # or GROQ_API_KEY, ANTHROPIC_API_KEY,
                                  # GEMINI_API_KEY, MISTRAL_API_KEY, DEEPSEEK_API_KEY
```

**Path B — fully local, no API key.** Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull mistral
```

MultiMind talks to Ollama over HTTP; no extra Python packages needed.

To see which providers are configured at any time (works offline):

```bash
multimind models list
```

## 3. First generation

Path A (cloud):

```python
import asyncio
from multimind import OpenAIModel

async def main():
    model = OpenAIModel(model_name="gpt-4o-mini")
    print(await model.generate("Explain quantum computing in one paragraph"))

asyncio.run(main())
```

Path B (local, no key):

```python
import asyncio
from multimind import OllamaModel

async def main():
    model = OllamaModel(model_name="mistral")
    print(await model.generate("Explain quantum computing in one paragraph"))

asyncio.run(main())
```

The interface is identical for every provider — `ClaudeModel`, `GeminiModel`, `GroqModel`, `MistralAIModel`, `DeepSeekModel` all work the same way. Prefer plain functions over `asyncio`? Use `model.generate_sync(...)` — see the [cookbook](cookbook.md#sync-usage-no-asyncio).

## 4. Guard it against PII leaks

Wrap any model so emails, SSNs, credit cards, API keys, and other PII are redacted before they ever reach the provider — with an audit trail:

```python
import asyncio
from multimind import OpenAIModel
from multimind.compliance import guard

async def main():
    model = guard(
        OpenAIModel(model_name="gpt-4o-mini"),   # or OllamaModel(model_name="mistral")
        strategy="mask",                          # emails become [EMAIL], SSNs [SSN], ...
        block_on=("credit_card", "ssn"),          # refuse these outright
        audit_log="compliance_audit.jsonl",       # types & counts only — never raw PII
    )
    print(await model.generate("Email jane.doe@corp.com about the invoice"))
    # The provider only ever saw: "Email [EMAIL] about the invoice"

asyncio.run(main())
```

Then look at `compliance_audit.jsonl` — one JSON line per event, recording what was detected and redacted (types, counts, and hashes only; never the raw values).

## 5. Track what it costs

```python
import asyncio
from multimind import OpenAIModel
from multimind.observability import CostTracker, track_costs

async def main():
    tracker = CostTracker()   # add jsonl_path="costs.jsonl" to persist records
    model = track_costs(OpenAIModel(model_name="gpt-4o-mini"), tracker=tracker)

    await model.generate("Say hello in five languages")
    print(tracker.report())   # tokens and dollars, per model and total

asyncio.run(main())
```

Want a hard spending ceiling instead of just a report? Add `budget=Budget(max_cost=1.00)` — calls past the ceiling are blocked before dispatch. Recipe: [budget-capped calls](cookbook.md#budget-capped-calls).

## 6. Scan for PII from the command line

No Python needed — one command, works completely offline:

```bash
multimind compliance scan-text "My email is jane.doe@corp.com and my card is 4111 1111 1111 1111"
```

Output is a table of findings (type, position, matched text) and the command exits with code 1 if any PII was found, so it drops straight into CI pipelines. Scan a file instead, and print a redacted version:

```bash
multimind compliance scan-text --file customer_notes.txt --redact mask
```

## Where to next

- [Cookbook](cookbook.md) — short task-oriented recipes: switch providers, structured output with Pydantic, hallucination detection, the gateway API, and more
- [Compliance guide](compliance.md) — the full GDPR/HIPAA compliance module
- [Deployment guide](deployment.md) — Docker and docker compose, including a fully local stack
- [Docs index](README.md) — everything else
