# Cookbook

Task-oriented recipes. Each one is self-contained and copy-paste runnable. New here? Do the [quickstart](quickstart.md) first.

Contents:

- [Switch providers with one line](#switch-providers-with-one-line)
- [Sync usage (no asyncio)](#sync-usage-no-asyncio)
- [Structured output with Pydantic](#structured-output-with-pydantic)
- [PII-guard any model](#pii-guard-any-model)
- [Budget-capped calls](#budget-capped-calls)
- [Detect hallucinations in RAG answers](#detect-hallucinations-in-rag-answers)
- [Switch models mid-conversation without losing knowledge](#switch-models-mid-conversation-without-losing-knowledge)
- [Agents that spawn sub-agents (bounded)](#agents-that-spawn-sub-agents-bounded)
- [Run the gateway API](#run-the-gateway-api)
- [Fully local stack with docker compose](#fully-local-stack-with-docker-compose)

## Switch providers with one line

Every provider implements the same interface — swap the constructor, nothing else changes.

```python
import asyncio
from multimind import (
    OpenAIModel, ClaudeModel, GeminiModel,
    GroqModel, MistralAIModel, DeepSeekModel, OllamaModel,
)

model = OpenAIModel(model_name="gpt-4o-mini")               # OPENAI_API_KEY
# model = ClaudeModel(model_name="claude-3-5-sonnet-20241022")  # ANTHROPIC_API_KEY
# model = GeminiModel(model_name="gemini-2.0-flash")            # GEMINI_API_KEY
# model = GroqModel(model_name="llama-3.3-70b-versatile")       # GROQ_API_KEY
# model = MistralAIModel(model_name="mistral-small-latest")     # MISTRAL_API_KEY
# model = DeepSeekModel(model_name="deepseek-chat")             # DEEPSEEK_API_KEY
# model = OllamaModel(model_name="mistral")                     # local, no key

print(asyncio.run(model.generate("Hello!")))
```

Check which keys are configured (offline, no calls made):

```bash
multimind models list
```

## Sync usage (no asyncio)

Every model has `generate_sync`, `chat_sync`, and `embeddings_sync`:

```python
from multimind import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")
print(model.generate_sync("Explain quantum computing simply"))
```

Inside an already-running event loop (Jupyter, FastAPI handlers), use the async methods instead.

## Structured output with Pydantic

Pass a Pydantic class as `response_format` and get an instance back, not a string:

```python
import asyncio
from pydantic import BaseModel
from multimind import OpenAIModel

class City(BaseModel):
    name: str
    population: int

async def main():
    model = OpenAIModel(model_name="gpt-4o-mini")
    city = await model.generate("Largest city in France?", response_format=City)
    print(city.name, city.population)   # a City instance

asyncio.run(main())
```

Works with `OpenAIModel`, `ClaudeModel`, `GeminiModel`, `MistralAIModel`, `GroqModel`, and `DeepSeekModel`.

## PII-guard any model

`guard()` wraps a model so PII is redacted before it reaches the provider, blocked types raise, and every event lands in a JSONL audit trail:

```python
import asyncio
from multimind import OpenAIModel
from multimind.compliance import guard

async def main():
    model = guard(
        OpenAIModel(model_name="gpt-4o-mini"),
        strategy="mask",                    # or "hash", "remove"
        block_on=("credit_card", "ssn"),
        audit_log="audit.jsonl",
    )
    print(await model.generate("Email jane.doe@corp.com about the invoice"))

asyncio.run(main())
```

The guard is duck-typed: anything with an async `generate` (and optionally `chat` / `generate_stream`) can be wrapped — including clients that are not MultiMind models:

```python
import asyncio
from multimind.compliance import guard

class MyExistingClient:
    """Adapter around any SDK you already use (openai, httpx, ...)."""
    async def generate(self, prompt, **kwargs):
        return f"echo: {prompt}"   # replace with your real client call

async def main():
    safe = guard(MyExistingClient(), strategy="mask", audit_log="audit.jsonl")
    print(await safe.generate("Call me at +1 (555) 123-4567"))
    # Your client only received: "Call me at [PHONE]"

asyncio.run(main())
```

Detects emails, phones, SSNs, credit cards (Luhn-validated), IPs, IBANs, passports, dates of birth, and API keys (entropy-checked) — including across streaming chunk boundaries. Stdlib-only, no extra dependencies.

## Budget-capped calls

`Budget` enforces a hard spending ceiling — the call past the limit is blocked before it is dispatched:

```python
import asyncio
from multimind import OpenAIModel
from multimind.observability import Budget, BudgetExceededError, CostTracker, track_costs

async def main():
    tracker = CostTracker(jsonl_path="costs.jsonl")   # persist per-call records
    budget = Budget(max_cost=0.50)                     # dollars, this session

    model = track_costs(
        OpenAIModel(model_name="gpt-4o-mini"),
        tracker=tracker,
        budget=budget,
        tag="support-bot",     # group spend by feature or team
    )

    try:
        await model.generate("Summarize the quarterly report")
    except BudgetExceededError as exc:
        print(f"Blocked: {exc}")

    print(tracker.report())        # totals per model
    print(tracker.by_tag())        # totals per tag
    print(f"Remaining: ${budget.remaining:.4f}")

asyncio.run(main())
```

A warning is logged at 80% of the ceiling (tune with `warn_ratio`). Offline demo that runs with no API key: [examples/observability/cost_tracking.py](../examples/observability/cost_tracking.py).

## Detect hallucinations in RAG answers

Wrap the generating model so every answer is grounding-checked against the retrieved sources. The default check is lexical (deterministic, offline, no judge LLM needed):

```python
import asyncio
from multimind import OpenAIModel
from multimind.evaluation.hallucination import detect_hallucinations

SOURCES = [
    "The Eiffel Tower is located in Paris. It was completed in 1889. "
    "The tower is 330 metres tall.",
]

async def main():
    model = detect_hallucinations(
        OpenAIModel(model_name="gpt-4o-mini"),
        sources=SOURCES,        # or a callable: lambda prompt: retriever.search(prompt)
        threshold=0.5,          # minimum grounding score
        on_flag="annotate",     # or "raise" (HallucinationError) or "retry"
    )
    answer = await model.generate("How tall is the Eiffel Tower?")
    print(answer)
    print(model.last_report.summary())   # per-sentence grounding verdicts

asyncio.run(main())
```

With `sources=callable`, sources are fetched per prompt — plug in your retriever for RAG. `on_flag="retry"` regenerates once with a strict grounding instruction. For a score without wrapping anything:

```python
from multimind.evaluation.hallucination import HallucinationDetector

report = HallucinationDetector().check_grounding(
    "The tower is 330 metres tall. It was designed by Leonardo da Vinci.",
    ["The Eiffel Tower is 330 metres tall. Gustave Eiffel's company built it."],
)
print(report.summary())   # 1 supported, 1 unsupported
```

Offline end-to-end demo: [examples/evaluation/hallucination_detection.py](../examples/evaluation/hallucination_detection.py).

## Switch models mid-conversation without losing knowledge

Start a conversation on one provider, move to another — the context travels with you.

```python
import asyncio
from multimind import OpenAIModel, ClaudeModel
from multimind.client import ModelSession

async def main():
    session = ModelSession(OpenAIModel(model_name="gpt-4o-mini"))
    await session.chat("My project is called Zephyr, written in Rust. Deadline is March 15.")

    # Switch to Claude; "summary" compacts the history into a context message
    await session.switch(ClaudeModel(model_name="claude-3-5-sonnet-20241022"), strategy="summary")

    answer = await session.chat("What language is my project written in?")
    # Claude answers "Rust" — knowledge survived the switch
    print(session.transfers)  # audit log: who switched to whom, when, how

asyncio.run(main())
```

Strategies: `"full"` (replay raw history, zero loss), `"summary"` (compact — best for
long chats crossing context limits), `"adapted"` (provider-specific formatting).
With `max_history_tokens=...` long conversations are compacted automatically.
Composes with `guard(...)` and `track_costs(...)` wrappers.
Runnable demo: [examples/context_transfer/model_switching.py](../examples/context_transfer/model_switching.py)

## Agents that spawn sub-agents (bounded)

The coordinator decides at runtime whether to answer, delegate, or create a new
sub-agent for a subtask — inside hard safety bounds.

```python
import asyncio
from multimind import OpenAIModel
from multimind.agents import AgentOrchestrator

async def main():
    orchestrator = AgentOrchestrator(
        OpenAIModel(model_name="gpt-4o-mini"),
        max_agents=6,   # total sub-agents per run
        max_depth=2,    # spawn depth limit
    )

    result = await orchestrator.run("Research solar panel efficiency and summarize it in one paragraph.")
    print(result.answer)
    print(result.agent_tree.to_dict())  # who spawned whom, each task and result
    print(result.bounds_hit)            # never silently truncated

asyncio.run(main())
```

Pass `budget_tracker=` (a `multimind.observability` Budget-backed tracker) to cap
spend across the whole agent tree, and `audit_hook=` to log every spawn/delegate
event. Runnable demo: [examples/agents/self_orchestrating_agent.py](../examples/agents/self_orchestrating_agent.py)

## Run the gateway API

Expose chat, generation, and model comparison over HTTP:

```bash
pip install "multimind-sdk[gateway]"
python -m multimind.gateway.api
```

Then:

- Swagger UI (interactive docs): http://localhost:8000/docs
- Health check: `curl http://localhost:8000/health`
- Chat:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"model": "openai", "messages": [{"role": "user", "content": "Hello"}]}'
```

Provider keys are read from the environment at request time; a request naming an unconfigured model returns 400 with a clear error (503 if no provider at all is available).

## Fully local stack with docker compose

Gateway plus Ollama, no API keys, nothing leaves your machine:

```bash
git clone https://github.com/multimindlab/multimind-sdk.git
cd multimind-sdk
DEFAULT_MODEL=ollama docker compose --profile local up -d
docker compose exec ollama ollama pull mistral

curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"model": "ollama", "messages": [{"role": "user", "content": "Hello"}]}'
```

Swagger UI is at http://localhost:8000/docs. Models persist in the `ollama_models` volume. Full container reference (env vars, health checks, production notes): [deployment guide](deployment.md).
