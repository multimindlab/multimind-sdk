<p align="center">
  <img src="https://raw.githubusercontent.com/multimindlab/multimind-sdk/develop/assets/Logo-with-name-final2.png" alt="MultiMind SDK" />
</p>

<p align="center">
  <strong>The compliance-first AI agent framework.</strong><br>
  Multi-model AI with built-in GDPR, HIPAA & NIS2 support. Works with any model. Runs anywhere.
</p>

<p align="center">
  <a href="https://pypi.org/project/multimind-sdk/"><img src="https://img.shields.io/pypi/v/multimind-sdk.svg" alt="PyPI"></a>
  <a href="https://github.com/multimindlab/multimind-sdk/actions/workflows/ci.yml"><img src="https://github.com/multimindlab/multimind-sdk/actions/workflows/ci.yml/badge.svg?branch=develop" alt="CI"></a>
  <a href="https://pypi.org/project/multimind-sdk/"><img src="https://img.shields.io/pypi/pyversions/multimind-sdk.svg" alt="Python versions"></a>
  <a href="https://github.com/multimindlab/multimind-sdk/blob/develop/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://discord.gg/K64U65je7h"><img src="https://img.shields.io/badge/chat-Discord-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>

---

> **New to AI?** Start with the [plain-language getting started guide](docs/getting-started-non-technical.md) — no coding required.

## Why MultiMind?

Most AI frameworks assume you'll handle compliance yourself. MultiMind doesn't.

- **One API for all models** — OpenAI, Anthropic Claude, Google Gemini, Mistral, Groq, DeepSeek, xAI, Together, Perplexity, Fireworks, Cerebras, 300+ models via OpenRouter, and local models via Ollama through a single async interface with streaming
- **Built-in compliance** — a drop-in PII guard that detects, redacts, and audits sensitive data on every model call, plus GDPR & HIPAA policy modeling and compliance dashboards as a first-class module, not an afterthought
- **Govern the stack you already have** — don't migrate off LangChain, LlamaIndex, or CrewAI; wrap them. Run `multimind serve` and point any OpenAI-compatible app at it to add PII redaction, budgets, and audit with one line changed
- **Governed by default** — cost budgets that stop overspending before the call, hallucination detection with sentence-level evidence, and audit trails — each a one-line wrapper around any model
- **Switch models without losing knowledge** — move a live conversation from GPT to Claude to a local model; the context travels with you
- **Adaptive routing** — route requests across providers by cost, latency, or fallback strategy
- **RAG that works** — FAISS and Chroma with document processing out of the box; 40+ client-backed vector stores including Pinecone, Qdrant, Weaviate, Milvus, pgvector, and LanceDB
- **Beyond transformers** — run Mamba and RWKV models through the same interface
- **Runs anywhere** — Cloud, on-prem, air-gapped with local models

## Quick Start

```bash
pip install multimind-sdk
```

```python
import asyncio
from multimind import OpenAIModel

async def main():
    model = OpenAIModel(model_name="gpt-4o-mini")
    response = await model.generate("Explain quantum computing simply")
    print(response)

asyncio.run(main())
```

> Requires `OPENAI_API_KEY` in your environment. The same pattern works for `ClaudeModel` (Anthropic, `ANTHROPIC_API_KEY`), `GeminiModel` (Google, `GEMINI_API_KEY`), `MistralAIModel` (`MISTRAL_API_KEY`), `GroqModel` (`GROQ_API_KEY`), `DeepSeekModel` (`DEEPSEEK_API_KEY`), and `OllamaModel` (local models — Mistral, Llama, and anything else Ollama serves).
> Also: `OpenRouterModel` (`OPENROUTER_API_KEY` — 300+ models via one key), `TogetherModel` (`TOGETHER_API_KEY`), `XAIModel` (`XAI_API_KEY`), `PerplexityModel` (`PERPLEXITY_API_KEY`), `FireworksModel` (`FIREWORKS_API_KEY`), and `CerebrasModel` (`CEREBRAS_API_KEY`).

### Install what you need

```bash
pip install multimind-sdk                # Core (incl. Ollama via HTTP, no extra needed)
pip install multimind-sdk[rag]           # + RAG & vector stores (FAISS, Chroma)
pip install multimind-sdk[agents]        # + Agent framework with memory
pip install multimind-sdk[compliance]    # + GDPR/HIPAA/NIS2 compliance + dashboards
pip install multimind-sdk[finetune]      # + LoRA/QLoRA fine-tuning (CPU)
pip install multimind-sdk[finetune-gpu]  # + 8-bit quantization (Linux/CUDA only)
pip install multimind-sdk[gateway]       # + FastAPI gateway server
pip install multimind-sdk[all]           # Everything
```

> **Ollama users**: no extra needed — `multimind.models.ollama` talks to a running Ollama instance over HTTP. Just `pip install multimind-sdk` and point at `http://localhost:11434`.

## Features

| Feature                                              | Status   | Install Extra      |
| ---------------------------------------------------- | -------- | ------------------ |
| Multi-model chat (OpenAI, Claude, Gemini, Groq, ...) | Stable   | core               |
| Streaming responses                                  | Stable   | core               |
| Runtime PII guard (detect, redact, block, audit)     | Stable   | core               |
| Cost tracking & budgets                              | Stable   | core               |
| Hallucination detection (grounding checks)           | Stable   | core               |
| Mid-conversation model switching (`ModelSession`)    | Stable   | core               |
| Compliance proxy (`multimind serve`, OpenAI-compat)  | Stable   | `[gateway]`        |
| Framework adapters (LangChain, LlamaIndex, CrewAI)   | Stable   | `[langchain]` etc. |
| AI usage audit & cost chargeback (`multimind audit`) | Stable   | core               |
| Anthropic MCP compliance server                      | Stable   | `[mcp]`            |
| RAG pipeline (FAISS, Chroma)                         | Stable   | `[rag]`            |
| Context transfer between models                      | Stable   | core               |
| CLI interface                                        | Stable   | core               |
| REST gateway with Swagger UI (`/docs`)               | Stable   | `[gateway]`        |
| Docker deployment (lean ~300 MB image)               | Stable   | —                  |
| AI Agents with tools & memory                        | Beta     | `[agents]`         |
| Self-orchestrating agents (bounded spawning)         | Beta     | `[agents]`         |
| GDPR & HIPAA compliance (policy modeling, audits)    | Beta     | `[compliance]`     |
| Non-transformer models (Mamba, RWKV)                 | Beta     | `[finetune]`       |
| Vector stores (40+ incl. Qdrant, Weaviate, Pinecone) | Beta     | `[vector-stores]`  |
| Fine-tuning (LoRA)                                   | Beta     | `[finetune]`       |

> Note: `multimind.mcp` is MultiMind's internal **Model Composition Protocol** — a workflow executor for chaining models. Anthropic's Model Context Protocol is supported separately via `multimind.mcp_server` (see [docs/mcp-server.md](docs/mcp-server.md)).

Full status: [FEATURES.md](FEATURES.md) · Roadmap: [ROADMAP.md](ROADMAP.md)

## Examples

### Govern an app you already have — change one line

Start the compliance proxy, then point any OpenAI-compatible client (LangChain, LlamaIndex, the raw `openai` SDK, anything) at it:

```bash
multimind serve --port 8400 --upstream openai --block-on ssn,credit_card --budget 25.00 --audit-log audit.jsonl
```

```python
from openai import OpenAI

# The ONLY change: base_url. Every call now gets PII redaction, budgets, and an audit trail.
client = OpenAI(base_url="http://localhost:8400/v1", api_key="unused-upstream-key-is-server-side")
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "..."}])
```

Prefer to stay in code? Wrap your existing framework objects instead — see [docs/integrations.md](docs/integrations.md) for LangChain, LlamaIndex, CrewAI, and OpenAI-SDK adapters.

### Guard any model against PII leaks

```python
from multimind import OpenAIModel
from multimind.compliance import guard

model = guard(
    OpenAIModel(model_name="gpt-4o-mini"),
    strategy="mask",                    # emails become [EMAIL], SSNs [SSN], ...
    block_on=("credit_card", "ssn"),    # refuse these outright
    audit_log="compliance_audit.jsonl", # types & counts only — never raw PII
)

response = await model.generate("Email jane.doe@corp.com about the invoice")
# The provider only ever saw: "Email [EMAIL] about the invoice"
```

Detects emails, phone numbers, SSNs, credit cards (Luhn-validated), IPs, IBANs,
API keys (entropy-checked), and more — including across streaming chunk boundaries.
Works with any model object that has `generate`/`chat`, so you can wrap
non-MultiMind clients too. No extra dependencies. Runnable demo:
[`examples/compliance/guarded_model.py`](examples/compliance/guarded_model.py)

### Multi-model chat

```python
from multimind import OpenAIModel, ClaudeModel, GeminiModel, GroqModel

gpt = OpenAIModel(model_name="gpt-4o-mini")
claude = ClaudeModel(model_name="claude-3-5-sonnet-20241022")
gemini = GeminiModel(model_name="gemini-2.0-flash")
groq = GroqModel(model_name="llama-3.3-70b-versatile")

# Same interface, different providers
response = await gpt.generate("Hello!")
response = await claude.generate("Hello!")
response = await gemini.generate("Hello!")
```

`MistralAIModel` and `DeepSeekModel` work the same way.

### Sync usage (no asyncio needed)

```python
from multimind import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")
response = model.generate_sync("Explain quantum computing simply")
```

`chat_sync` and `embeddings_sync` are also available. Inside a running event loop, use the async methods instead.

### Structured output

```python
from pydantic import BaseModel
from multimind import OpenAIModel

class City(BaseModel):
    name: str
    population: int

model = OpenAIModel(model_name="gpt-4o-mini")
city = await model.generate("Largest city in France?", response_format=City)
print(city.name, city.population)  # a City instance, not a string
```

Works with `OpenAIModel`, `ClaudeModel`, `GeminiModel`, `MistralAIModel`, `GroqModel`, and `DeepSeekModel`.

### RAG over your documents

```python
from multimind.rag.fluent import RAGPipeline, RAGConfig
from multimind.vector_store.base import VectorStoreConfig, VectorStoreFactory
from multimind.core.router import Router

router = Router()  # register your providers with router.register_provider(...)

vector_store = VectorStoreFactory.create_store(
    "faiss",
    VectorStoreConfig.create_faiss_config(dimension=1536, metric="cosine"),
)

pipeline = RAGPipeline(router, RAGConfig(
    vector_store=vector_store,
    embedding_provider="openai",
    embedding_model="text-embedding-ada-002",
    generation_provider="openai",
    generation_model="gpt-4o-mini",
))

result = await (
    pipeline
    .load_documents(["Your documents here"])
    .query("What does this say?")
    .generate()
    .execute()
)
print(result.answer)
```

Full working example: [`examples/rag/fluent_rag_example.py`](examples/rag/fluent_rag_example.py)

### Agent with tools

```python
from multimind import OpenAIModel
from multimind.agents import Agent
from multimind.agents.tools import CalculatorTool

agent = Agent(
    model=OpenAIModel(model_name="gpt-4o-mini"),
    tools=[CalculatorTool()],
)

# The task should mention the tool by name to route to it.
# Required parameters for the tool are passed as kwargs to agent.run().
response = await agent.run("Use the calculator", expression="42 * 17")
print(response)
```

More examples: [`examples/`](examples/)

## Documentation

- [Docs index](docs/README.md) — everything in one place
- [Quickstart](docs/quickstart.md) — guarded, cost-tracked model calls in 5 minutes
- [Cookbook](docs/cookbook.md) — task-oriented recipes (providers, structured output, budgets, hallucination checks, gateway, docker)
- [Getting started without writing code](docs/getting-started-non-technical.md) — for analysts, compliance officers, and PMs
- [Deployment Guide](docs/deployment.md) — Docker and docker compose, including a fully local stack
- [Compliance Guide](docs/compliance.md)
- [API Reference](docs/api_reference/)
- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

```bash
git clone https://github.com/multimindlab/multimind-sdk.git
cd multimind-sdk
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

<p align="center">
  <a href="https://discord.gg/K64U65je7h">Discord</a> ·
  <a href="https://x.com/multimindsdk">Twitter</a> ·
  <a href="https://opencollective.com/multimind-sdk">Support</a>
</p>
