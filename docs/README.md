# MultiMind Documentation

Start here and pick the path that fits you.

## Start

| Guide | Who it is for |
| ----- | ------------- |
| [Quickstart](quickstart.md) | Developers — install to guarded, cost-tracked model calls in 5 minutes |
| [Getting started without writing code](getting-started-non-technical.md) | Analysts, compliance officers, PMs — plain language, terminal only |
| [Cookbook](cookbook.md) | Task-oriented recipes: switch providers, structured output, PII guard, budgets, hallucination checks, gateway, docker |
| [Installation guide](../INSTALLATION.md) | Which pip extras exist and what each one adds |

## Guides

- [Compliance](compliance.md) — GDPR/HIPAA compliance module ([quick version](compliance_quickstart.md))
- [Guard proxy](guard-proxy.md) — `multimind serve`: govern any OpenAI-compatible app by changing one line
- [Framework integrations](integrations.md) — add PII guard, budgets, and audit to LangChain, LlamaIndex, CrewAI, and the OpenAI SDK
- [MCP server](mcp-server.md) — expose compliance tools to Claude Desktop / Claude Code and other MCP clients
- [AI inventory & chargeback](ai-inventory.md) — `multimind audit`: find shadow AI usage and attribute costs
- [Deployment](deployment.md) — Docker image, docker compose, fully local stack, production notes
- [CLI](cli.md) — the `multimind` command-line interface
- [RAG](rag.md) — retrieval-augmented generation pipeline
- [Memory](memory.md) — agent memory systems
- [Configuration](configuration.md) — settings and environment variables

## Reference

- [API reference](api_reference/README.md) — REST and Python APIs
- [Architecture](architecture.md) — how the pieces fit together
- [Feature status](../FEATURES.md) — what is stable, beta, and planned
- [Roadmap](../ROADMAP.md) and [Changelog](../CHANGELOG.md)

## Contributing

- [Contributing guide](../CONTRIBUTING.md) — dev setup, code style, good first areas
- [Examples](../examples/) — runnable example scripts

Questions? Join the [Discord community](https://discord.gg/K64U65je7h) or open a [GitHub issue](https://github.com/multimindlab/multimind-sdk/issues).
