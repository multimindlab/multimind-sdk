---
title: "MultiMind SDK"
description: "The compliance-first AI agent framework: multi-model calls, a PII guard, cost tracking, RAG, agents, and governance tooling for Python."
---

MultiMind SDK is a Python framework for building AI applications that have to pass a compliance review. One install gives you 13 first-party model providers behind one interface, a PII guard that scans prompts and responses, cost tracking with budgets, hallucination checks, RAG, function-calling agents, and an OpenAI-compatible proxy that adds all of this to apps you already run.

```bash
pip install multimind-sdk
```

## Pick a path

| Guide | Who it is for |
| ----- | ------------- |
| [Quickstart](/quickstart) | Developers. Install to guarded, cost-tracked model calls in 5 minutes. |
| [Getting started without code](/getting-started-non-technical) | Analysts, compliance officers, PMs. Terminal only. |
| [Cookbook](/cookbook) | Task-oriented recipes: switch providers, structured output, PII guard, budgets, hallucination checks, gateway, Docker. |
| [Installation](/installation) | Every pip extra and what it adds. |

## Guides

- [Compliance](/compliance) and the [compliance quickstart](/compliance_quickstart): GDPR and HIPAA checks, evidence reports.
- [Guard proxy](/guard-proxy): `multimind serve` governs any OpenAI-compatible app by changing one line.
- [Governance dashboard](/dashboard): `multimind dashboard` for audit trail, costs, inventory, and no-code guardrails.
- [Framework integrations](/integrations): add the PII guard, budgets, and audit to LangChain, LlamaIndex, CrewAI, and the OpenAI SDK.
- [MCP server](/mcp-server): expose compliance tools to Claude Desktop, Claude Code, and other MCP clients.
- [AI inventory and chargeback](/ai-inventory): `multimind audit` finds shadow AI usage and attributes cost.
- [Run tracing](/tracing): `trace_model` records runs and spans to SQLite or a self-hosted platform, storing hashes and lengths instead of prompt text.
- [RAG](/rag), [memory](/memory), [configuration](/configuration), [CLI](/cli), and [deployment](/deployment).

## Reference

- [REST API](/api_reference/overview): the gateway, RAG, and platform endpoints with generated OpenAPI specs.
- [Architecture](/architecture): how the pieces fit together.
- [Feature status](/feature-status): what is stable, beta, and planned.
- [Changelog](https://github.com/multimindlab/multimind-sdk/blob/develop/CHANGELOG.md) and [roadmap](https://github.com/multimindlab/multimind-sdk/blob/develop/ROADMAP.md) on GitHub.

## Community

Source and issues are on [GitHub](https://github.com/multimindlab/multimind-sdk). Questions go to the [Discord server](https://discord.gg/K64U65je7h). Runnable scripts live in the [examples directory](https://github.com/multimindlab/multimind-sdk/tree/develop/examples).
