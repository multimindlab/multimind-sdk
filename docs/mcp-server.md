# MCP Server

MultiMind ships an [Anthropic Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that exposes the compliance toolkit — PII scanning, redaction, grounding checks, policy gating, and audit logging — as tools any MCP client can call: Claude Desktop, Claude Code, Cursor, or your own agent.

Not to be confused with `multimind.mcp`, MultiMind's internal Model Composition Protocol for workflow orchestration. The MCP server lives in `multimind.mcp_server`.

## Install and run

```bash
pip install 'multimind-sdk[mcp]'
python -m multimind.mcp_server
```

The server is named `multimind-compliance` and speaks MCP over stdio. The base `multimind.mcp_server` package imports without the `mcp` extra; only running the server requires it.

## Client configuration

Claude Desktop (`claude_desktop_config.json`) or Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "multimind-compliance": {
      "command": "python",
      "args": ["-m", "multimind.mcp_server"]
    }
  }
}
```

Claude Code one-liner:

```bash
claude mcp add multimind-compliance -- python -m multimind.mcp_server
```

Environment:

| Variable | Default | Purpose |
|---|---|---|
| `MULTIMIND_AUDIT_LOG` | `./multimind_audit.jsonl` | Destination JSONL file for the `audit_log` tool |

## Tools

### `scan_text(text)`

Detect PII in text. Returns finding types and character spans plus per-type counts — never the raw matched values.

```json
{"text": "contact jane.doe@example.com"}
```

```json
{"findings": [{"type": "email", "start": 8, "end": 28}], "counts": {"email": 1}, "total": 1}
```

Detected types: `email`, `phone`, `ssn`, `credit_card` (Luhn-validated), `ip_address`, `iban`, `passport`, `dob`, `api_key` (entropy-filtered).

### `redact_text(text, strategy="mask")`

Redact detected PII. Strategies: `mask` (`[EMAIL]`), `hash` (`[EMAIL:1f9a3c2b]`, a stable 8-char tag for correlation without exposure), `remove` (delete outright).

```json
{"text": "mail jane.doe@example.com now", "strategy": "mask"}
```

```json
{"redacted": "mail [EMAIL] now", "strategy": "mask", "counts": {"email": 1}, "total": 1}
```

### `check_grounding(response, sources)`

Score how well each sentence of a response is supported by the given source texts. Deterministic n-gram containment — offline, no LLM call. Catches fabricated sentences; being lexical, it cannot catch paraphrased contradictions.

```json
{"response": "The sky is blue. Elephants can fly.", "sources": ["The sky is blue. Grass is green."]}
```

```json
{
  "score": 0.5,
  "unsupported_count": 1,
  "summary": "grounding score 0.50 over 2 sentence(s): 1 supported, 0 partial, 1 unsupported",
  "sentences": [
    {"sentence": "The sky is blue.", "verdict": "supported", "score": 1.0, "best_source_snippet": "The sky is blue."},
    {"sentence": "Elephants can fly.", "verdict": "unsupported", "score": 0.0, "best_source_snippet": null}
  ]
}
```

### `audit_log(event, metadata?)`

Append a caller-declared compliance event to the JSONL audit trail (`MULTIMIND_AUDIT_LOG`). The event text and any string metadata values are PII-masked before writing, so raw PII never lands in the log; detected types are recorded as counts.

```json
{"event": "user pasted jane.doe@example.com into the chat", "metadata": {"session": "abc123"}}
```

```json
{"ok": true, "path": "multimind_audit.jsonl"}
```

Written line:

```json
{"timestamp": "2026-07-06T12:00:00+00:00", "source": "mcp", "event": "user pasted [EMAIL] into the chat", "pii_types": {"email": 1}, "metadata": {"session": "abc123"}}
```

### `check_policy(text, block_on)`

Pre-send gate: call before pushing content to an external service. Returns whether the text is free of the blocked PII types (case-insensitive type names).

```json
{"text": "ssn is 123-45-6789", "block_on": ["ssn", "credit_card"]}
```

```json
{"allowed": false, "violations": [{"type": "ssn", "count": 1}]}
```

## Positioning

These are agent-invocable compliance tools: the agent decides when to scan, redact, or gate. For transparent interception of *all* traffic — every prompt and response, no cooperation required — put the compliance guard in the request path instead with `multimind serve` (see [compliance.md](compliance.md)), or wrap a model in-process with `multimind.compliance.guard`.
