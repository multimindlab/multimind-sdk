# Framework Integrations: Govern Your Existing Stack

**Don't migrate off LangChain — govern it.** MultiMind's governance features
(PII redaction and blocking, spend budgets, cost tracking, JSONL audit
trails) ship as drop-in wrappers for the frameworks you already use:
LangChain, LlamaIndex, CrewAI, and raw OpenAI clients. Each adapter adds
three lines to an existing app and removes nothing; your chains, indexes,
crews, and API calls keep working exactly as before, with a compliance
layer in front of every LLM call.

All adapters live in `multimind.integrations.frameworks` and resolve
lazily: importing the package never requires any framework, and a missing
framework raises a clear `pip install ...` hint only when its adapter is
actually used.

Shared governance kwargs (all adapters):

| kwarg | meaning |
|---|---|
| `redact_input` / `redact_output` | redact detected PII (default `True`) |
| `strategy` | `"mask"`, `"hash"`, or `"remove"` |
| `block_on` | PII types that raise `ComplianceViolationError` instead of redacting, e.g. `("ssn", "credit_card")` |
| `detector` | custom `PIIDetector` (e.g. `PIIDetector(use_presidio=True)`) |
| `audit_log` | path or stream for the JSONL audit trail (types/counts/hash tags — never raw PII) |
| `tracker` / `budget` / `tag` | `CostTracker`, `Budget` (raises `BudgetExceededError` at the ceiling), and a grouping tag |
| `pricing` | `{model_prefix: price_per_token}` (or `{"input": ..., "output": ...}`) used to cost calls made through the foreign framework; unmatched models are recorded at $0 with `unpriced=True`, never a fabricated price |

The PII guard wrappers (`guard_runnable`, `guard_llm`, `guard_crew_llm`,
`guard_openai`) take the compliance kwargs; the callback handlers take the
cost kwargs. `guard_crew_llm` and `guard_openai` take both.

---

## LangChain

Requires `pip install langchain-core` (any framework version that provides it).

### Guard any Runnable

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# --- add these 3 lines ---
from multimind.integrations.frameworks import guard_runnable
llm = guard_runnable(llm, block_on=("ssn",), audit_log="audit.jsonl")
# -------------------------

chain = prompt | llm | parser   # unchanged
```

`guard_runnable` returns a real `Runnable`, so it composes with LCEL
(`|`), agents, and `RunnableConfig` unchanged. String content in inputs
(plain strings, message lists, prompt values, dict values) is redacted
before it reaches the wrapped runnable; `invoke`/`ainvoke` outputs are
redacted; `stream`/`astream` pass chunks through untouched and scan the
accumulated text afterwards (streamed text cannot be retracted — the scan
lands in the audit trail).

### Cost + audit from any run — zero wrapping

```python
from multimind.integrations.frameworks import MultiMindCallbackHandler
from multimind.observability import Budget, CostTracker

handler = MultiMindCallbackHandler(
    tracker=CostTracker(), budget=Budget(max_cost=5.0),
    pricing={"gpt-4o-mini": {"input": 15e-8, "output": 60e-8}},
)
chain.invoke({"question": "..."}, config={"callbacks": [handler]})
```

The handler records every LLM call in the run tree — provider-reported
token usage when available (`usage_metadata` / `token_usage`), chars/4
estimate otherwise (flagged `estimated=True`) — and raises
`BudgetExceededError` before the next call once the budget is exhausted.

### Use MultiMind models inside LangChain

```python
from multimind.integrations.frameworks import MultiMindChatModel
from multimind.models.claude import ClaudeModel

llm = MultiMindChatModel(model=ClaudeModel(model_name="claude-sonnet-4-5"))
chain = prompt | llm   # a native BaseChatModel
```

Any MultiMind BaseLLM-compatible model works, including one already wrapped
in `ComplianceGuard`/`track_costs`.

Honest notes:

- Tool/function-call arguments and multimodal content blocks are **not**
  scanned by `guard_runnable` — only string content is. In agent loops,
  intermediate tool inputs bypass the guard unless the tool-calling model
  itself is the wrapped runnable.
- The callback handler observes; it does not redact. Combine with
  `guard_runnable` for redaction.
- Synchronous `invoke` on `MultiMindChatModel` runs the async MultiMind
  model via `asyncio.run` and therefore cannot be called from inside a
  running event loop — use `ainvoke` there.

---

## LlamaIndex

Requires `pip install llama-index-core`.

```python
from llama_index.core import Settings, VectorStoreIndex

# --- add these 3 lines ---
from multimind.integrations.frameworks import guard_llm
Settings.llm = guard_llm(Settings.llm, block_on=("credit_card",), audit_log="audit.jsonl")
# -------------------------

query_engine = VectorStoreIndex.from_documents(docs).as_query_engine()  # unchanged
```

`guard_llm` returns a real `llama_index` `LLM`, usable anywhere LlamaIndex
accepts one. All eight LLM methods are guarded: `complete`/`chat` (and
async) redact input and output; the four streaming variants pass chunks
through and post-scan the accumulated text.

Cost + audit via a native callback handler:

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from multimind.integrations.frameworks import MultiMindLlamaIndexHandler
from multimind.observability import Budget, CostTracker

handler = MultiMindLlamaIndexHandler(tracker=CostTracker(), budget=Budget(max_cost=5.0))
Settings.callback_manager = CallbackManager([handler])
```

Honest notes:

- The guard sees what the LLM sees: **retrieved node text is scanned**
  (it flows through the LLM prompt), but embedding calls, retrieval
  scoring, and node parsing are not intercepted. PII stored in your index
  is only caught on its way into the LLM.
- The handler costs `CBEventType.LLM` events only; usage comes from
  `response.raw` when the provider reports it, otherwise chars/4 estimate.

---

## CrewAI

Requires `pip install crewai`.

```python
from crewai import Agent, Crew, LLM

llm = LLM(model="gpt-4o")

# --- add these 3 lines ---
from multimind.integrations.frameworks import guard_crew_llm
from multimind.observability import Budget
llm = guard_crew_llm(llm, budget=Budget(max_cost=5.0), pricing={"gpt-4o": 5e-6})
# -------------------------

agent = Agent(role="researcher", goal="...", llm=llm)  # unchanged
```

The wrapper subclasses `crewai.BaseLLM`, so CrewAI treats it as a
first-class LLM; `call`/`acall` are guarded and every other attribute is
proxied to the wrapped LLM.

**Budget enforcement is the headline.** Agent crews are prone to runaway
token consumption — delegation loops, retries, verbose tool chatter. With
a `Budget` attached, the guard raises `BudgetExceededError` before the
next LLM call once the ceiling is hit, stopping a crew mid-run instead of
letting it burn through spend. Catch it around `crew.kickoff()`.

Honest notes:

- CrewAI's `call` returns no token usage, so all cost records are chars/4
  estimates (`estimated=True`); provide `pricing` or costs record as $0.
- Task descriptions, tool outputs, and agent-to-agent delegation text are
  scanned only when they pass through the LLM prompt. Tool *execution*
  (e.g. a tool that posts data externally) is not intercepted.
- This adapter is built against CrewAI's documented custom-LLM interface
  (`BaseLLM.call`/`acall`) and verified against a faithful fake, not the
  installed package — report interface drift as a bug.

---

## Raw OpenAI clients

Works with `openai.OpenAI`, `openai.AsyncOpenAI`, Azure variants, and any
OpenAI-compatible client object.

```python
from openai import OpenAI

client = OpenAI()

# --- add these 3 lines ---
from multimind.integrations.frameworks import guard_openai
from multimind.observability import Budget
client = guard_openai(client, budget=Budget(max_cost=5.0), audit_log="audit.jsonl")
# -------------------------

resp = client.chat.completions.create(model="gpt-4o", messages=[...])  # unchanged
```

`guard_openai` patches `client.chat.completions.create` in place (and
returns the same client). Message content — including `text` parts of
multimodal content lists — is redacted before dispatch; response message
content is redacted; costs are recorded from the API's own `usage` field
(exact, `estimated=False`). With `stream=True` you get a thin pass-through
wrapper: chunks arrive unmodified and the accumulated text is scanned and
costed (estimated) after the last chunk.

Honest notes:

- Only `chat.completions.create` is wrapped. `responses`, `embeddings`,
  `completions` (legacy), and the `chat.completions.stream(...)` helper
  are untouched.
- Streaming output is passthrough + post-scan, not redacted mid-stream.
  If you need output held back until scanned, use the non-streaming call —
  or run the [guard proxy](guard-proxy.md) in front of the API instead.
- Tool-call arguments in responses are not scanned.

---

## Alternative: the zero-code option

If you would rather not touch application code at all, `multimind serve`
(see [guard-proxy.md](guard-proxy.md)) provides the same redaction, budget,
and audit controls as an OpenAI-compatible HTTP proxy — point any
framework's `base_url` at it.
