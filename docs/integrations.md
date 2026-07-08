# Framework Integrations: Govern Your Existing Stack

**Don't migrate off LangChain — govern it.** MultiMind's governance features
(PII redaction and blocking, spend budgets, cost tracking, JSONL audit
trails) ship as drop-in wrappers for the frameworks you already use:
LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, and raw OpenAI clients.
Each adapter adds a few lines to an existing app and removes nothing; your
chains, indexes, crews, agents, and API calls keep working exactly as
before, with a compliance layer in front of every LLM call.

All adapters live in `multimind.integrations.frameworks` and resolve
lazily: importing the package never requires any framework, and a missing
framework raises a clear `pip install ...` hint only when its adapter is
actually used (the Haystack adapter is the one exception — see below).

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

The PII guard wrappers (`guard_runnable`, `guard_tool`/`guard_tools`,
`guard_llm`, `guard_crew_llm`, `guard_autogen_client`,
`guard_haystack_generator`, `guard_openai`) take the compliance kwargs; the
callback handlers take the cost kwargs. `guard_crew_llm`, `guard_autogen_client`,
and `guard_openai` take both.

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

### Guard agent tool inputs

```python
from my_tools import search_tool, calculator_tool

# --- add these 2 lines ---
from multimind.integrations.frameworks import guard_tools
tools = guard_tools([search_tool, calculator_tool], block_on=("ssn",))
# --------------------------

agent = create_tool_calling_agent(llm, tools, prompt)   # unchanged
```

`guard_tool`/`guard_tools` wrap a `Tool`/`BaseTool` and return a `BaseTool`
themselves — `name`, `description`, and `args_schema` are copied from the
wrapped tool, so LLM tool-binding (`bind_tools`, `convert_to_openai_tool`,
...) sees an identical schema and agents can't tell the difference. Only
the tool's *input* — the string or dict argument the agent hands it — is
screened/redacted before it reaches the tool's `_run`/`_arun`; this is the
gap `guard_runnable` cannot close on its own, since intermediate tool
arguments in an agent loop never pass through the wrapped LLM Runnable.

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
It also runs the PII detector over every outgoing prompt and writes a
`pii_detected` audit event (types/counts only) — see the honest notes below
for why detection, not redaction, is what a callback can offer here.

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
  intermediate tool inputs bypass `guard_runnable` unless the tool-calling
  model itself is the wrapped runnable; use `guard_tool`/`guard_tools` to
  close that specific gap.
- `guard_tool` screens the tool's *input* only; the tool's return value is
  not screened by this wrapper — route the agent's LLM through
  `guard_runnable` to catch PII flowing back from tool output through the
  model. Tool-role messages (a tool's own execution results fed back into
  the conversation) aren't reconstructed by `guard_tool` either.
- The callback handler is observe-only **by construction, not omission**:
  we checked — `langchain_core.callbacks.manager.handle_event` invokes every
  handler's `on_llm_start`/`on_chat_model_start` purely for side effects and
  discards the return value, so no LangChain callback, this one included,
  can rewrite the prompt actually sent to the model. That's why it detects
  and audits PII instead of pretending to redact it; combine with
  `guard_runnable` when you need the prompt itself changed.
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
- Same verified observe-only limitation as the LangChain handler:
  `CallbackManager.on_event_start` discards each handler's return value, so
  this handler cannot rewrite the prompt either — it detects and audits PII
  (`pii_detected` events) instead of redacting it.

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
- Verified against crewai 1.15.1, installed and exercised directly (not
  just faked): `BaseLLM` is a Pydantic v2 model whose `call`/`acall` accept
  `messages: str | list[LLMMessage]`, where `LLMMessage.content` may be a
  plain string or a list of content parts (both are redacted). Report
  interface drift on newer crewai releases as a bug.

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

`client.responses.create` (the Responses API) and `client.embeddings.create`
are patched the same way whenever the client exposes them — each guarded
independently and tolerant of absence, so an older SDK client missing one or
both still gets the rest:

```python
client = guard_openai(client, budget=Budget(max_cost=5.0))

client.responses.create(model="gpt-4o", input="...")             # input + output redacted
client.embeddings.create(model="text-embedding-3-small", input=[...])  # input redacted
```

For `responses.create`: `input` (a string, or a list of message dicts whose
`content` is a string or a list of `{"type": "input_text", "text": ...}`
parts) and `instructions` are redacted before dispatch; output text in
`response.output[*].content[*].text` is redacted after. `stream=True`
behaves like the chat-completions stream: events pass through unmodified,
and the accumulated text plus the terminal event's usage are scanned/costed
after the last event.

For `embeddings.create`: `input` — a string or a list of strings — is
redacted before dispatch; token-ID inputs (`Iterable[int]` /
`Iterable[Iterable[int]]`) have no text to redact and pass through
unchanged. There is no response text to redact (embeddings return vectors),
so only the request side and cost (`prompt_tokens`, no output tokens) are
guarded.

Honest notes:

- `completions` (the legacy, non-chat API) and the `chat.completions.stream(...)`
  helper are still untouched.
- Streaming output (both APIs) is passthrough + post-scan, not redacted
  mid-stream. If you need output held back until scanned, use the
  non-streaming call — or run the [guard proxy](guard-proxy.md) in front of
  the API instead.
- Tool-call arguments in chat-completions/responses output are not scanned.

---

## AutoGen

Requires `pip install autogen-core` (the foundation package used by both
AutoGen/autogen-agentchat and any custom model client built against its
documented interface).

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# --- add these 3 lines ---
from multimind.integrations.frameworks import guard_autogen_client
from multimind.observability import Budget
model_client = guard_autogen_client(model_client, budget=Budget(max_cost=5.0))
# -------------------------

agent = AssistantAgent("assistant", model_client=model_client)  # unchanged
```

**Interface choice:** the adapter wraps `autogen_core.models.ChatCompletionClient`
— the documented, versioned abstract base class every AutoGen/autogen-agentchat
model client implements (`OpenAIChatCompletionClient`,
`AzureOpenAIChatCompletionClient`, and any custom subclass), and the seam
the AutoGen docs point to for building a custom model client. AG2's
alternative, a loosely-typed `llm_config` dict consumed by
`autogen.OpenAIWrapper`, has no formal ABC and no stable method contract to
guard — `ChatCompletionClient` is the clearer, more stable choice between
the two. Verified directly against autogen-core 0.7.5 (a lightweight
package — its only extra pin is protobuf — so it's installed in this
project's dev/test environment rather than faked).

`guard_autogen_client` returns a real `ChatCompletionClient` (an
`isinstance` match), so it drops in anywhere an agent accepts a model
client. `create` and `create_stream` are guarded: `SystemMessage.content`,
`UserMessage.content` (string or the string entries of a list of
str/`Image` parts), plain-text `AssistantMessage.content`, and
`FunctionExecutionResultMessage`'s result strings are all redacted before
the call; response `CreateResult.content` is redacted after. Every other
abstract method (`actual_usage`, `total_usage`, `count_tokens`,
`remaining_tokens`, `close`, `model_info`, `capabilities`) is proxied
straight through.

Honest notes:

- `AssistantMessage.content` that is a list of `FunctionCall` (the model's
  own tool-call arguments) passes through unscreened — same tool-call
  handling as the other adapters.
- `create_stream`'s string chunks are passthrough, not redacted (they've
  already left the process); the terminal `CreateResult` is redacted, and
  the raw streamed text is scanned into the audit trail.
- Cost is recorded from `CreateResult.usage`/the stream's terminal
  `response.completed` event when present; there's no framework-wide
  estimate fallback beyond what `record_usage` already does from chars/4.

---

## Haystack

No install required for the adapter itself — `guard_haystack_generator` is
duck-typed and never imports `haystack` (see below for why). Target Haystack
2.x's chat generators (`OpenAIChatGenerator`, `AzureOpenAIChatGenerator`, or
any component with the same `run`/`run_async` contract).

```python
from haystack.components.generators.chat import OpenAIChatGenerator

generator = OpenAIChatGenerator(model="gpt-4o")

# --- add these 2 lines ---
from multimind.integrations.frameworks import guard_haystack_generator
generator = guard_haystack_generator(generator, block_on=("credit_card",))
# --------------------------

generator.run(messages=[...])  # unchanged call shape
```

`guard_haystack_generator` wraps `run` (always) and `run_async` (if the
component defines one); every other attribute is proxied to the wrapped
component. Message screening targets Haystack 2.x's *documented public*
`ChatMessage` API — `.text`, `.role`, `.meta`, and the
`ChatMessage.from_system`/`from_user`/`from_assistant` factory methods —
rather than its private dataclass fields, so redacted messages are
reconstructed through the same constructors application code would use.

Honest notes:

- Because the project doesn't install `haystack-ai` (a heavy, torch-adjacent
  dependency) just for this adapter, it is verified here against a
  `sys.modules`-independent fake mirroring that documented `ChatMessage`
  shape, not the installed package — report interface drift as a bug.
- Tool-role messages (and any message with an unrecognized role) are left
  unmodified rather than guessed at: reconstructing them needs the
  originating `ToolCall`/tool result, which this generic wrapper doesn't
  have.
- Not itself a `@component`-decorated Haystack component — it cannot be
  inserted into a `Pipeline` via `add_component()`. Call
  `.run()`/`.run_async()` directly, or guard a generator used outside
  pipeline orchestration.
- Cost is recorded from the first reply's `meta["model"]`/`meta["usage"]`
  (the shape `OpenAIChatGenerator`-style components populate); components
  that don't set `meta["usage"]` record at chars/4 estimate via the usual
  `record_usage` fallback.

---

## Alternative: the zero-code option

If you would rather not touch application code at all, `multimind serve`
(see [guard-proxy.md](guard-proxy.md)) provides the same redaction, budget,
and audit controls as an OpenAI-compatible HTTP proxy — point any
framework's `base_url` at it.
