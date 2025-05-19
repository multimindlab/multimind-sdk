# Cursor.dev Prompt Templates for Step-by-Step MultiMind SDK Development

# ------------------------------
# 1. LangChain Adapter
# ------------------------------
"""
Create a file: adapters/langchain_adapter.py
Goal: Wrap LangChain chains into a callable MultiMind agent format.
"""

Create a Python class `LangChainAdapter` that:
- Accepts a chain config or chain object
- Implements `run(input: str)` and returns response
- Follows the BaseAdapter interface
- Includes example using `LLMChain` from LangChain
- Supports loading chain from YAML if path given

# ------------------------------
# 2. Routing Logic Engine
# ------------------------------
"""
Create a file: router/engine.py
Goal: Enable smart fallback or routing across multiple models/adapters.
"""

Build a `RouterEngine` class that:
- Accepts a routing config (YAML/JSON)
- Defines fallback strategy: primary, fallback, condition
- Tries each adapter in order until success
- Logs response time, source adapter, and success status
- Expose `route(input: str)` method

# ------------------------------
# 3. Multi-Agent Orchestration Engine
# ------------------------------
"""
Create a file: orchestrator/engine.py
Goal: Compose agents from multiple SDKs using config (YAML/JSON or dict).
"""

Implement `AgentOrchestrator` class:
- Accepts agent config (roles, adapters, memory store)
- Supports turn-based conversation
- Maintains memory dict or Redis
- Logs full transcript
- Includes `run_conversation(task: str)` method

# ------------------------------
# 4. Fine-tuning with QLoRA
# ------------------------------
"""
Create file: finetune/train.py
Goal: Fine-tune LLaMA or Qwen with LoRA/QLoRA.
"""

Build a CLI script that:
- Accepts base model, dataset (JSONL/CSV), and LoRA config
- Trains using PEFT + HuggingFace + bitsandbytes
- Saves adapter to `./adapters/` with name
- Use `argparse` for CLI

# ------------------------------
# 5. Unified API for All Models
# ------------------------------
"""
Create file: api/unified.py
Goal: One interface to interact with OpenAI, Claude, Ollama, etc.
"""

Build `UnifiedLLM` class:
- Accepts model name or adapter name
- Supports `.chat()`, `.completion()`, `.embed()` methods
- Internally routes to correct adapter
- Example: `llm = UnifiedLLM(model='openai:gpt-4')`

# ------------------------------
# 6. CLI Entrypoint
# ------------------------------
"""
Create file: cli/multimind.py
Goal: CLI to run models, agents, RAG, and fine-tuning.
"""

Use `click` to create commands:
- `multimind run --adapter langchain --task chatbot`
- `multimind route --config route.yaml --prompt "Hello"`
- `multimind agents --config agents.yaml`
- `multimind finetune --model qwen --dataset sales.json`

# ------------------------------
# 7. Plugin Loader System
# ------------------------------
"""
Create file: plugins/loader.py
Goal: Load external Python plugins (adapters, agents).
"""

Implement `PluginLoader` class:
- Loads `.py` files from `/plugins` dir
- Uses `importlib` to load dynamically
- Calls `register()` method from each plugin
- Logs available plugins

# ------------------------------
# 8. Benchmark CLI Tool
# ------------------------------
"""
Add to cli/multimind.py
Command: benchmark
Goal: Compare prompt output across multiple adapters.
"""

Add CLI command:
`multimind benchmark --prompt "Explain LLMs" --models openai,ollama,langchain`

- Calls each model and logs:
  - Response
  - Latency
  - Token usage (if available)
- Outputs result in table or JSON

# ------------------------------
# 9. Agent YAML Config Schema
# ------------------------------
"""
Sample YAML Config: multimind.yaml
"""

agents:
  - name: Researcher
    adapter: openai
    role: "Finds background info"
  - name: Summarizer
    adapter: langchain
    role: "Summarizes results"
  - name: Closer
    adapter: local:ollama
    role: "Writes conclusions"

task: "Generate market report on AGI startups."

# ------------------------------
# 10. Swagger API + FastAPI Setup
# ------------------------------
"""
Create main.py and api/unified.py
Goal: Expose all SDK features over API.
"""

Use FastAPI:
- Endpoints: `/chat`, `/embed`, `/agents/run`, `/finetune`
- Accept JSON input
- Call `UnifiedLLM` or `AgentOrchestrator`
- Serve with Uvicorn

# ------------------------------
# End of Prompts
# ------------------------------
