# MultiMind SDK Installation Guide

## Quick Start

### Basic Installation (Core Only)
```bash
pip install multimind-sdk
```
This installs the core: multi-model chat (OpenAI, Claude, Ollama), routing, CLI, and context transfer. Ollama needs no extra — the client talks to a running Ollama instance over HTTP.

---

## Optional Features (Install What You Need)

These are the extras actually defined in `pyproject.toml`:

### RAG Support
```bash
pip install "multimind-sdk[rag]"
```
Adds: faiss-cpu, sentence-transformers, beautifulsoup4, lxml, numpy

### Additional Vector Store Backends
```bash
pip install "multimind-sdk[vector-stores]"
```
Adds: chromadb, qdrant-client, weaviate-client, pinecone-client, pymilvus, elasticsearch

### Agent Framework
```bash
pip install "multimind-sdk[agents]"
```
Adds: the `[memory]` extra (redis, numpy) that agents rely on

### Memory Backends
```bash
pip install "multimind-sdk[memory]"
```
Adds: redis, numpy

### Document Processing
```bash
pip install "multimind-sdk[documents]"
```
Adds: pdfplumber, PyPDF2, python-docx, python-pptx, pillow, pytesseract, unstructured

### Fine-tuning (CPU / general)
```bash
pip install "multimind-sdk[finetune]"
```
Adds: torch, transformers, datasets, accelerate, peft, scikit-learn, optuna. Also required for the non-transformer models (Mamba, RWKV), which are HuggingFace-backed.

### Fine-tuning (GPU, Linux + CUDA only)
```bash
pip install "multimind-sdk[finetune-gpu]"
```
Adds: everything in `[finetune]` plus bitsandbytes (Linux only; does not install on macOS/ARM)

### Compliance Features
```bash
pip install "multimind-sdk[compliance]"
```
Adds: cryptography, bcrypt, pycryptodome, plotly, dash, pandas (dashboard visualization included)

### Gateway / API Server
```bash
pip install "multimind-sdk[gateway]"
```
Adds: fastapi, uvicorn, redis, python-jose, python-multipart

### MCP Server (Claude Desktop / Claude Code integration)
```bash
pip install "multimind-sdk[mcp]"
```
Adds: mcp (the Model Context Protocol SDK) — needed for `multimind serve-mcp` / the MCP server described in [docs/mcp-server.md](./docs/mcp-server.md)

### Framework Integrations
```bash
pip install "multimind-sdk[langchain]"    # langchain-core
pip install "multimind-sdk[llamaindex]"   # llama-index-core
pip install "multimind-sdk[crewai]"       # crewai
```
Each adds only the corresponding framework's core package, for the adapters described in [docs/integrations.md](./docs/integrations.md).

### Everything
```bash
pip install "multimind-sdk[all]"
```
All of the above except `dev`, `finetune-gpu`, and the framework extras (`langchain`, `llamaindex`, `crewai`) — large download; includes torch and ML frameworks.

> Note: extras like `[router]`, `[llm]`, `[minimal]`, and `[fine-tuning]` documented in older versions of this guide do not exist. The router and LLM providers are part of the core install.

---

## Development Setup

### Clone & Install for Development
```bash
git clone https://github.com/multimindlab/multimind-sdk.git
cd multimind-sdk
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev tools
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest
pytest -v --cov=multimind  # With coverage report
```

### Code Quality
```bash
# Lint and format (ruff handles both)
ruff check multimind/
ruff format multimind/

# Type checking
mypy multimind/
```

See the `Makefile` for shortcuts (`make lint`, `make format`, `make test`, ...).

---

## Dependency Matrix

| Extra | Key Dependencies |
|-------|------------------|
| core (no extra) | openai, anthropic, httpx, pydantic, pydantic-settings, python-dotenv, click, rich, PyYAML, aiohttp, requests, tenacity, coloredlogs |
| `rag` | faiss-cpu, sentence-transformers, beautifulsoup4, lxml, numpy |
| `vector-stores` | chromadb, qdrant-client, weaviate-client, pinecone-client, pymilvus, elasticsearch |
| `agents` | multimind-sdk[memory] |
| `memory` | redis, numpy |
| `documents` | pdfplumber, PyPDF2, python-docx, python-pptx, pillow, pytesseract, unstructured |
| `finetune` | torch, transformers, datasets, accelerate, peft, scikit-learn, optuna |
| `finetune-gpu` | multimind-sdk[finetune] + bitsandbytes (Linux only) |
| `compliance` | cryptography, bcrypt, pycryptodome, plotly, dash, pandas |
| `gateway` | fastapi, uvicorn, redis, python-jose, python-multipart |
| `mcp` | mcp |
| `langchain` | langchain-core |
| `llamaindex` | llama-index-core |
| `crewai` | crewai |
| `dev` | pytest, ruff, mypy, pre-commit, bandit, pip-audit, build, twine, sphinx |

---

## Troubleshooting

### Issue: ModuleNotFoundError for specific feature
**Solution:** Install the corresponding extra:
```bash
pip install "multimind-sdk[extra-name]"
```

### Issue: CUDA/GPU support for PyTorch
**Solution:** Install the GPU build of torch first, then the extra:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "multimind-sdk[finetune]"
```

### Issue: bitsandbytes fails to install on macOS/ARM
**Solution:** `bitsandbytes` is Linux/CUDA-only. Use `[finetune]` instead of `[finetune-gpu]` on macOS.

---

## Verification

Test your installation:
```python
import multimind
print(multimind.__version__)

# Core model wrappers (no extras needed)
from multimind import OpenAIModel, ClaudeModel
from multimind.models.ollama import OllamaModel
print("Core models import OK")

# Optional features (if installed)
try:
    from multimind.vector_store.base import VectorStoreFactory
    print("Vector store module OK")
except ImportError:
    print("Vector stores not installed - run: pip install 'multimind-sdk[rag]'")
```

---

## Next Steps

1. Check out the [Getting Started Guide](./docs/quickstart.md)
2. Browse [Examples](./examples/)
3. Read the [API Reference](./docs/api_reference/)
4. Join our [Discord Community](https://discord.gg/K64U65je7h)
