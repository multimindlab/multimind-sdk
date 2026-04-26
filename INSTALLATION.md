# 📦 MultiMind SDK Installation Guide

## Quick Start

### Basic Installation (Core Only)
```bash
pip install multimind-sdk
```
This installs only the core dependencies and popular LLM providers.

---

## 🎯 Optional Features (Install What You Need)

### With Router Module
```bash
pip install multimind-sdk[router]
```
Adds: FastAPI, Uvicorn, HTTP client support

### With RAG Support
```bash
pip install multimind-sdk[rag]
```
Adds: FAISS, sentence-transformers, document parsing

### With All Vector Store Backends
```bash
pip install multimind-sdk[vector-stores]
```
Adds: Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, OpenSearch, etc.

### With Advanced Document Processing
```bash
pip install multimind-sdk[documents]
```
Adds: PDF handling, DOCX, PPTX, image OCR, HTML parsing

### With Fine-tuning Support
```bash
pip install multimind-sdk[fine-tuning]
```
Adds: PyTorch, Transformers, PEFT, LoRA, QLoRA support

### With Compliance Features
```bash
pip install multimind-sdk[compliance]
```
Adds: Cryptography, security, audit logging

---

## 🔥 Pre-configured Bundles

### Minimal (Quick Start)
```bash
pip install multimind-sdk[minimal]
```
Core + LLMs only (~100MB)

### Full RAG Stack
```bash
pip install multimind-sdk[llm,rag]
```
Everything for RAG applications

### Complete Installation (Everything)
```bash
pip install multimind-sdk[all]
```
All features (~2GB, includes ML frameworks)

---

## 👨‍💻 Development Setup

### Clone & Install for Development
```bash
git clone https://github.com/multimind-dev/multimind-sdk.git
cd multimind-sdk
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev tools
pip install -e .[dev]
```

### Run Tests
```bash
pytest
pytest -v --cov=multimind  # With coverage report
```

### Code Quality
```bash
# Format code
black multimind/
isort multimind/

# Lint
ruff check multimind/

# Type checking
mypy multimind/
```

---

## 📋 Dependency Matrix

| Feature | Size | Dependencies |
|---------|------|--------------|
| `core` | ~50MB | pydantic, requests, numpy, pandas |
| `llm` | +20MB | openai, anthropic |
| `router` | +15MB | fastapi, uvicorn |
| `rag` | +100MB | faiss-cpu, sentence-transformers |
| `vector-stores` | +200MB | pinecone, weaviate, qdrant, milvus, etc. |
| `documents` | +150MB | torch, transformers, pdf tools |
| `fine-tuning` | +500MB | pytorch, transformers, peft |
| `compliance` | +10MB | cryptography |
| `dev` | +50MB | pytest, black, mypy, sphinx |

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError for specific feature
**Solution:** Install the corresponding feature group:
```bash
pip install multimind-sdk[feature-name]
```

### Issue: CUDA/GPU support for PyTorch
**Solution:** Replace `torch` with GPU version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install multimind-sdk[fine-tuning]
```

### Issue: PostgreSQL build errors (timescale-vector)
**Solution:** Vector stores are optional. If not needed, skip them. They're included in `[all]` but not in base installation.

---

## 📝 Requirements Files

### requirements.txt
- Core dependencies for end users
- Minimal set of packages
- ~30 packages total

### requirements-dev.txt
- Development and testing tools
- Code quality tools
- Documentation generators

### pyproject.toml
- Modern Python packaging standard
- Defines all optional features
- Configuration for tools (pytest, black, mypy, ruff)

---

## ✅ Verification

Test your installation:
```python
import multimind
print(multimind.__version__)

# Test core import
from multimind.core import BaseLLM
print("✅ Core module works!")

# Test LLM provider
from multimind.llm.openai_client import OpenAIClient
print("✅ LLM module works!")

# Test optional features (if installed)
try:
    from multimind.router import Router
    print("✅ Router module works!")
except ImportError:
    print("⚠️  Router not installed - run: pip install multimind-sdk[router]")
```

---

## 🚀 Next Steps

1. Check out the [Getting Started Guide](./docs/quickstart.md)
2. Browse [Examples](./examples/)
3. Read the [API Reference](./docs/api_reference/)
4. Join our [Discord Community](https://discord.gg/K64U65je7h)

