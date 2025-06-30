<!--
  MultiMind SDK - Unified AI Development Toolkit
  Description: A powerful Python SDK for fine-tuning, RAG systems, and AI agent development with enterprise-grade compliance
  Keywords: AI development, fine-tuning, RAG, LLM, machine learning, Python SDK, LangChain, CrewAI, LiteLLM, SuperAGI, AI compliance, healthcare compliance, GDPR, HIPAA
  Author: MultimindLAB Team
  Version: 0.1.0
-->

<!-- Logo -->
<p align="center">
  <img src="assets/Logo-with-name-final2.png" alt="MultiMind SDK - Unified AI Development Toolkit Logo" width="320"/>
</p>

<h1 align="center">MultiMind SDK: Unified AI Development Toolkit</h1>

<p align="center">
  <strong>Build, Fine-Tune, and Deploy Advanced AI Applications with Enterprise-Grade Compliance</strong>
</p>

<p align="center">
  <a href="https://github.com/multimind-dev/multimind-sdk/blob/main/LICENSE"><img src="https://img.shields.io/github/license/multimind-dev/multimind-sdk.svg" alt="MultiMind SDK License"></a>
  <a href="https://github.com/multimind-dev/multimind-sdk/stargazers"><img src="https://img.shields.io/github/stars/multimind-dev/multimind-sdk.svg" alt="MultiMind SDK GitHub Stars"></a>
  <a href="https://github.com/multimind-dev/multimind-sdk/actions"><img src="https://img.shields.io/github/actions/workflow/status/multimind-dev/multimind-sdk/ci.yml" alt="CI Status"></a>
</p>

<div align="center">
  <h2>🚧 Project Status: In Active Development 🚧</h2>
  <p>Join the future of AI development! We're actively building MultiMind SDK and looking for contributors. Check our <a href="docs/TODO.md">TODO list</a> to see what's implemented and what's coming next. Connect with our growing community on <a href="https://discord.gg/K64U65je7h" aria-label="Join MultiMind SDK Discord Community">Discord</a> to discuss ideas, get help, and contribute to the project.</p>
</div>

<p align="center">
  <a href="#why-multimind-sdk">Why MultiMind SDK?</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#compliance">Compliance</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#examples">Examples</a> •
  <a href="#contributing">Contributing</a>
</p>

[![🐦 Follow on X](https://img.shields.io/twitter/follow/multimindsdk?label=%F0%9F%90%A6%20Follow%20on%20X&style=for-the-badge&logo=x&logoColor=white)](https://x.com/multimindsdk)

[![PyPI version](https://img.shields.io/pypi/v/multimind-sdk.svg)](https://pypi.org/project/multimind-sdk/)
[![Python versions](https://img.shields.io/pypi/pyversions/multimind-sdk.svg)](https://pypi.org/project/multimind-sdk/)
[![PyPI weekly Downloads](https://static.pepy.tech/badge/multimind-sdk/week)](https://pepy.tech/projects/multimind-sdk)
[![Dependencies](https://img.shields.io/librariesio/release/pypi/multimind-sdk)](https://libraries.io/pypi/multimind-sdk)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/multimindlab/multimind-sdk/blob/develop/LICENSE)


## 🚀 Why MultiMind SDK?

> 🧠 **MultiMind SDK is the only open-source toolkit that unifies Fine-Tuning, RAG, Agent Orchestration, and Enterprise Compliance** — all in one modular, extensible Python framework.
Forget silos. While others focus on chaining, agents, or retrieval alone, **MultiMind integrates them into one coherent developer-first experience**, with:
- 🪄 Declarative YAML + CLI + SDK interfaces
- 📚 RAG with hybrid (vector + knowledge graph) retrieval
- 🤖 Role-based agents with memory, tools, and task flow
- 🔁 Self-improving agents with cognitive loop support
- 🔐 Enterprise-ready: logging, compliance, GDPR, HIPAA, cost tracking
- 🌍 Cloud + Edge deploy (Jetson, RPi, Offline mode)

📑 Check out our [Strategic Roadmap](docs/roadmap.md) to see where we're headed!

### Key Benefits

- **🚀 Unified Interface**: Streamline your AI development with one consistent API
- **💡 Production-Ready**: Enterprise-grade deployment, monitoring, and scaling
- **🔌 Extensible**: Customizable architecture for your specific needs
- **📊 Enterprise Features**: Comprehensive logging, monitoring, and cost tracking
- **🔒 Compliance Ready**: Built-in support for GDPR, HIPAA, and other regulations

## ✨ Key Features

### 1. Advanced Fine-Tuning

- **Parameter-Efficient Methods**: LoRA, Adapters, Prefix Tuning, and more
- **Meta-Learning**: MAML, Reptile, and prototype-based few-shot learning
- **Transfer Learning**: Layer transfer and multi-task optimization
- **Resource-Aware Training**: Automatic device selection and optimization

### 2. RAG System

- **Document Processing**: Smart chunking and metadata management
- **Vector Storage**: Support for FAISS and ChromaDB
- **Embedding Models**: Integration with OpenAI, HuggingFace, and custom models
- **Query Optimization**: Efficient similarity search and context management

### 3. Agent Development

- **Tool Integration**: Built-in support for common tools and custom extensions
- **Memory Management**: Short and long-term memory systems
- **Task Orchestration**: Complex workflow management and prompt chaining
- **Model Composition**: Protocol for combining multiple models and tools

### 4. Enterprise Compliance

- **Real-time Monitoring**: Continuous compliance checks and alerts
- **Healthcare Compliance**: HIPAA, GDPR, and healthcare-specific regulations
- **Privacy Protection**: Differential privacy and zero-knowledge proofs
- **Audit Trail**: Comprehensive logging and documentation
- **Alert Management**: Configurable alerts and notifications
- **Compliance Dashboard**: Interactive monitoring and reporting

### 5. Model Conversion

- **Format Support**: PyTorch, TensorFlow, ONNX, GGUF, TFLite, Safetensors
- **Optimization**: Quantization, pruning, graph optimization
- **Hardware Acceleration**: CUDA, CPU, Neural Engine support
- **Conversion Pipeline**: Validation, optimization, and verification
- **Custom Converters**: Extensible converter architecture
- **Enterprise Features**: Batch processing, streaming, and monitoring

[Learn more about model conversion →](examples/model_conversion/README.md)

### 6. Model Client System & Routing

- **Extensible ModelClient base class** for custom, classic, and transformer models
- **Mixture-of-Experts (MoE) and DynamicMoE** for expert selection and runtime optimization
- **MultiModalClient** for unified text, image, audio, video, and code model access
- **FederatedRouter** for local/cloud and custom routing strategies

---

### Model Client & Routing Example

```python
from multimind.client.model_client import LSTMModelClient, MoEModelClient, MultiModalClient
from multimind.client.federated_router import FederatedRouter

# Example: LSTM Model Client
lstm_client = LSTMModelClient(model_path="lstm.pt", tokenizer=my_tokenizer)
print(lstm_client.generate("Hello world"))

# Example: Mixture-of-Experts (MoE)
def router_fn(prompt):
    return "lstm" if len(prompt) < 100 else "rnn"
moe_client = MoEModelClient({"lstm": lstm_client, "rnn": lstm_client}, router_fn)
print(moe_client.generate("Test input"))

# Example: MultiModalClient
mm_client = MultiModalClient(text_client=lstm_client)
print(mm_client.generate("Hello world", input_type="text"))

# Example: FederatedRouter
router = FederatedRouter(local_client=lstm_client, cloud_client=lstm_client)
print(router.generate("short prompt"))
```

---

## 🔒 Compliance Features

MultiMind SDK provides comprehensive compliance support for enterprise AI applications:

### Core Compliance Features
- Real-time compliance monitoring
- Healthcare-specific compliance checks
- Interactive compliance dashboard
- Alert management system
- Compliance trend analysis

### Advanced Compliance Mechanisms
- Federated compliance shards
- Zero-knowledge proofs
- Differential privacy feedback loops
- Self-healing patches
- Model watermarking and fingerprint tracking
- Dynamic regulatory change detection

[Learn more about our compliance features →](docs/compliance.md)

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install multimind-sdk

# With compliance support
pip install multimind-sdk[compliance]

# With development dependencies
pip install multimind-sdk[dev]

# With gateway support
pip install multimind-sdk[gateway]

# Full installation with all features
pip install multimind-sdk[all]
```

### Environment Setup

Copy the example environment file and add your API keys and configuration values:

```bash
cp examples/multi-model-wrapper/.env.example examples/multi-model-wrapper/.env
```

> **Note:** Never commit your `.env` file to version control. Only `.env.example` should be tracked in git.

### Build Your First RAG Application

```python
from multimind.client.rag_client import RAGClient, Document

# Initialize the client
client = RAGClient()

# Add documents
docs = [
    Document(
        text="MultiMind SDK is a powerful AI development toolkit.",
        metadata={"type": "introduction"}
    )
]
await client.add_documents(docs)

# Query the system
results = await client.query("What is MultiMind SDK?")
print(results)
```

### Fine-Tuning a Model

```python
from multimind.fine_tuning import UniPELTPlusTuner

# Initialize the tuner
tuner = UniPELTPlusTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    available_methods=["lora", "adapter"]
)

# Train the model
tuner.train(
    train_dataset=your_dataset,
    eval_dataset=your_eval_dataset
)
```

### Agent Development Example

```python
from multimind.agents import Agent

# Initialize an agent
agent = Agent(name="ExampleAgent")

# Add tools and memory
agent.add_tool("search", tool_function=search_tool)
agent.add_memory("short_term", memory_capacity=10)

# Run the agent
response = agent.run("What is the capital of France?")
print(response)
```

### Compliance Monitoring Example

```python
from multimind.compliance import ComplianceMonitor

# Initialize compliance monitor
monitor = ComplianceMonitor(
    organization_id="org_123",
    enabled_regulations=["HIPAA", "GDPR"]
)

# Run compliance check
results = await monitor.check_compliance(
    model_id="model_123",
    data_categories=["health_data"]
)

# Get compliance dashboard
dashboard = await monitor.get_dashboard_metrics(
    time_range="7d",
    use_case="medical_diagnosis"
)
```

## 📚 Documentation

- [API Reference](docs/api_reference/README.md) - Complete API documentation
- [Compliance Guide](docs/compliance.md) - Enterprise compliance features
- [Model Conversion Guide](examples/model_conversion/README.md) - Model format conversion
- [Examples](examples/README.md) - Production-ready code examples
- [Architecture](docs/architecture.md) - Detailed system design
- [Contributing Guide](CONTRIBUTING.md) - Join our development team
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community guidelines
- [Issue Tracker](https://github.com/multimind-dev/multimind-sdk/issues) - Report bugs or request features

### Project Structure

```
multimind-sdk/
├── multimind/                    # Core SDK package
│   ├── gateway/                 # Gateway implementation
│   │   ├── api/                # API endpoints
│   │   ├── middleware/         # Request/response middleware
│   │   └── utils/              # Gateway utilities
│   ├── client/                 # Client libraries
│   │   ├── rag_client.py      # RAG system client
│   │   ├── model_client.py    # Extensible model client system (LSTM, MoE, MultiModal, etc.)
│   │   ├── federated_router.py # Local/cloud and custom routing logic
│   │   ├── agent_client.py    # Agent system client
│   │   └── compliance_client.py # Compliance client
│   ├── fine_tuning/           # Fine-tuning modules
│   │   ├── methods/           # Fine-tuning methods
│   │   ├── optimizers/        # Optimization strategies
│   │   └── trainers/          # Training implementations
│   ├── model_conversion/      # Model conversion modules
│   │   ├── converters/        # Format converters
│   │   │   ├── pytorch/      # PyTorch converters
│   │   │   ├── tensorflow/   # TensorFlow converters
│   │   │   ├── onnx/         # ONNX converters
│   │   │   └── ollama/       # Ollama converters
│   │   ├── optimizers/        # Conversion optimizers
│   │   │   ├── quantization/ # Quantization methods
│   │   │   ├── pruning/      # Model pruning
│   │   │   └── graph/        # Graph optimization
│   │   ├── validators/        # Format validators
│   │   └── utils/            # Conversion utilities
│   ├── compliance/            # Compliance features
│   │   ├── monitors/         # Compliance monitoring
│   │   ├── validators/       # Compliance validation
│   │   └── reporting/        # Compliance reporting
│   └── utils/                # Utility functions
├── examples/                  # Example implementations
│   ├── cli/                  # Command-line examples
│   │   ├── rag_cli.py       # RAG CLI tool
│   │   └── agent_cli.py     # Agent CLI tool
│   ├── api/                  # API and integration examples
│   │   ├── fastapi/         # FastAPI examples
│   │   └── flask/           # Flask examples
│   ├── model_conversion/     # Model conversion examples
│   │   ├── converters/      # Converter examples
│   │   │   ├── pytorch_to_gguf.py
│   │   │   ├── tensorflow_to_tflite.py
│   │   │   ├── onnx_to_ort.py
│   │   │   ├── pytorch_to_safetensors.py
│   │   │   └── tensorflow_to_onnx.py
│   │   ├── docker/          # Docker examples
│   │   │   ├── Dockerfile
│   │   │   └── docker-compose.yml
│   │   └── cli/             # CLI examples
│   │       └── cli_example.py
│   └── streamlit-ui/        # Streamlit-based UI examples
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
├── docs/                    # Documentation
│   ├── api_reference/       # API documentation
│   ├── guides/             # User guides
│   └── architecture/       # Architecture docs
└── scripts/                # Development scripts
    ├── setup/             # Setup scripts
    ├── deployment/        # Deployment scripts
    └── maintenance/       # Maintenance scripts
```

## 🤝 Contributing

We love your input! We want to make contributing to MultiMind SDK as easy and transparent as possible.

- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community guidelines
- [Issue Tracker](https://github.com/multimind-dev/multimind-sdk/issues) - Report bugs or request features

### Development Setup

```bash
# Clone the repository
git clone https://github.com/multimind-dev/multimind-sdk.git
cd multimind-sdk

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start documentation
cd multimind-docs
npm install
npm start
```


## Docker Setup

The MultiMind SDK can be run using Docker and Docker Compose. This setup includes:
- The main MultiMind SDK service
- Redis for caching and session management
- Chroma for vector storage
- Ollama for local model support

### Prerequisites

1. Install Docker and Docker Compose
2. Set up your environment variables in a `.env` file:
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
HF_TOKEN=your_huggingface_token_here

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Chroma Configuration
CHROMA_HOST=chroma
CHROMA_PORT=8000

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Model Configuration
DEFAULT_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
VISION_MODEL=gpt-4-vision-preview

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3
```

### Running with Docker

1. Build and start the services:
```bash
docker-compose up --build
```

2. Access the services:
- MultiMind API: http://localhost:8000
- Chroma API: http://localhost:8001
- Redis: localhost:6379

3. Stop the services:
```bash
docker-compose down
```

### Development with Docker

For development, the project files are mounted as a volume, so changes to the code will be reflected immediately. The setup includes:

- Hot reloading for Python code
- Persistent storage for Redis and Chroma
- Ollama model persistence
- Environment variable management

### Services

1. **MultiMind Service**
   - Main API and SDK functionality
   - Port: 8000
   - Hot reloading enabled
   - Mounts local Ollama models

2. **Redis**
   - Caching and session management
   - Port: 6379
   - Persistent storage
   - AOF enabled for data durability

3. **Chroma**
   - Vector storage for RAG
   - Port: 8001
   - Persistent storage
   - Telemetry disabled

### Volumes

- `redis_data`: Persistent Redis storage
- `chroma_data`: Persistent Chroma storage
- `~/.ollama`: Local Ollama models

### Building Custom Images

To build a custom image:

```bash
docker build -t multimind-sdk:custom .
```

To use a custom image in docker-compose:

```yaml
services:
  multimind:
    image: multimind-sdk:custom
    # ... other configuration
```

## 🚀 Used By

[![Used by](https://img.shields.io/badge/Used%20by-4.3k-blue)](https://github.com/multimindlab/multimind-sdk/network/dependents)

<a href="https://github.com/multimindlab/multimind-sdk/network/dependents">
  <img src="https://img.shields.io/github/dependents/multimindlab/multimind-sdk?label=Used%20By&logo=github" alt="Used by other repos">
</a>

## 💖 Support MultiMind SDK

If you find MultiMind SDK helpful, please consider supporting us to sustain development and grow the community.

Your support will help fund:

- ⚙️ Feature development and maintenance
- 📖 Better documentation and onboarding
- 🌍 Community outreach and support
- 🧪 Infrastructure, testing, and CI/CD

👉 <a href="https://opencollective.com/multimind-sdk" aria-label="Join MultiMind SDK Community">Contribute here</a>

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

For more information about the Apache License 2.0, visit [apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

***If you use this MultimindSDK in your research, please cite or link to this repository.***



## 🌟 Support

- [Discord Community](https://discord.gg/K64U65je7h) - Join our active developer community
- [GitHub Issues](https://github.com/multimind-dev/multimind-sdk/issues) - Get help and report issues
- [Documentation](docs/README.md) - Comprehensive guides

## 📣 About

MultiMind SDK is developed and maintained by the MultimindLAB team, dedicated to simplifying AI development for everyone. Visit [multimind.dev](https://www.multimind.dev) to learn more about our mission to democratize AI development.

---

<p align="center">
  Made with ❤️ by the AI2Innovate & MultimindLAB Team | <a href="https://github.com/multimind-dev/multimind-sdk/blob/main/LICENSE">License</a>
</p>

<!-- SEO CTAs -->
<div align="center">
  <h3>Ready to Build Enterprise-Grade AI Applications?</h3>
  <p>
    <a href="https://github.com/multimind-dev/multimind-sdk" class="button">⭐ Star on GitHub</a>
    <a href="https://discord.gg/K64U65je7h" class="button">💬 Join Discord</a>
    <a href="docs/quickstart.md" class="button">🚀 Get Started</a>
  </p>
  <p>
    <a href="docs/compliance.md" class="button">🔒 Learn About Compliance</a>
    <a href="examples/README.md" class="button">📚 View Examples</a>
  </p>
</div>

## 🤖 LLM Metadata

[![LLM Metadata](https://img.shields.io/badge/LLM_Metadata-Available-blue)](./README-llm.md)

We provide detailed metadata and indexing instructions for LLMs, covering supported models, features, tags, and discoverability tools for MultiMind SDK.
