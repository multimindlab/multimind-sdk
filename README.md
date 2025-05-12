<!-- Logo -->
<p align="center">
  <img src="https://raw.githubusercontent.com/multimind-dev/multimind-sdk/main/assets/logo.png" alt="MultiMind SDK Logo" width="120"/>
</p>

# MultiMind SDK

MultiMind SDK is a powerful and flexible library for fine-tuning and adapting large language models using Parameter-Efficient Fine-Tuning (PEFT) methods. It provides advanced features for meta-learning, few-shot learning, and transfer learning, making it easier to adapt models to new tasks with minimal data.

## Features

- **Unified PEFT Framework:** LoRA, Adapters, Prefix/Prompt Tuning, IA³, and more
- **Meta-Learning:** MAML, Reptile, prototype-based few-shot learning
- **Transfer & Multi-Task Learning:** Layer transfer, dynamic task weighting, cross-task distillation
- **Adaptive Methods:** Resource-aware, performance-based, model-agnostic
- **Framework Integrations:** LangChain, CrewAI, LiteLLM, SuperAGI, and more

---

## 🚀 Installation

```bash
# Basic installation
pip install multimind-sdk

# With development dependencies
pip install multimind-sdk[dev]

# With specific framework support
pip install multimind-sdk[langchain,lite-llm,superagi]
```

---

## 🛠️ Usage Examples

### Quick Start Examples

We provide comprehensive examples demonstrating various features of the SDK. Check out the [examples directory](examples/) for:

- [Basic Agent Usage](examples/basic_agent.py) - Creating and using agents with different models
- [Prompt Chaining](examples/prompt_chain.py) - Complex reasoning with multi-step prompts
- [Task Running](examples/task_runner.py) - Orchestrating complex workflows
- [MCP Workflows](examples/mcp_workflow.py) - Model Composition Protocol examples
- [Usage Tracking](examples/usage_tracking.py) - Monitoring model usage and costs

Each example includes detailed comments and demonstrates best practices. See the [examples README](examples/README.md) for more details.

### Basic PEFT Fine-Tuning
```python
from multimind.fine_tuning import UniPELTPlusTuner
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc")
tuner = UniPELTPlusTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    available_methods=["lora", "adapter"],
    model_type="causal_lm"
)
tuner.train(
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)
```

### Few-Shot Learning (MAML)
```python
from multimind.fine_tuning import FewShotMetaTuner
from datasets import load_dataset

dataset = load_dataset("tweet_eval", "sentiment")
tuner = FewShotMetaTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    tasks=["sentiment", "emotion"],
    available_methods=["lora"],
    few_shot_strategy="maml"
)
tuner.train(
    train_datasets={"sentiment": dataset["train"]},
    few_shot_tasks=["sentiment"]
)
```

### Framework Integration (LangChain)
```python
from multimind.integrations import create_adapter
from langchain.llms import LLMChain
from langchain.prompts import PromptTemplate

adapter = create_adapter(
    framework="langchain",
    model_path="./output/fine_tuned_model"
)
prompt = PromptTemplate(input_variables=["text"], template="Analyze sentiment: {text}")
chain = LLMChain(llm=adapter, prompt=prompt)
result = chain.run("I love this product!")
print(result)
```

---

## 📚 Documentation

- [Full Documentation](https://multimind-sdk.readthedocs.io/)
- [API Reference](https://multimind-sdk.readthedocs.io/en/latest/api.html)
- [Architecture Overview](docs/architecture.md)
- [Development Guide](docs/development.md)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.
- [Report bugs or request features](https://github.com/multimind-dev/multimind-sdk/issues)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📣 About

Your SDK solves all of this. One interface. Unified logic. Local + hosted models. Fine-tuning. Agent tools. No switching headaches.

[www.multimind.dev](https://www.multimind.dev)

---

## 🖥️ Command-Line Interface

MultiMind SDK provides a powerful CLI for training, evaluation, inference, and model management.

### Basic Usage

```bash
multimind train --config configs/train_config.yaml
# or (alias)
multimind finetune --config configs/train_config.yaml
```

### Example Config (YAML or JSON)

<details>
<summary>train_config.yaml</summary>

```yaml
base_model_name: "bert-base-uncased"
output_dir: "./output"
methods:
  - lora
  - adapter
method_configs:
  lora:
    r: 8
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]
    lora_dropout: 0.05
    bias: "none"
  adapter:
    adapter_type: "houlsby"
    adapter_size: 64
    adapter_non_linearity: "relu"
    adapter_dropout: 0.1
    target_modules: ["k_proj", "o_proj"]
training_args:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 1e-3
  fp16: true
  logging_steps: 10
  save_strategy: "epoch"
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
train_dataset: "path/to/train_dataset.json"
eval_dataset: "path/to/eval_dataset.json"
model_type: "causal_lm"
```
</details>

- You can use either YAML or JSON for the config file.
- Both `multimind train` and `multimind finetune` work identically.
- The CLI will prompt for missing arguments interactively.

### More CLI Commands

- `multimind evaluate --model ./output/model --dataset data.json`
- `multimind infer --model ./output/model --input "What is PEFT?"`
- `multimind list-models`
- `multimind download --model bert-base-uncased`
- `multimind export --model ./output/model --format onnx --output ./output/model.onnx`
- `multimind delete --model ./output/model`
- `multimind config --set default_dir ./models`
- `multimind info`
- `multimind completion bash`

See `multimind --help` for all options.

---

## 🗂️ Project Structure (Current & Planned)

Below is the planned modular structure for MultiMind SDK. **Modules marked as [implemented] are available now; others are planned for future releases.**

```
multimind-sdk/
├── multimind/
│   ├── __init__.py
│   ├── config.py                   # [implemented] Central config loader
│
│   ├── models/                    # [implemented] Unified model wrappers
│   │   ├── base.py                # [implemented]
│   │   ├── openai.py              # [implemented]
│   │   ├── claude.py              # [planned]
│   │   ├── mistral.py             # [planned]
│   │   ├── huggingface.py         # [implemented]
│   │   └── ollama.py              # [implemented]
│
│   ├── router/                    # [implemented] Fallback, latency-aware, cost-aware logic
│   │   ├── strategy.py            # [implemented]
│   │   ├── fallback.py            # [implemented]
│   │   └── router.py              # [implemented]
│
│   ├── rag/                       # [implemented] Retrieval-Augmented Generation support
│   │   ├── base.py                # [implemented]
│   │   ├── embedder.py            # [implemented]
│   │   └── vector_store.py        # [implemented]
│
│   ├── fine_tuning/              # [implemented] Training logic
│   │   ├── lora_trainer.py        # [implemented]
│   │   ├── qlora_trainer.py       # [implemented]
│   │   └── dataset_loader.py      # [implemented]
│
│   ├── agents/                   # [planned] Agent abstraction
│   │   ├── agent.py               # [planned]
│   │   ├── memory.py              # [planned]
│   │   ├── agent_loader.py        # [planned]
│   │   └── tools/
│   │       ├── web_search.py      # [planned]
│   │       ├── calculator.py      # [planned]
│   │       └── file_reader.py     # [planned]
│
│   ├── orchestration/           # [planned] Prompt chaining and flow control
│   │   ├── prompt_chain.py        # [planned]
│   │   └── task_runner.py         # [planned]
│
│   ├── mcp/                     # [planned] Model Composition Protocol
│   │   ├── parser.py             # [planned]
│   │   ├── executor.py           # [planned]
│   │   └── schema.json           # [planned]
│
│   ├── integrations/           # [implemented] Compatibility with LangChain, CrewAI, etc.
│   │   ├── langchain_adapter.py   # [implemented]
│   │   ├── crewai_adapter.py      # [implemented]
│   │   └── lite_llm.py           # [implemented]
│
│   ├── logging/
│   │   ├── trace_logger.py        # [planned]
│   │   └── usage_tracker.py       # [planned]
│
│   ├── cli/
│   │   ├── main.py                # [implemented] multimind CLI entry
│   │   └── commands/
│   │       ├── agent.py           # [planned]
│   │       ├── run_mcp.py         # [planned]
│   │       └── finetune.py        # [planned]
│
├── examples/
├── configs/                    # [implemented] Sample config files (e.g., train_config.yaml)
├── tests/                      # [implemented]
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.py
```

> **Note:** This structure is forward-looking. All [implemented] modules are available now; [planned] modules will be added in future releases. Current functionality is not impacted.
