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
