# MultiMind SDK

MultiMind SDK is a powerful and flexible library for fine-tuning and adapting large language models using Parameter-Efficient Fine-Tuning (PEFT) methods. It provides advanced features for meta-learning, few-shot learning, and transfer learning, making it easier to adapt models to new tasks with minimal data.

## Features

### Core PEFT Methods
- **UniPELT++**: Unified framework for multiple PEFT methods
  - LoRA (Low-Rank Adaptation)
  - Adapters
  - Prefix Tuning
  - Prompt Tuning
  - IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

### Advanced Fine-Tuning
- **Adaptive Methods**
  - Resource-aware adaptation
  - Performance-based method selection
  - Dynamic component weighting
  - Model-agnostic implementation

### Meta-Learning
- **Few-Shot Learning Strategies**
  - Prototype-based learning
  - MAML (Model-Agnostic Meta-Learning)
  - Reptile
  - Support for n-way k-shot learning

- **Transfer Learning**
  - Layer-wise similarity analysis
  - Automatic layer selection
  - Frozen and fine-tuning strategies
  - Multi-task transfer support

### Multi-Task Learning
- Task-specific method selection
- Dynamic task weighting
- Cross-task knowledge distillation
- Hyperparameter optimization

## Installation

```bash
# Basic installation
pip install multimind-sdk

# With development dependencies
pip install multimind-sdk[dev]

# With specific framework support
pip install multimind-sdk[langchain,lite-llm,superagi]
```

## Quick Start

### Basic PEFT Fine-Tuning
```python
from multimind.fine_tuning import UniPELTPlusTuner
from datasets import load_dataset

# Load dataset
dataset = load_dataset("glue", "mrpc")

# Initialize tuner
tuner = UniPELTPlusTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    available_methods=["lora", "adapter"],
    model_type="causal_lm",
    peft_config={
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1
        },
        "adapter": {
            "reduction_factor": 16,
            "non_linearity": "relu"
        }
    }
)

# Train model
tuner.train(
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    training_args={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-4,
        "warmup_steps": 100
    }
)
```

### Few-Shot Learning with MAML
```python
from multimind.fine_tuning import FewShotMetaTuner
from datasets import load_dataset

# Load few-shot dataset
dataset = load_dataset("tweet_eval", "sentiment")

# Initialize tuner with MAML strategy
tuner = FewShotMetaTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    tasks=["sentiment", "emotion"],
    available_methods=["lora"],
    few_shot_strategy="maml",
    few_shot_config={
        "n_way": 5,
        "k_shot": 5,
        "n_query": 5,
        "n_episodes": 100,
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "adaptation_steps": 5,
        "first_order": True
    }
)

# Train with few-shot learning
tuner.train(
    train_datasets={
        "sentiment": dataset["train"],
        "emotion": emotion_dataset["train"]
    },
    few_shot_tasks=["sentiment", "emotion"],
    meta_training_args={
        "meta_batch_size": 4,
        "meta_epochs": 10,
        "eval_episodes": 20
    }
)

# Adapt to new task
new_task_dataset = load_dataset("tweet_eval", "hate")
adapted_model = tuner.adapt_to_new_task(
    new_task_dataset["train"],
    n_way=3,
    k_shot=5,
    adaptation_steps=10
)
```

### Transfer Learning
```python
from multimind.fine_tuning import TransferMetaTuner
from datasets import load_dataset

# Load source and target datasets
source_dataset = load_dataset("conll2003", "ner")
target_dataset = load_dataset("wnut17", "ner")

# Initialize tuner
tuner = TransferMetaTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    tasks=["conll_ner", "wnut_ner"],
    available_methods=["lora", "adapter"],
    transfer_config={
        "transfer_strategy": "frozen",
        "layer_selection": "auto",
        "similarity_threshold": 0.7,
        "transfer_layers": ["bert.encoder.layer.10", "bert.encoder.layer.11"]
    }
)

# Train with transfer learning
tuner.train(
    train_datasets={
        "conll_ner": source_dataset["train"],
        "wnut_ner": target_dataset["train"]
    },
    source_tasks=["conll_ner"],
    transfer_training_args={
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.1
    }
)

# Evaluate on target task
metrics = tuner.evaluate(
    eval_dataset=target_dataset["test"],
    task="wnut_ner"
)
print(f"Target task F1 score: {metrics['f1']:.3f}")
```

### Multi-Task Learning
```python
from multimind.fine_tuning import MultiTaskTuner
from datasets import load_dataset

# Load multiple datasets
sentiment_dataset = load_dataset("tweet_eval", "sentiment")
ner_dataset = load_dataset("conll2003", "ner")
qa_dataset = load_dataset("squad")

# Initialize tuner
tuner = MultiTaskTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    tasks={
        "sentiment": {
            "type": "classification",
            "num_labels": 3
        },
        "ner": {
            "type": "token_classification",
            "num_labels": 9
        },
        "qa": {
            "type": "question_answering"
        }
    },
    available_methods=["lora", "adapter"],
    task_weights={
        "sentiment": 1.0,
        "ner": 1.5,
        "qa": 2.0
    }
)

# Train on multiple tasks
tuner.train(
    train_datasets={
        "sentiment": sentiment_dataset["train"],
        "ner": ner_dataset["train"],
        "qa": qa_dataset["train"]
    },
    eval_datasets={
        "sentiment": sentiment_dataset["validation"],
        "ner": ner_dataset["validation"],
        "qa": qa_dataset["validation"]
    },
    training_args={
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "task_specific_optimization": True
    }
)
```

### Framework Integration
```python
from multimind.integrations import create_adapter
from langchain.llms import LLMChain
from langchain.prompts import PromptTemplate

# Create LangChain adapter
adapter = create_adapter(
    framework="langchain",
    model_path="./output/fine_tuned_model",
    config={
        "temperature": 0.7,
        "max_length": 100,
        "device": "cuda"
    }
)

# Use in LangChain
prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze sentiment: {text}"
)
chain = LLMChain(llm=adapter, prompt=prompt)

# Generate text
result = chain.run("I love this product!")
print(f"Sentiment: {result}")

# Use with CrewAI
from crewai import Agent, Task, Crew

agent = Agent(
    role="Researcher",
    goal="Research AI trends",
    backstory="Expert in AI research",
    llm=adapter
)

task = Task(
    description="Analyze recent AI developments",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task]
)
result = crew.kickoff()
```

## Advanced Usage

### Custom PEFT Method
```python
from multimind.fine_tuning.base import BasePEFTMethod
import torch.nn as nn

class CustomPEFTMethod(BasePEFTMethod):
    def __init__(self, config: dict):
        super().__init__(config)
        self.reduction_factor = config.get("reduction_factor", 16)
        
    def apply(self, model: nn.Module) -> nn.Module:
        # Implement custom PEFT method
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Add custom adaptation
                pass
        return model

# Use custom method
tuner = UniPELTPlusTuner(
    base_model_name="bert-base-uncased",
    output_dir="./output",
    available_methods=["custom"],
    custom_methods={"custom": CustomPEFTMethod}
)
```

### Hyperparameter Optimization
```python
from multimind.fine_tuning import MetaOptimizedMultiTaskTuner
import optuna

def objective(trial):
    # Define hyperparameter search space
    config = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", 4, 32),
        "lora_r": trial.suggest_int("lora_r", 4, 32),
        "adapter_reduction": trial.suggest_int("adapter_reduction", 8, 64)
    }
    
    # Initialize and train tuner
    tuner = MetaOptimizedMultiTaskTuner(
        base_model_name="bert-base-uncased",
        output_dir=f"./output/trial_{trial.number}",
        tasks=tasks,
        peft_config=config
    )
    
    # Train and evaluate
    metrics = tuner.train_and_evaluate(
        train_datasets=train_datasets,
        eval_datasets=eval_datasets
    )
    
    return metrics["average_f1"]

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Get best parameters
best_params = study.best_params
print(f"Best parameters: {best_params}")
```

## Documentation

For detailed documentation, visit our [documentation site](https://multimind-sdk.readthedocs.io/).

Key documentation sections:
- [Architecture Overview](docs/architecture.md)
- [Development Guide](docs/development.md)
- [API Reference](https://multimind-sdk.readthedocs.io/en/latest/api.html)
- [Examples](https://multimind-sdk.readthedocs.io/en/latest/examples.html)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MultiMind SDK in your research, please cite:

```bibtex
@software{multimind_sdk,
  author = {MultiMind Team},
  title = {MultiMind SDK: Advanced PEFT and Meta-Learning for Language Models},
  year = {2024},
  url = {https://github.com/yourusername/multimind-sdk}
}
```

## Acknowledgments

- Hugging Face Transformers library
- PEFT research community
- Meta-learning research community

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/multimind-sdk/issues)
- Email: support@multimind.dev
- Discord: [Join our community](https://discord.gg/your-invite-link)
