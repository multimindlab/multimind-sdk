# Example: Hyperparameter search for Mamba PEFT/adapter fine-tuning using Optuna
# Requires: pip install transformers mamba-ssm torch optuna
# See: https://huggingface.co/state-spaces/mamba-130m-hf

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import optuna

MODEL_NAME = "state-spaces/mamba-130m-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# Toy dataset
class ToyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.examples = [tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt") for text in texts]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.examples[idx].items()}

texts = [
    "Adapters enable parameter-efficient fine-tuning.",
    "Mamba models can be adapted with small modules.",
    "PEFT is useful for large language models.",
    "PyTorch makes it easy to experiment."
]
dataset = ToyTextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

def objective(trial):
    adapter_dim = trial.suggest_categorical("adapter_dim", [8, 16, 32, 64])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    hidden_dim = model.config.hidden_size
    adapter = nn.Sequential(
        nn.Linear(hidden_dim, adapter_dim),
        nn.ReLU(),
        nn.Linear(adapter_dim, hidden_dim)
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True
    original_forward = model.forward
    def forward_with_adapter(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        logits = outputs.logits
        adapted_logits = adapter(logits)
        return outputs.__class__(logits=adapted_logits, **{k: v for k, v in outputs.items() if k != 'logits'})
    model.forward = forward_with_adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    adapter.to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("[INFO] Best hyperparameters:")
print(study.best_params)

# For real use, extend to more epochs, larger datasets, and more parameters 