# Example: Hyperparameter search for Hyena PEFT/adapter fine-tuning using Optuna
# Requires: pip install torch optuna
# For real Hyena, see: https://github.com/HazyResearch/hyena

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import optuna

# Dummy Hyena model for demonstration (replace with real model for production)
class DummyHyenaModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
    def forward(self, input_ids):
        # Simulate output logits
        batch, seq = input_ids.shape
        return torch.randn(batch, seq, self.hidden_size)

# Dummy tokenizer
class DummyTokenizer:
    def __call__(self, text, truncation=True, padding="max_length", max_length=32, return_tensors=None):
        # Simulate tokenization
        return {"input_ids": torch.randint(0, 100, (1, max_length))}

model = DummyHyenaModel(hidden_size=64)
tokenizer = DummyTokenizer()

# Toy dataset
class ToyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.examples = [tokenizer(text, max_length=max_length) for text in texts]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.examples[idx].items()}

texts = [
    "Adapters enable parameter-efficient fine-tuning.",
    "Hyena models can be adapted with small modules.",
    "PEFT is useful for large language models.",
    "PyTorch makes it easy to experiment."
]
dataset = ToyTextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

def objective(trial):
    adapter_dim = trial.suggest_categorical("adapter_dim", [8, 16, 32, 64])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    hidden_dim = model.hidden_size
    adapter = nn.Sequential(
        nn.Linear(hidden_dim, adapter_dim),
        nn.ReLU(),
        nn.Linear(adapter_dim, hidden_dim)
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    adapter.to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        # Simulate logits
        logits = model(input_ids)
        adapted_logits = adapter(logits)
        # Simulate loss (MSE to zeros)
        loss = ((adapted_logits - 0) ** 2).mean()
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

# For real use, replace DummyHyenaModel with the real Hyena model and tokenizer 