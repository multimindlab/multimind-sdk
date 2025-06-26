# Example: PEFT/Adapter fine-tuning for S4 (HuggingFace) with PyTorch
# Requires: pip install transformers mamba-ssm torch
# See: https://huggingface.co/state-spaces/s4-small

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

MODEL_NAME = "state-spaces/s4-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# Simple Adapter module (bottleneck MLP)
class SimpleAdapter(nn.Module):
    def __init__(self, hidden_dim, adapter_dim=32):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_dim)
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

# Insert adapter into S4 model (for demo, after the first layer)
# For real use, insert into all layers or use a PEFT library
hidden_dim = model.config.hidden_size
adapter = SimpleAdapter(hidden_dim)

# Freeze all model parameters except adapter
for param in model.parameters():
    param.requires_grad = False
for param in adapter.parameters():
    param.requires_grad = True

# Monkey-patch adapter into model (for demo)
# For real use, subclass the model and insert adapters properly
original_forward = model.forward

def forward_with_adapter(*args, **kwargs):
    outputs = original_forward(*args, **kwargs)
    # outputs.logits: [batch, seq, hidden] or [batch, seq, vocab]
    # For demo, apply adapter to logits before loss (not correct for real use)
    logits = outputs.logits
    adapted_logits = adapter(logits)
    return outputs.__class__(logits=adapted_logits, **{k: v for k, v in outputs.items() if k != 'logits'})

model.forward = forward_with_adapter

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
    "S4 models can be adapted with small modules.",
    "PEFT is useful for large language models.",
    "PyTorch makes it easy to experiment."
]
dataset = ToyTextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
adapter.to(device)

# Training loop (only adapter is trainable)
model.train()
for epoch in range(2):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("[INFO] Adapter fine-tuning complete. Save or evaluate your adapter as needed.")

# For real use, insert adapters into all layers and use a PEFT library (e.g., peft, LoRA, etc.) 