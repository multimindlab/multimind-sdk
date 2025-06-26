# Example: Fine-tuning a real S4 model (HuggingFace) with PyTorch
# Requires: pip install transformers mamba-ssm torch
# See: https://huggingface.co/state-spaces/s4-small

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

MODEL_NAME = "state-spaces/s4-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.train()

# Toy dataset (for demonstration)
class ToyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.examples = [tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt") for text in texts]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.examples[idx].items()}

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Structured State Space Models are efficient for long sequences.",
    "S4 models can handle long-range dependencies.",
    "PyTorch is great for deep learning research."
]
dataset = ToyTextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Simple training loop (for demonstration)
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

print("[INFO] Fine-tuning complete. Save or evaluate your model as needed.")

# For real tasks, use a larger dataset, more epochs, and advanced strategies (PEFT, adapters, hyperparameter search, etc.) 