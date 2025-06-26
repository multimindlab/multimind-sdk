# Example: LoRA/PEFT fine-tuning for S4 using HuggingFace PEFT library
# Requires: pip install transformers mamba-ssm torch peft
# See: https://huggingface.co/state-spaces/s4-small and https://github.com/huggingface/peft

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "state-spaces/s4-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Apply LoRA using PEFT
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # adjust for S4 arch
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# Toy dataset
class ToyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.examples = [tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt") for text in texts]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.examples[idx].items()}

texts = [
    "LoRA enables efficient fine-tuning.",
    "S4 models can be adapted with LoRA.",
    "PEFT is useful for large language models.",
    "PyTorch makes it easy to experiment."
]
dataset = ToyTextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop (LoRA params are trainable)
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

# Save LoRA adapter
model.save_pretrained("./s4_lora_adapter")
print("[INFO] LoRA fine-tuning complete. Adapter saved to ./s4_lora_adapter")

# For production: use a real dataset, more epochs, and advanced evaluation/checkpointing 