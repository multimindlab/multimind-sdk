# Example: Advanced evaluation for a fine-tuned S4 model (perplexity & accuracy)
# Requires: pip install transformers mamba-ssm torch peft
# See: https://huggingface.co/state-spaces/s4-small

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import math

MODEL_NAME = "state-spaces/s4-small"
ADAPTER_PATH = "./s4_lora_adapter"  # Path to LoRA adapter (if used)

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Optionally load LoRA/PEFT adapter
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("[INFO] Loaded LoRA adapter from", ADAPTER_PATH)
except Exception:
    print("[INFO] No LoRA adapter found, evaluating base model.")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Toy validation set
class ToyTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.examples = [tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt") for text in texts]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.examples[idx].items()}

val_texts = [
    "LoRA enables efficient fine-tuning.",
    "S4 models can be adapted with LoRA.",
    "PEFT is useful for large language models.",
    "PyTorch makes it easy to experiment."
]
val_dataset = ToyTextDataset(val_texts, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=2)

# Evaluation loop: compute perplexity and accuracy
all_losses = []
all_accs = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        # Perplexity
        all_losses.append(loss.item())
        # Accuracy (next-token prediction)
        preds = logits.argmax(-1)
        acc = (preds[:, :-1] == labels[:, 1:]).float().mean().item()
        all_accs.append(acc)

mean_loss = sum(all_losses) / len(all_losses)
perplexity = math.exp(mean_loss)
mean_acc = sum(all_accs) / len(all_accs)

print(f"Validation Perplexity: {perplexity:.2f}")
print(f"Validation Next-Token Accuracy: {mean_acc:.2%}")

# For real use, evaluate on a larger validation/test set and use more metrics as needed 