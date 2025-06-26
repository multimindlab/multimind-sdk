import torch
import torch.nn as nn
from multimind.client.model_client import (
    LSTMModelClient, MoEModelClient, MambaClient, DiffusionTextClient, RWKVClient, SpaCyClient, S4Client, HyenaClient
)

# Dummy tokenizer for demonstration
class DummyTokenizer:
    def encode(self, text, return_tensors=None):
        return torch.tensor([[1, 2, 3]])
    def decode(self, ids, skip_special_tokens=True):
        return "dummy decoded"

tokenizer = DummyTokenizer()

# Dummy LSTM model for demonstration
class DummyLSTM(nn.Module):
    def forward(self, x, hidden=None):
        return torch.randn_like(x, dtype=torch.float), None

dummy_lstm = DummyLSTM()
# Save and load for LSTMModelClient demo (in real use, save a real model)
torch.save(dummy_lstm, "dummy_lstm.pt")

# --- LSTMModelClient ---
lstm_client = LSTMModelClient("dummy_lstm.pt", tokenizer)
print("LSTMModelClient:", lstm_client.generate("hello world"))

# --- MoEModelClient ---
def router_fn(prompt):
    return "lstm" if "lstm" in prompt else "dummy"
moe_client = MoEModelClient({"lstm": lstm_client, "dummy": lstm_client}, router_fn)
print("MoEModelClient (lstm):", moe_client.generate("use lstm"))

# --- MambaClient (stub, requires real Mamba) ---
try:
    mamba_client = MambaClient(config_path="dummy_config.yaml")
    print("MambaClient:", mamba_client.generate("test prompt"))
except Exception as e:
    print("MambaClient: [stub/demo]", e)

# --- DiffusionTextClient (stub) ---
class DummyDiffusion:
    def sample(self, prompt):
        return f"diffusion output for: {prompt}"
diff_client = DiffusionTextClient(DummyDiffusion())
print("DiffusionTextClient:", diff_client.generate("generate text"))

# --- RWKVClient (stub, requires real RWKV) ---
try:
    rwkv_client = RWKVClient(model_path="dummy_rwkv.pth")
    print("RWKVClient:", rwkv_client.generate("test prompt"))
except Exception as e:
    print("RWKVClient: [stub/demo]", e)

# --- SpaCyClient ---
try:
    import spacy
    nlp = spacy.blank("en")
    spacy_client = SpaCyClient(nlp)
    print("SpaCyClient:", spacy_client.generate("Apple is looking at buying U.K. startup for $1 billion"))
except ImportError:
    print("SpaCyClient: spaCy not installed")

# --- S4Client (stub) ---
class DummyS4:
    def generate(self, input_ids):
        return input_ids
s4_client = S4Client(DummyS4(), tokenizer)
print("S4Client:", s4_client.generate("test prompt"))

# --- HyenaClient (stub) ---
class DummyHyena:
    def generate(self, input_ids):
        return input_ids
hyena_client = HyenaClient(DummyHyena(), tokenizer)
print("HyenaClient:", hyena_client.generate("test prompt")) 