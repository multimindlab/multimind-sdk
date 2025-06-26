import torch
import torch.nn as nn
from typing import Any, Dict, Callable

# --- Base ModelClient ---
class ModelClient:
    """
    Base class for all model clients (transformer and non-transformer).
    Subclass this and implement the generate method for your model.
    """
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("Implement generate for your model client.")

# --- LSTM/GRU Example ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        return out, hidden

class LSTMModelClient(ModelClient):
    def __init__(self, model_path, tokenizer):
        self.model = torch.load(model_path)
        self.model.eval()
        self.tokenizer = tokenizer
    def generate(self, prompt: str, **kwargs) -> str:
        tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output, _ = self.model(tokens)
        next_token = output.argmax(dim=-1)[0, -1].item()
        return self.tokenizer.decode([next_token])

# --- Mixture-of-Experts (MoE) Client ---
class MoEModelClient(ModelClient):
    def __init__(self, expert_clients: Dict[str, ModelClient], router_fn: Callable[[str], str]):
        self.expert_clients = expert_clients  # e.g., {"rnn": LSTMModelClient(), "mamba": MambaClient()}
        self.router_fn = router_fn  # Function to choose expert based on prompt
    def generate(self, prompt: str, **kwargs):
        selected_expert = self.router_fn(prompt)
        client = self.expert_clients[selected_expert]
        return client.generate(prompt, **kwargs)

# --- State Space Model (e.g., Mamba) Client ---
# Note: Requires state-spaces/mamba repo and dependencies
try:
    from state_spaces.mamba import Mamba
except ImportError:
    Mamba = None

class MambaClient(ModelClient):
    def __init__(self, config_path):
        if Mamba is None:
            raise ImportError("state-spaces/mamba is not installed.")
        self.model = Mamba.load_from_config(config_path)
        self.model.eval()
    def generate(self, prompt: str, **kwargs) -> str:
        return self.model.generate(prompt)

# --- Diffusion Text Generator Client ---
class DiffusionTextClient(ModelClient):
    def __init__(self, model):
        self.model = model  # e.g., diffuSeq or similar
    def generate(self, prompt: str, **kwargs):
        return self.model.sample(prompt)

# --- RWKV Model Client ---
try:
    from rwkv.model import RWKV
except ImportError:
    RWKV = None

class RWKVClient(ModelClient):
    def __init__(self, model_path):
        if RWKV is None:
            raise ImportError("rwkv is not installed.")
        self.model = RWKV(model=model_path)
    def generate(self, prompt: str, **kwargs):
        return self.model.generate(prompt)

# --- SpaCy Pipeline Client ---
class SpaCyClient(ModelClient):
    """
    ModelClient for spaCy pipelines (NER, text classification, etc.).
    """
    def __init__(self, nlp):
        self.nlp = nlp
    def generate(self, prompt: str, **kwargs):
        doc = self.nlp(prompt)
        # Example: return named entities
        return [(ent.text, ent.label_) for ent in doc.ents]

# --- S4 Model Client (stub, extend for real S4 integration) ---
class S4Client(ModelClient):
    """
    ModelClient for S4 state-space models. Plug in your real S4 model and tokenizer.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def generate(self, prompt: str, **kwargs):
        # Example: encode, run model, decode (user must implement details)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output_ids = self.model.generate(input_ids)
        return self.tokenizer.decode(output_ids[0])

# --- Hyena Model Client (stub, extend for real Hyena integration) ---
class HyenaClient(ModelClient):
    """
    ModelClient for Hyena sequence models. Plug in your real Hyena model and tokenizer.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def generate(self, prompt: str, **kwargs):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output_ids = self.model.generate(input_ids)
        return self.tokenizer.decode(output_ids[0])

# --- Add more custom clients as needed following this template --- 