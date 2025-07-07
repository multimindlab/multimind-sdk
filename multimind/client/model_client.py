import torch
import torch.nn as nn
from typing import Any, Dict, Callable
import time

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

# --- RNN Example ---
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.linear(out)
        return out, hidden

class RNNModelClient(ModelClient):
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

# --- GRU Example ---
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.linear(out)
        return out, hidden

class GRUModelClient(ModelClient):
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

# --- Dynamic MoE Model Client ---
class DynamicMoEModelClient(MoEModelClient):
    """
    MoE client that routes based on runtime metrics (latency, input length, etc.).
    Keeps a history of model latencies and can auto-switch based on input features.
    """
    def __init__(self, expert_clients: Dict[str, ModelClient], router_fn: Callable[[str, dict], str]):
        super().__init__(expert_clients, None)
        self.router_fn = router_fn  # router_fn(prompt, metrics) -> expert key
        self.metrics = {k: {"latency": [], "count": 0} for k in expert_clients}

    def generate(self, prompt: str, **kwargs):
        # Gather input features
        input_length = len(prompt)
        # Compute average latency for each expert
        avg_latencies = {k: (sum(v["latency"]) / len(v["latency"]) if v["latency"] else float('inf')) for k, v in self.metrics.items()}
        # Call router_fn with prompt and metrics
        selected_expert = self.router_fn(prompt, {"input_length": input_length, "avg_latencies": avg_latencies})
        client = self.expert_clients[selected_expert]
        start = time.time()
        result = client.generate(prompt, **kwargs)
        elapsed = time.time() - start
        # Track latency
        self.metrics[selected_expert]["latency"].append(elapsed)
        self.metrics[selected_expert]["count"] += 1
        return result

# Example router_fn for DynamicMoEModelClient:
# def router_fn(prompt, metrics):
#     if metrics["input_length"] > 1000:
#         return "rwkv"
#     if metrics["avg_latencies"].get("mamba", float('inf')) < 0.5:
#         return "mamba"
#     return "llama"

# --- Add more custom clients as needed following this template ---

class MultiModalClient(ModelClient):
    """
    Unified client for multimodal input/output. Routes to the correct model client based on input type.
    Supports text, image, audio, video, and code (stubs for non-text).
    """
    def __init__(self, text_client=None, image_client=None, audio_client=None, video_client=None, code_client=None):
        self.text_client = text_client
        self.image_client = image_client
        self.audio_client = audio_client
        self.video_client = video_client
        self.code_client = code_client
    def generate(self, prompt: str, input_type: str = "text", **kwargs):
        if input_type == "text" and self.text_client:
            return self.text_client.generate(prompt, **kwargs)
        elif input_type == "image" and self.image_client:
            return self.image_client.generate(prompt, **kwargs)
        elif input_type == "audio" and self.audio_client:
            return self.audio_client.generate(prompt, **kwargs)
        elif input_type == "video" and self.video_client:
            return self.video_client.generate(prompt, **kwargs)
        elif input_type == "code" and self.code_client:
            return self.code_client.generate(prompt, **kwargs)
        else:
            raise ValueError(f"No client for input_type: {input_type}")

# --- Stubs for image/audio/video/code clients ---
class ImageModelClient(ModelClient):
    """Basic image model client that returns a placeholder image result."""
    def generate(self, prompt: str, **kwargs):
        return f"[ImageModelClient] Placeholder image for prompt: {prompt}"

class AudioModelClient(ModelClient):
    """Basic audio model client that returns a placeholder audio result."""
    def generate(self, prompt: str, **kwargs):
        return f"[AudioModelClient] Placeholder audio for prompt: {prompt}"

class VideoModelClient(ModelClient):
    """Stub for video model client (e.g., Video LLMs)."""
    def generate(self, prompt: str, **kwargs):
        return f"[VideoModelClient] Generated video for prompt: {prompt}"

class CodeModelClient(ModelClient):
    """Basic code model client that returns a placeholder code result."""
    def generate(self, prompt: str, **kwargs):
        return f"[CodeModelClient] Placeholder code for prompt: {prompt}" 