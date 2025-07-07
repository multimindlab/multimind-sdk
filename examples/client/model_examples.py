# Example model classes for demonstration purposes only
import torch
import torch.nn as nn
from multimind.client.model_client import ModelClient

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

class DiffusionTextClient(ModelClient):
    def __init__(self, model):
        self.model = model  # e.g., diffuSeq or similar
    def generate(self, prompt: str, **kwargs):
        return self.model.sample(prompt)

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