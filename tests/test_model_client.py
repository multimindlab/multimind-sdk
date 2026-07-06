import pytest  # noqa: E402

pytest.importorskip("torch", reason="requires multimind-sdk[finetune]")

import pytest
import torch
import torch.nn as nn

from multimind.client.model_client import (
    GRUModelClient,
    HyenaClient,
    LSTMModelClient,
    MoEModelClient,
    RNNModelClient,
    S4Client,
    SpaCyClient,
)


class DummyTokenizer:
    def encode(self, text, return_tensors=None):
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids, skip_special_tokens=True):
        return "dummy decoded"


# Defined at module scope so torch.save/pickle can resolve it by qualified
# name when tests reload the saved model. (Local classes inside test
# functions can't be pickled, which is why these tests used to be skipped.)
_VOCAB_SIZE = 16


class DummyRecurrentModel(nn.Module):
    """Minimal recurrent stand-in for LSTM/RNN/GRU model clients.

    The real clients call ``output.argmax(dim=-1)[0, -1].item()``, so the
    forward pass needs to return logits with shape ``[batch, seq_len, vocab]``.
    """

    def forward(self, x, hidden=None):
        batch, seq_len = x.shape
        return torch.randn(batch, seq_len, _VOCAB_SIZE), None


tokenizer = DummyTokenizer()


def _save_model(tmp_path) -> str:
    """Persist a DummyRecurrentModel to a temp file and return the path."""
    path = tmp_path / "model.pt"
    torch.save(DummyRecurrentModel(), str(path))
    return str(path)


def test_lstm_model_client(tmp_path):
    path = _save_model(tmp_path)
    client = LSTMModelClient(path, tokenizer)
    out = client.generate("hello")
    assert isinstance(out, (torch.Tensor, str))


def test_rnn_model_client(tmp_path):
    path = _save_model(tmp_path)
    client = RNNModelClient(path, tokenizer)
    out = client.generate("hello")
    assert isinstance(out, (torch.Tensor, str))


def test_gru_model_client(tmp_path):
    path = _save_model(tmp_path)
    client = GRUModelClient(path, tokenizer)
    out = client.generate("hello")
    assert isinstance(out, (torch.Tensor, str))

def test_spacy_client():
    try:
        import spacy
        nlp = spacy.blank("en")
        client = SpaCyClient(nlp)
        out = client.generate("Apple is looking at buying U.K. startup for $1 billion")
        assert isinstance(out, list)
    except ImportError:
        pytest.skip("spaCy not installed")

def test_s4_client():
    class DummyS4:
        def generate(self, input_ids):
            return input_ids
    client = S4Client(DummyS4(), tokenizer)
    out = client.generate("test")
    assert isinstance(out, torch.Tensor) or isinstance(out, str)

def test_hyena_client():
    class DummyHyena:
        def generate(self, input_ids):
            return input_ids
    client = HyenaClient(DummyHyena(), tokenizer)
    out = client.generate("test")
    assert isinstance(out, torch.Tensor) or isinstance(out, str)

def test_moe_model_client():
    class DummyClient:
        def generate(self, prompt, **kwargs):
            return "dummy"
    client = MoEModelClient({"a": DummyClient(), "b": DummyClient()}, lambda p: "a")
    out = client.generate("test")
    assert out == "dummy" 