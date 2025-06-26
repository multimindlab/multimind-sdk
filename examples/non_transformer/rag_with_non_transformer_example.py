import torch
from multimind.client.model_client import LSTMModelClient, SpaCyClient

# Dummy retriever for demonstration
class DummyRetriever:
    def retrieve(self, query):
        return ["retrieved context 1", "retrieved context 2"]

def simple_rag_pipeline(query, retriever, generator):
    # Retrieve context
    contexts = retriever.retrieve(query)
    # Concatenate context and query
    full_prompt = "\n".join(contexts + [query])
    # Generate answer
    return generator.generate(full_prompt)

# Dummy tokenizer and model for LSTM
class DummyTokenizer:
    def encode(self, text, return_tensors=None):
        return torch.tensor([[1, 2, 3]])
    def decode(self, ids, skip_special_tokens=True):
        return "dummy decoded"

tokenizer = DummyTokenizer()
class DummyLSTM(torch.nn.Module):
    def forward(self, x, hidden=None):
        return torch.randn_like(x, dtype=torch.float), None

torch.save(DummyLSTM(), "dummy_lstm.pt")
lstm_client = LSTMModelClient("dummy_lstm.pt", tokenizer)
retriever = DummyRetriever()

query = "What is the capital of France?"
rag_response = simple_rag_pipeline(query, retriever, lstm_client)
print("RAG with LSTMModelClient:", rag_response)

# Example with SpaCyClient
try:
    import spacy
    nlp = spacy.blank("en")
    spacy_client = SpaCyClient(nlp)
    rag_response_spacy = simple_rag_pipeline(query, retriever, spacy_client)
    print("RAG with SpaCyClient:", rag_response_spacy)
except ImportError:
    print("spaCy not installed, skipping SpaCyClient RAG example.") 