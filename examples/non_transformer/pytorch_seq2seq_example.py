from multimind.llm.non_transformer_llm import NonTransformerLLM
import torch
import torch.nn as nn
import asyncio

# Toy vocabulary and tokenizer
vocab = ["<pad>", "I", "love", "AI", "and", "Python", "."]
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

def encode(text):
    return [word2idx.get(w, 0) for w in text.split()]

def decode(indices):
    return " ".join([idx2word.get(i, "<unk>") for i in indices])

class SimpleTokenizer:
    def encode(self, text):
        return encode(text)
    def decode(self, indices):
        return decode(indices)

tokenizer = SimpleTokenizer()

# Minimal Seq2Seq model
def make_seq2seq_model(vocab_size, embedding_dim, hidden_dim, output_dim):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        def forward(self, x):
            embedded = self.embedding(x)
            outputs, hidden = self.rnn(embedded)
            return hidden
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        def forward(self, x, hidden):
            embedded = self.embedding(x)
            output, hidden = self.rnn(embedded, hidden)
            logits = self.fc(output[:, -1, :])
            return logits, hidden
    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
        def forward(self, src, tgt):
            hidden = self.encoder(src)
            logits, _ = self.decoder(tgt, hidden)
            return logits
    return Seq2Seq()

embedding_dim = 8
hidden_dim = 16
output_dim = vocab_size
model = make_seq2seq_model(vocab_size, embedding_dim, hidden_dim, output_dim)

class PyTorchSeq2SeqLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, tokenizer, device="cpu", **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    async def generate(self, prompt: str, max_tokens: int = 5, temperature: float = 1.0, **kwargs) -> str:
        src = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        tgt = src[:, :1]  # Start with first token
        generated = tgt
        hidden = self.model.encoder(src)
        for _ in range(max_tokens):
            with torch.no_grad():
                logits, hidden = self.model.decoder(generated, hidden)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        output_tokens = generated[0].tolist()
        return self.tokenizer.decode(output_tokens)

llm = PyTorchSeq2SeqLLM(
    model_name="simple_seq2seq",
    model_instance=model,
    tokenizer=tokenizer,
    device="cpu"
)

async def main():
    prompt = "I love"
    result = await llm.generate(prompt, max_tokens=5)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 