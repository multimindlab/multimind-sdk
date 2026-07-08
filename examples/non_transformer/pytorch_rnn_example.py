from multimind.llm.non_transformer_llm import CustomRNNLLM
import torch
import torch.nn as nn
import asyncio

# Toy vocabulary and tokenizer
vocab = ["<pad>", "I", "love", "AI", "and", "Python", "."]
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

class SimpleTokenizer:
    # Matches the interface CustomRNNLLM expects (HF-style encode/decode)
    def encode(self, text, return_tensors=None):
        ids = [word2idx.get(w, 0) for w in text.split()]
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, indices, skip_special_tokens=True):
        return " ".join(idx2word.get(int(i), "<unk>") for i in indices)

class SimpleRNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def generate(self, input_ids, max_length=16, temperature=1.0):
        ids = input_ids
        hidden = None
        for _ in range(max_length - ids.size(1)):
            emb = self.embedding(ids)
            out, hidden = self.rnn(emb)
            next_id = self.fc(out[:, -1]).argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

tokenizer = SimpleTokenizer()
model = SimpleRNNTextGenerator(vocab_size, embedding_dim=8, hidden_dim=16)

# For demonstration, random weights (no training)
llm = CustomRNNLLM(
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
