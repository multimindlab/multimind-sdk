from multimind.llm.non_transformer_llm import SimpleRNNTextGenerator, PyTorchRNNLLM
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

# Create and initialize the RNN model
embedding_dim = 8
hidden_dim = 16
output_dim = vocab_size
model = SimpleRNNTextGenerator(vocab_size, embedding_dim, hidden_dim, output_dim)

# For demonstration, random weights (no training)

# Wrap with PyTorchRNNLLM
llm = PyTorchRNNLLM(
    model_name="simple_rnn",
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