import asyncio

from multimind.llm.non_transformer_llm import NonTransformerLLM

# --- Hyena Model Integration Stub ---
# There is no built-in HyenaLLM class: NonTransformerLLM is an extension scaffold.
# To integrate Hyena:
# 1. Install Hyena from the official repo (see: https://github.com/HazyResearch/hyena)
# 2. Load your Hyena model and tokenizer as per their documentation.
# 3. Subclass NonTransformerLLM and implement generate (as below) with real inference.

# For demonstration, we use a dummy model
class DummyHyenaModel:
    def generate(self, input_ids, **kwargs):
        return input_ids + [99]

class DummyTokenizer:
    def encode(self, text): return [4, 5, 6]
    def decode(self, ids): return "hyena output"

dummy_model = DummyHyenaModel()
dummy_tokenizer = DummyTokenizer()

class DemoHyenaLLM(NonTransformerLLM):
    def __init__(self, model_name, model_instance, tokenizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer

    async def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output_ids)

llm = DemoHyenaLLM("demo_hyena", dummy_model, dummy_tokenizer)

async def main():
    prompt = "Hello Hyena"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main())
