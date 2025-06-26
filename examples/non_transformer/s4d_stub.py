from multimind.llm.non_transformer_llm import S4DLLM
import asyncio

# --- S4D Model Integration Stub ---
# To use this stub:
# 1. Install S4D from official repo (see: https://github.com/HazyResearch/state-spaces)
# 2. Load your S4D model and tokenizer as per their documentation.
# 3. Implement the generate method in a subclass or pass a compatible model instance.

# Dummy model for demonstration
class DummyS4DModel:
    def generate(self, input_ids, **kwargs):
        return input_ids + [222]
class DummyTokenizer:
    def encode(self, text): return [31, 32, 33]
    def decode(self, ids): return "s4d output"

dummy_model = DummyS4DModel()
dummy_tokenizer = DummyTokenizer()

class DemoS4DLLM(S4DLLM):
    def __init__(self, model_name, model_instance, tokenizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer
    async def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output_ids)

llm = DemoS4DLLM("demo_s4d", dummy_model, dummy_tokenizer)

async def main():
    prompt = "Hello S4D"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 