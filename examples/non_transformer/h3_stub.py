from multimind.llm.non_transformer_llm import H3LLM
import asyncio

# --- H3 Model Integration Stub ---
# To use this stub:
# 1. Install H3 from official repo (see: https://github.com/HazyResearch/hyena)
# 2. Load your H3 model and tokenizer as per their documentation.
# 3. Implement the generate method in a subclass or pass a compatible model instance.

# Dummy model for demonstration
class DummyH3Model:
    def generate(self, input_ids, **kwargs):
        return input_ids + [404]
class DummyTokenizer:
    def encode(self, text): return [10, 11, 12]
    def decode(self, ids): return "h3 output"

dummy_model = DummyH3Model()
dummy_tokenizer = DummyTokenizer()

class DemoH3LLM(H3LLM):
    def __init__(self, model_name, model_instance, tokenizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer
    async def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output_ids)

llm = DemoH3LLM("demo_h3", dummy_model, dummy_tokenizer)

async def main():
    prompt = "Hello H3"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 