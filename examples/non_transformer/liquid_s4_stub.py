from multimind.llm.non_transformer_llm import LiquidS4LLM
import asyncio

# --- Liquid-S4 Model Integration Stub ---
# To use this stub:
# 1. Install Liquid-S4 from official repo (see: https://github.com/HazyResearch/state-spaces)
# 2. Load your Liquid-S4 model and tokenizer as per their documentation.
# 3. Implement the generate method in a subclass or pass a compatible model instance.

# Dummy model for demonstration
class DummyLiquidS4Model:
    def generate(self, input_ids, **kwargs):
        return input_ids + [111]
class DummyTokenizer:
    def encode(self, text): return [21, 22, 23]
    def decode(self, ids): return "liquid-s4 output"

dummy_model = DummyLiquidS4Model()
dummy_tokenizer = DummyTokenizer()

class DemoLiquidS4LLM(LiquidS4LLM):
    def __init__(self, model_name, model_instance, tokenizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer
    async def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output_ids)

llm = DemoLiquidS4LLM("demo_liquid_s4", dummy_model, dummy_tokenizer)

async def main():
    prompt = "Hello Liquid-S4"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 