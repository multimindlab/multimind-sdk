from multimind.llm.non_transformer_llm import GSSLLM
import asyncio

# --- GSS Model Integration Stub ---
# To use this stub:
# 1. Install GSS from official repo (see: https://github.com/HazyResearch/state-spaces)
# 2. Load your GSS model and tokenizer as per their documentation.
# 3. Implement the generate method in a subclass or pass a compatible model instance.

# Dummy model for demonstration
class DummyGSSModel:
    def generate(self, input_ids, **kwargs):
        return input_ids + [555]
class DummyTokenizer:
    def encode(self, text): return [61, 62, 63]
    def decode(self, ids): return "gss output"

dummy_model = DummyGSSModel()
dummy_tokenizer = DummyTokenizer()

class DemoGSSLLM(GSSLLM):
    def __init__(self, model_name, model_instance, tokenizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer
    async def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.model.generate(input_ids, **kwargs)
        return self.tokenizer.decode(output_ids)

llm = DemoGSSLLM("demo_gss", dummy_model, dummy_tokenizer)

async def main():
    prompt = "Hello GSS"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 