from multimind.llm.non_transformer_llm import DiffusionTextLLM
import asyncio

# --- Diffusion Model for Text Integration Stub ---
# To use this stub:
# 1. Install a text diffusion model from open-source repos (see e.g. https://github.com/booydar/t2d, https://github.com/yang-song/score_sde_pytorch)
# 2. Load your diffusion model and tokenizer as per their documentation.
# 3. Implement the generate method in a subclass or pass a compatible model instance.

# Example (pseudo-code):
# from diffusion_library import DiffusionTextModel, DiffusionTokenizer
# model = DiffusionTextModel.from_pretrained('...')
# tokenizer = DiffusionTokenizer.from_pretrained('...')
# class MyDiffusionTextLLM(DiffusionTextLLM):
#     async def generate(self, prompt: str, **kwargs) -> str:
#         input_ids = tokenizer.encode(prompt)
#         output = model.sample(input_ids, **kwargs)
#         return tokenizer.decode(output)

# For demonstration, we use a dummy model
class DummyDiffusionModel:
    def sample(self, input_ids, **kwargs):
        return input_ids + [123]
class DummyTokenizer:
    def encode(self, text): return [10, 11, 12]
    def decode(self, ids): return "diffusion output"

dummy_model = DummyDiffusionModel()
dummy_tokenizer = DummyTokenizer()

class DemoDiffusionTextLLM(DiffusionTextLLM):
    def __init__(self, model_name, model_instance, tokenizer, **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer
    async def generate(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer.encode(prompt)
        output_ids = self.model.sample(input_ids, **kwargs)
        return self.tokenizer.decode(output_ids)

llm = DemoDiffusionTextLLM("demo_diffusion", dummy_model, dummy_tokenizer)

async def main():
    prompt = "Hello diffusion"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 