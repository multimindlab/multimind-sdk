# Example: Real Mamba Model Integration (HuggingFace)
# Requires: pip install transformers mamba-ssm torch
# See: https://huggingface.co/state-spaces/mamba-130m-hf

from multimind.llm.non_transformer_llm import MambaLLM
import torch
import asyncio

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("Please install transformers and mamba-ssm: pip install transformers mamba-ssm torch")

# Load real Mamba model and tokenizer from HuggingFace
MODEL_NAME = "state-spaces/mamba-130m-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

class RealMambaLLM(MambaLLM):
    def __init__(self, model_name, model_instance, tokenizer, device="cpu", **kwargs):
        super().__init__(model_name, model_instance, **kwargs)
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
    async def generate(self, prompt: str, max_new_tokens: int = 32, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

llm = RealMambaLLM("mamba-130m-hf", model, tokenizer, device="cpu")

async def main():
    prompt = "The future of AI is"
    result = await llm.generate(prompt)
    print(f"Prompt: {prompt}\nGenerated: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 