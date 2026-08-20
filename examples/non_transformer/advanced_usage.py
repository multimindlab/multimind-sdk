import asyncio

import torch

from multimind.llm.non_transformer_llm import RWKVLLM, CustomRNNLLM, MambaLLM

# Example: Advanced non-transformer LLM usage.
# MambaLLM and RWKVLLM run real HuggingFace inference; the other wrapper
# classes are extension scaffolds that must be subclassed and implemented.

async def main():
    # Mamba (downloads the HF model on first use)
    mamba = MambaLLM(model_name="state-spaces/mamba-130m")
    print("[Mamba]", await mamba.generate("What is Mamba?"))

    # RWKV (downloads the HF model on first use)
    rwkv = RWKVLLM(model_name="BlinkDL/rwkv-4-pile-169m")
    print("[RWKV]", await rwkv.generate("What is RWKV?"))

    # A custom RNN/MLP model (template)
    # Replace with your real PyTorch/Keras model and tokenizer
    class DummyRNN(torch.nn.Module):
        def generate(self, input_ids, max_length=32, temperature=0.7):
            # Dummy: just echo input
            return input_ids
    class DummyTokenizer:
        def encode(self, text, return_tensors=None):
            return torch.tensor([[1, 2, 3]])
        def decode(self, ids, skip_special_tokens=True):
            return "dummy response"
    custom_rnn = CustomRNNLLM(model_instance=DummyRNN(), tokenizer=DummyTokenizer())
    print("[CustomRNN]", await custom_rnn.generate("Hello from custom RNN!"))

    # Example chat usage
    messages = [
        {"role": "user", "content": "Tell me about state space models."},
        {"role": "assistant", "content": "State space models are..."},
        {"role": "user", "content": "And Mamba?"}
    ]
    print("[Mamba Chat]", await mamba.chat(messages))

    # Extension: Use LoRA/PEFT adapters, batching, streaming, etc.
    # mamba = MambaLLM(model_name="state-spaces/mamba-130m", adapter_path="./mamba_lora_adapter")
    # See multimind/llm/non_transformer_llm.py for more advanced options

if __name__ == "__main__":
    asyncio.run(main())
