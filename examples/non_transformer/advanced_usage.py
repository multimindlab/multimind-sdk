import asyncio
from multimind.llm.non_transformer_llm import MambaLLM, HyenaLLM, RWKVLLM, CustomRNNLLM
from multimind.llm.model_registry import register_model, create_model

# Example: Register and use advanced non-transformer LLMs

async def main():
    # Register Mamba
    register_model("mamba", MambaLLM)
    mamba = create_model("mamba", model_name="state-spaces/mamba-130m")
    print("[Mamba]", await mamba.generate("What is Mamba?"))

    # Register Hyena
    register_model("hyena", HyenaLLM)
    hyena = create_model("hyena", model_name="EleutherAI/hyena-small")
    print("[Hyena]", await hyena.generate("What is Hyena?"))

    # Register RWKV
    register_model("rwkv", RWKVLLM)
    rwkv = create_model("rwkv", model_name="BlinkDL/rwkv-4-pile-169m")
    print("[RWKV]", await rwkv.generate("What is RWKV?"))

    # Register a custom RNN/MLP model (template)
    # Replace with your real PyTorch/Keras model and tokenizer
    class DummyRNN:
        def to(self, device): return self
        def eval(self): return self
        def generate(self, input_ids, max_length=32, temperature=0.7):
            # Dummy: just echo input
            return input_ids
    class DummyTokenizer:
        def encode(self, text, return_tensors=None):
            return torch.tensor([[1,2,3]])
        def decode(self, ids, skip_special_tokens=True):
            return "dummy response"
    dummy_model = DummyRNN()
    dummy_tokenizer = DummyTokenizer()
    register_model("custom-rnn", CustomRNNLLM)
    custom_rnn = create_model("custom-rnn", model_instance=dummy_model, tokenizer=dummy_tokenizer)
    print("[CustomRNN]", await custom_rnn.generate("Hello from custom RNN!"))

    # Example chat usage
    messages = [
        {"role": "user", "content": "Tell me about state space models."},
        {"role": "assistant", "content": "State space models are..."},
        {"role": "user", "content": "And Mamba?"}
    ]
    print("[Mamba Chat]", await mamba.chat(messages))

    # Extension: Use LoRA/PEFT adapters, batching, streaming, etc.
    # mamba = create_model("mamba", model_name="state-spaces/mamba-130m", adapter_path="./mamba_lora_adapter")
    # See multimind/llm/non_transformer_llm.py for more advanced options

if __name__ == "__main__":
    asyncio.run(main()) 