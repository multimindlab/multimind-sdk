"""
Simple example: Load and use a HuggingFace model locally

This is the most basic example showing how to load and use
a HuggingFace model without any API keys.

Usage:
    python simple_local_hf.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_and_test_model(model_name: str = "gpt2"):
    """
    Load a HuggingFace model locally and generate text.
    
    Args:
        model_name: Name of the HuggingFace model to load
    """
    print(f"Loading {model_name}...")
    print("(First time will download the model, this may take a minute)")
    
    # Load tokenizer and model
    # No token needed for public models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print()
    
    # Generate text
    prompt = "The future of AI is"
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt from response
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text
    
    print(f"Response: {response}")
    print()
    
    return tokenizer, model


if __name__ == "__main__":
    # Test with GPT-2 (smallest, fastest)
    print("=" * 60)
    print("Local HuggingFace Model Test")
    print("=" * 60)
    print()
    
    try:
        load_and_test_model("gpt2")
        
        print("=" * 60)
        print("Success! You can now use HuggingFace models locally.")
        print()
        print("Other models you can try:")
        print("  - distilgpt2 (even smaller)")
        print("  - microsoft/DialoGPT-small (for conversations)")
        print()
        print("Note: Models are cached in ~/.cache/huggingface/")
        print("=" * 60)
        
    except ImportError:
        print("ERROR: transformers and torch are required!")
        print("Install with: pip install transformers torch")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


