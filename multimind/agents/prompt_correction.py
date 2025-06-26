from typing import Callable, Any, Dict, List
import logging

class PromptCorrectionLayer:
    """
    Observability and self-healing layer for LLM/agent pipelines.
    Monitors for failures/hallucinations, allows live prompt/adapters edits, and supports trace-based correction.
    """
    def __init__(self):
        self.error_hooks: List[Callable[[str, Exception, Dict], None]] = []
        self.correction_hooks: List[Callable[[str, Dict], str]] = []
        self.adapter_update_hooks: List[Callable[[str, str], None]] = []
        self.logger = logging.getLogger("PromptCorrectionLayer")

    def add_error_hook(self, hook: Callable[[str, Exception, Dict], None]):
        self.error_hooks.append(hook)
    def add_correction_hook(self, hook: Callable[[str, Dict], str]):
        self.correction_hooks.append(hook)
    def add_adapter_update_hook(self, hook: Callable[[str, str], None]):
        self.adapter_update_hooks.append(hook)

    def monitor(self, prompt: str, output: str, trace: Dict = None) -> str:
        """
        Monitor output for errors/hallucinations and apply corrections if needed.
        """
        trace = trace or {}
        try:
            # Example: simple hallucination check (can be replaced with real logic)
            if "[error]" in output or "hallucination" in output.lower():
                self.logger.warning(f"Detected issue in output: {output}")
                for hook in self.error_hooks:
                    hook(prompt, Exception("Detected hallucination"), trace)
                # Apply correction hooks
                for hook in self.correction_hooks:
                    prompt = hook(prompt, trace)
                self.logger.info(f"Corrected prompt: {prompt}")
                return prompt
            return prompt
        except Exception as e:
            self.logger.error(f"Prompt correction failed: {e}")
            return prompt

    def update_adapter(self, adapter_key: str, new_adapter_path: str):
        """
        Live update of adapters (e.g., swap LoRA/PEFT on the fly).
        """
        for hook in self.adapter_update_hooks:
            hook(adapter_key, new_adapter_path)
        self.logger.info(f"Adapter {adapter_key} updated to {new_adapter_path}")

# --- Example usage ---
if __name__ == "__main__":
    pcl = PromptCorrectionLayer()
    def error_logger(prompt, exc, trace):
        print(f"Error detected for prompt '{prompt}': {exc}")
    def simple_correction(prompt, trace):
        return prompt + " [CORRECTED]"
    def adapter_updater(adapter_key, new_path):
        print(f"Adapter {adapter_key} updated to {new_path}")
    pcl.add_error_hook(error_logger)
    pcl.add_correction_hook(simple_correction)
    pcl.add_adapter_update_hook(adapter_updater)
    # Simulate monitoring
    new_prompt = pcl.monitor("What is the capital of France?", "[error] hallucination detected", {"step": 1})
    print("New prompt after correction:", new_prompt)
    pcl.update_adapter("user123", "lora_adapter_v2") 