import logging
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


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

    def _compute_issue_score(self, prompt: str, output: str, trace: Dict) -> float:
        """
        Heuristic scoring function for potential issues / hallucinations.
        Returns a score in [0, 1], where higher means more suspicious.
        """
        text = output.lower()
        score = 0.0

        # Strong indicators
        strong_markers = [
            "[error]",
            "hallucination",
            "not based on real data",
            "fabricated answer",
            "made this up",
        ]
        if any(marker in text for marker in strong_markers):
            score += 0.7

        # Weaker indicators based on uncertainty phrases
        weak_markers = [
            "i am not sure",
            "i'm not sure",
            "i do not know",
            "i don't know",
            "cannot verify",
            "not certain",
        ]
        if any(marker in text for marker in weak_markers):
            score += 0.2

        # If trace provides an explicit model_score / confidence, incorporate it.
        # Expecting trace.get("confidence") in [0, 1] where low is suspicious.
        confidence = trace.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_clamped = max(0.0, min(1.0, float(confidence)))
            score += (1.0 - confidence_clamped) * 0.3

        return min(score, 1.0)

    def monitor(self, prompt: str, output: str, trace: Dict = None) -> str:
        """
        Monitor output for errors/hallucinations and apply corrections if needed.
        Uses a heuristic score instead of a single string check.
        """
        trace = trace or {}
        try:
            issue_score = self._compute_issue_score(prompt, output, trace)
            threshold = trace.get("hallucination_threshold", 0.6)
            if issue_score >= threshold:
                self.logger.warning(
                    "Detected potential hallucination (score=%.2f, threshold=%.2f): %s",
                    issue_score,
                    threshold,
                    output,
                )
                for hook in self.error_hooks:
                    hook(prompt, Exception("Detected hallucination"), trace)
                corrected_output = output
                for hook in self.correction_hooks:
                    corrected_output = hook(corrected_output, trace)
                self.logger.info("Corrected output: %s", corrected_output)
                return corrected_output
            return output
        except Exception as e:
            self.logger.error(f"Prompt correction failed: {e}")
            return output

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
        logger.error("Error detected for prompt '%s': %s", prompt, exc)

    def simple_correction(prompt, trace):
        return prompt + " [CORRECTED]"

    def adapter_updater(adapter_key, new_path):
        logger.info("Adapter %s updated to %s", adapter_key, new_path)

    pcl.add_error_hook(error_logger)
    pcl.add_correction_hook(simple_correction)
    pcl.add_adapter_update_hook(adapter_updater)
    # Simulate monitoring
    corrected_output = pcl.monitor(
        "What is the capital of France?", "[error] hallucination detected", {"step": 1}
    )
    logger.info("Corrected output after correction: %s", corrected_output)
    pcl.update_adapter("user123", "lora_adapter_v2")
