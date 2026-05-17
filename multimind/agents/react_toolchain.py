from typing import Any, Callable, Dict, List


class ReasoningChainExecutionError(RuntimeError):
    """Raised when a reasoning chain step fails."""


class ReasoningStep:
    """
    Represents a single step in a reasoning/toolchain. Can be a model call, tool call, or custom function.
    """

    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ReasoningChain:
    """
    Modular chain for step-by-step reasoning and tool use (ReAct/Toolformer style).
    Each step can be a model, tool, or function. Hooks can be added for logging/inspection.
    """

    def __init__(self, steps: List[ReasoningStep]):
        self.steps = steps
        self.hooks = []  # List of callables: hook(step, input, output)

    def add_hook(self, hook: Callable[[ReasoningStep, Any, Any], None]):
        self.hooks.append(hook)

    def run(self, input_data: Any, context: Dict = None):
        context = context or {}
        data = input_data
        for step in self.steps:
            try:
                output = step(data, context=context)
            except Exception as e:
                context["last_failed_step"] = step.name
                context.setdefault("errors", []).append({"step": step.name, "error": str(e)})
                raise ReasoningChainExecutionError(
                    f"Reasoning chain failed at step '{step.name}': {e}"
                ) from e

            for hook in self.hooks:
                try:
                    hook(step, data, output)
                except Exception as e:
                    context.setdefault("hook_errors", []).append(
                        {
                            "step": step.name,
                            "hook": getattr(hook, "__name__", str(hook)),
                            "error": str(e),
                        }
                    )
            data = output
        return data


# --- Example usage ---
# This block is for demonstration purposes only.
