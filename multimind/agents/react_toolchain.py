from typing import List, Callable, Any, Dict

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
            output = step(data, context=context)
            for hook in self.hooks:
                hook(step, data, output)
            data = output
        return data

# --- Example usage ---
if __name__ == "__main__":
    # Dummy retriever, generator, calculator
    def retrieve(query, context=None):
        return f"[Retrieved context for: {query}]"
    def generate(prompt, context=None):
        return f"[Generated answer for: {prompt}]"
    def calculate(expression, context=None):
        try:
            return str(eval(expression))
        except Exception:
            return "error"
    chain = ReasoningChain([
        ReasoningStep("retriever", retrieve, "Retrieve context"),
        ReasoningStep("generator", generate, "Generate answer"),
        ReasoningStep("calculator", calculate, "Calculate expression")
    ])
    def print_hook(step, inp, out):
        print(f"Step: {step.name}, Input: {inp}, Output: {out}")
    chain.add_hook(print_hook)
    result = chain.run("What is 2+2?")
    print("Final result:", result) 