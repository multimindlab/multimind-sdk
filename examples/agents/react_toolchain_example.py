# Example usage for demonstration purposes only
from multimind.agents.react_toolchain import ReasoningChain, ReasoningStep

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