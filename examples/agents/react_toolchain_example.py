"""
Sequential reasoning chain example (runs offline, no LLM required).

ReasoningChain passes each step's output to the next step in order:
question -> extracted expression -> computed value -> formatted answer.
"""

import ast
import operator
import re

from multimind.agents.react_toolchain import ReasoningChain, ReasoningStep

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def extract_expression(question, context=None):
    """Pull the arithmetic expression out of a natural-language question."""
    match = re.search(r"\d[\d\s+\-*/().^]*", question)
    if not match:
        raise ValueError(f"No arithmetic expression found in: {question!r}")
    return match.group().strip().replace("^", "**")


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression, context=None):
    """Safely evaluate an arithmetic expression (no eval)."""
    return _eval_node(ast.parse(expression, mode="eval").body)


def format_answer(value, context=None):
    return f"The answer is {value}"


chain = ReasoningChain(
    [
        ReasoningStep("extractor", extract_expression, "Extract arithmetic expression"),
        ReasoningStep("calculator", calculate, "Evaluate expression"),
        ReasoningStep("formatter", format_answer, "Format final answer"),
    ]
)


def print_hook(step, inp, out):
    print(f"Step: {step.name}, Input: {inp!r}, Output: {out!r}")


chain.add_hook(print_hook)


def main():
    result = chain.run("What is 2+2?")
    print("Final result:", result)


if __name__ == "__main__":
    main()
