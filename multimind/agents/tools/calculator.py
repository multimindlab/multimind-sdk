"""
Calculator tool for agents.
"""

import as
import operator
from typing import Any, Dic
from multimind.agents.tools.base import BaseTool

class CalculatorTool(BaseTool):
    """A tool for performing mathematical calculations."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg
        }

    async def run(self, expression: str) -> float:
        """Evaluate a mathematical expression."""
        if not self.validate_parameters(expression=expression):
            raise ValueError("Invalid parameters")

        try:
            return self._evaluate(expression)
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get tool parameters schema."""
        return {
            "required": ["expression"],
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            }
        }

    def _evaluate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        def _eval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return self.operators[type(node.op)](
                    _eval(node.left),
                    _eval(node.right)
                )
            elif isinstance(node, ast.UnaryOp):
                return self.operators[type(node.op)](_eval(node.operand))
            else:
                raise TypeError(f"Unsupported operation: {type(node)}")

        tree = ast.parse(expression, mode='eval')
        return _eval(tree.body)