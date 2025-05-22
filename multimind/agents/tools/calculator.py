"""
Calculator tool for agents.
"""

import ast
import operator
from typing import Any, Dict, Union, Optional
from numbers import Real
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

    async def run(self, **kwargs) -> Union[int, float]:
        """Evaluate a mathematical expression."""
        if not self.validate_parameters(**kwargs):
            raise ValueError("Invalid parameters")

        expression = kwargs['expression']
        try:
            result = self._evaluate(expression)
            if isinstance(result, complex):
                raise ValueError("Complex numbers are not supported")
            return float(result)
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

    def _evaluate(self, expression: str) -> Union[int, float]:
        """Safely evaluate a mathematical expression."""
        def _eval(node) -> Union[int, float]:
            if isinstance(node, ast.Num):
                val = node.n
                if isinstance(val, complex):
                    raise ValueError("Complex numbers are not supported")
                return float(val) if isinstance(val, Real) else val
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
        result = _eval(tree.body)
        if isinstance(result, complex):
            raise ValueError("Complex numbers are not supported")
        return float(result)