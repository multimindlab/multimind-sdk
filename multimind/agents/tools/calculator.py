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

    MAX_EXPRESSION_LENGTH = 512
    MAX_AST_NODES = 128
    MAX_AST_DEPTH = 24

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
        if len(expression) > self.MAX_EXPRESSION_LENGTH:
            raise ValueError(
                f"Expression too long (max {self.MAX_EXPRESSION_LENGTH} characters)"
            )

        def _ast_depth(node: ast.AST) -> int:
            children = list(ast.iter_child_nodes(node))
            if not children:
                return 1
            return 1 + max(_ast_depth(child) for child in children)

        def _eval(node) -> Union[int, float]:
            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, complex):
                    raise ValueError("Complex numbers are not supported")
                if not isinstance(val, Real):
                    raise TypeError(f"Unsupported constant type: {type(val)}")
                return float(val)
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
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.MAX_AST_NODES:
            raise ValueError(f"Expression too complex (max {self.MAX_AST_NODES} AST nodes)")
        if _ast_depth(tree) > self.MAX_AST_DEPTH:
            raise ValueError(f"Expression too deeply nested (max depth {self.MAX_AST_DEPTH})")

        result = _eval(tree.body)
        if isinstance(result, complex):
            raise ValueError("Complex numbers are not supported")
        return float(result)