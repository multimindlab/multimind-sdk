"""
Prompt chaining functionality for orchestrating complex LLM interactions.
"""

from typing import List, Dict, Any, Optional, Callable
from multimind.models.base import BaseLLM

class PromptChain:
    """Manages a sequence of prompts and their execution."""

    def __init__(
        self,
        model: BaseLLM,
        prompts: Optional[List[Dict[str, Any]]] = None,
        variables: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.prompts = prompts or []
        self.variables = variables or {}
        self.results: List[Dict[str, Any]] = []

    def add_prompt(
        self,
        prompt: str,
        name: Optional[str] = None,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> None:
        """Add a prompt to the chain."""
        self.prompts.append({
            "prompt": prompt,
            "name": name,
            "condition": condition
        })

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable for use in prompts."""
        self.variables[name] = value

    async def run(self, initial_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run the prompt chain."""
        context = initial_context or {}
        self.results = []

        for prompt_info in self.prompts:
            # Check condition if presen
            if prompt_info["condition"] and not prompt_info["condition"](context):
                continue

            # Format prompt with variables
            formatted_prompt = self._format_prompt(prompt_info["prompt"], context)

            # Get response from model
            response = await self.model.generate(formatted_prompt)

            # Store resul
            result = {
                "prompt": formatted_prompt,
                "response": response,
                "name": prompt_info["name"]
            }
            self.results.append(result)

            # Update contex
            context.update({
                "last_response": response,
                "last_prompt": formatted_prompt
            })

        return self.results

    def _format_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Format prompt with variables and context."""
        # Combine variables and contex
        variables = {**self.variables, **context}

        # Replace variables in promp
        formatted = prompt
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(value))

        return formatted

    def get_results(self) -> List[Dict[str, Any]]:
        """Get results from the last run."""
        return self.results