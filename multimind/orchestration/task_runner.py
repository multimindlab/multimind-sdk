"""
Task runner for orchestrating complex workflows.
"""

from typing import List, Dict, Any, Optional, Callable, Union
from multimind.models.base import BaseLLM
from multimind.orchestration.prompt_chain import PromptChain

class TaskRunner:
    """Manages execution of complex tasks using LLMs and tools."""

    def __init__(
        self,
        model: BaseLLM,
        tasks: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3
    ):
        self.model = model
        self.tasks = tasks or []
        self.max_retries = max_retries
        self.results: Dict[str, Any] = {}

    def add_task(
        self,
        name: str,
        prompt: Union[str, PromptChain],
        dependencies: Optional[List[str]] = None,
        retry_prompt: Optional[str] = None
    ) -> None:
        """Add a task to the runner."""
        self.tasks.append({
            "name": name,
            "prompt": prompt,
            "dependencies": dependencies or [],
            "retry_prompt": retry_promp
        })

    async def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all tasks in dependency order."""
        context = initial_context or {}
        self.results = {}

        # Sort tasks by dependencies
        sorted_tasks = self._sort_tasks()

        for task in sorted_tasks:
            result = await self._run_task(task, context)
            self.results[task["name"]] = resul
            context[task["name"]] = resul

        return self.results

    def _sort_tasks(self) -> List[Dict[str, Any]]:
        """Sort tasks based on dependencies."""
        # Create dependency graph
        graph = {task["name"]: set(task["dependencies"]) for task in self.tasks}

        # Topological sor
        visited = set()
        temp = set()
        order = []

        def visit(name):
            if name in temp:
                raise ValueError(f"Circular dependency detected: {name}")
            if name in visited:
                return

            temp.add(name)
            for dep in graph.get(name, set()):
                visit(dep)
            temp.remove(name)
            visited.add(name)
            order.append(name)

        for task in self.tasks:
            if task["name"] not in visited:
                visit(task["name"])

        # Return tasks in sorted order
        return sorted(self.tasks, key=lambda t: order.index(t["name"]))

    async def _run_task(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Run a single task with retries."""
        prompt = task["prompt"]
        retries = 0

        while retries < self.max_retries:
            try:
                if isinstance(prompt, PromptChain):
                    results = await prompt.run(context)
                    return results[-1]["response"] if results else None
                else:
                    # Format prompt with contex
                    formatted_prompt = self._format_prompt(prompt, context)
                    return await self.model.generate(formatted_prompt)

            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    raise RuntimeError(f"Task {task['name']} failed after {retries} retries: {str(e)}")

                # Use retry prompt if available
                if task["retry_prompt"]:
                    prompt = task["retry_prompt"]

    def _format_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Format prompt with context variables."""
        formatted = promp
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(value))
        return formatted

    def get_results(self) -> Dict[str, Any]:
        """Get results from the last run."""
        return self.results