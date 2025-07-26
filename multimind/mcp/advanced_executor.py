"""
Advanced Executor for Model Composition Protocol (MCP) workflows.
Supports parallel execution, error handling, retries, and advanced workflow patterns.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
from datetime import datetime
import logging
from multimind.models.base import BaseLLM
from multimind.mcp.parser import MCPParser
from multimind.observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class DummyModel:
    async def execute(self, prompt: str) -> str:
        return "Dummy response"
    
    async def generate(self, prompt: str) -> str:
        return "Generated response for: " + prompt

class AdvancedMCPExecutor:
    """Advanced MCP workflow executor with enhanced capabilities."""

    def __init__(
        self,
        parser: Optional[MCPParser] = None,
        model_registry: Optional[Dict[str, BaseLLM]] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.parser = parser or MCPParser()
        self.metrics_collector = self.MetricsCollector()
        self.model_registry = {
            "ollama": OllamaModel(),
            "openai": OpenAIModel(),
            "claude": ClaudeModel(),
            "gemini": GeminiModel()
        }
        print(f"Model registry contents: {self.model_registry}")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.workflow_state: Dict[str, Any] = {}
        self.workflow_metadata: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "status": "pending",
            "error": None
        }

    async def execute(
        self,
        spec: Dict[str, Any],
        initial_context: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """Execute an advanced MCP workflow with enhanced features."""
        try:
            self.workflow_metadata["start_time"] = datetime.utcnow()
            self.workflow_metadata["status"] = "running"

            # Parse and validate spec
            validated_spec = self.parser.parse(spec)
            
            # Initialize workflow state
            self.workflow_state = initial_context or {}
            
            # Execute workflow steps
            if validated_spec["workflow"].get("parallel", False):
                await self._execute_parallel_steps(validated_spec)
            else:
                await self._execute_sequential_steps(validated_spec)

            # Execute callbacks if provided
            if callbacks and "on_success" in callbacks:
                await callbacks["on_success"](self.workflow_state)

            self.workflow_metadata["status"] = "completed"
            self.workflow_metadata["end_time"] = datetime.utcnow()
            
            return {
                "state": self.workflow_state,
                "metadata": self.workflow_metadata
            }

        except Exception as e:
            self.workflow_metadata["status"] = "failed"
            self.workflow_metadata["error"] = str(e)
            self.workflow_metadata["end_time"] = datetime.utcnow()
            
            if callbacks and "on_error" in callbacks:
                await callbacks["on_error"](e, self.workflow_state)
            
            raise

    async def _execute_sequential_steps(self, spec: Dict[str, Any]) -> None:
        """Execute workflow steps sequentially."""
        for step in spec["workflow"]["steps"]:
            await self._execute_step_with_retry(step, spec)

    async def _execute_parallel_steps(self, spec: Dict[str, Any]) -> None:
        """Execute workflow steps in parallel where possible."""
        # Group steps by their dependencies
        step_groups = self._group_steps_by_dependencies(spec)
        
        for group in step_groups:
            # Execute steps in each group in parallel
            await asyncio.gather(
                *[self._execute_step_with_retry(step, spec) for step in group]
            )

    def _group_steps_by_dependencies(self, spec: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Group steps by their dependencies for parallel execution."""
        steps = spec["workflow"]["steps"]
        connections = spec["workflow"]["connections"]
        
        # Build dependency graph
        dependencies = {step["id"]: set() for step in steps}
        for conn in connections:
            dependencies[conn["to"]].add(conn["from"])
        
        # Group steps by level
        groups = []
        remaining_steps = set(step["id"] for step in steps)
        
        while remaining_steps:
            # Find steps with no remaining dependencies
            current_group = []
            for step_id in remaining_steps.copy():
                if not dependencies[step_id]:
                    current_group.append(next(s for s in steps if s["id"] == step_id))
                    remaining_steps.remove(step_id)
            
            if not current_group:
                raise ValueError("Circular dependency detected in workflow")
            
            groups.append(current_group)
            
            # Update dependencies
            for step in current_group:
                for other_id in remaining_steps:
                    dependencies[other_id].discard(step["id"])
        
        return groups

    async def _execute_step_with_retry(
        self,
        step: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> None:
        """Execute a step with retry logic."""
        for attempt in range(self.max_retries):
            try:
                await self._execute_step(step, spec)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Step {step['id']} failed, retrying... ({attempt + 1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def _execute_step(
        self,
        step: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> None:
        """Execute a single workflow step with enhanced features."""
        step_type = step["type"]
        step_id = step["id"]
        config = step["config"]

        # Get step inputs from state
        inputs = self._get_step_inputs(step, spec)

        # Execute step based on type
        if step_type == "model":
            result = await self._execute_model_step(step_id, config, inputs)
        elif step_type == "transform":
            result = await self._execute_transform_step(step_id, config, inputs)
        elif step_type == "condition":
            result = await self._execute_condition_step(step_id, config, inputs)
        elif step_type == "integration":
            result = await self._execute_integration_step(step_id, config, inputs)
        else:
            raise ValueError(f"Unsupported step type: {step_type}")

        # Update workflow state
        self.workflow_state[step_id] = result

        # Collect metrics if available
        if self.metrics_collector:
            self.metrics_collector.record_step_execution(step_id=step_id, result=result)

    async def _execute_integration_step(
        self,
        step_id: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Any:
        """Execute an integration step."""
        integration_type = config["type"]
        integration_config = config["config"]

        # Import integration handler dynamically
        try:
            module = __import__(f"multimind.integrations.{integration_type}", fromlist=["IntegrationHandler"])
            handler_class = getattr(module, "IntegrationHandler")
            handler = handler_class(integration_config)
            return await handler.execute(inputs)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Integration {integration_type} not found or invalid: {str(e)}")

    async def _execute_model_step(
        self,
        step_id: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Any:
        """Execute a model step with advanced features."""
        model_name = config.get("model", "ollama")
        model = self.model_registry.get(model_name)
        if not model:
            raise ValueError(f"Model not registered or specified: {model_name}")

        # Prepare prompt from inputs
        prompt = self._prepare_model_prompt(config, inputs)

        # Generate response
        response = await model.generate(prompt)

        # Collect metrics if enabled
        if self.metrics_collector:
            self.metrics_collector.collect(step_id, response)

        return response

    async def _execute_transform_step(self, step_id: str, config: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        """Execute a transformation step."""
        # Example implementation: Apply a transformation to inputs
        return {key: value.upper() for key, value in inputs.items()}

    async def _execute_condition_step(self, step_id: str, config: Dict[str, Any], inputs: Dict[str, Any]) -> bool:
        """Execute a condition step."""
        # Example implementation: Check a condition on inputs
        return all(value.isalpha() for value in inputs.values())

    def _prepare_model_prompt(self, config: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """Prepare a prompt for the model based on config and inputs."""
        return f"Prompt: {config.get('prefix', '')} {inputs.get('text', '')}"

    class MetricsCollector:
        def record_step_execution(self, step_id: str, result: Any) -> None:
            """Record the execution of a step."""
            print(f"Step {step_id} executed with result: {result}")

        def collect(self, step_id: str, response: Any) -> None:
            """Collect metrics for a step."""
            print(f"Metrics collected for step {step_id}: {response}")

    def _get_step_inputs(
        self,
        step: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get inputs for a step from workflow state with enhanced validation."""
        inputs = {}
        required_inputs = step.get("required_inputs", [])

        # Find incoming connections
        for conn in spec["workflow"]["connections"]:
            if conn["to"] == step["id"]:
                from_step = conn["from"]
                if from_step in self.workflow_state:
                    inputs[from_step] = self.workflow_state[from_step]

        # Validate required inputs
        missing_inputs = [inp for inp in required_inputs if inp not in inputs]
        if missing_inputs:
            raise ValueError(f"Missing required inputs for step {step['id']}: {missing_inputs}")

        return inputs

class OllamaModel:
    def __init__(self):
        self.name = "Ollama"

    async def generate(self, prompt: str) -> str:
        return f"Ollama response for: {prompt}"

class OpenAIModel:
    async def generate(self, prompt: str) -> str:
        return f"OpenAI response for: {prompt}"

class ClaudeModel:
    async def generate(self, prompt: str) -> str:
        return f"Claude response for: {prompt}"

class GeminiModel:
    async def generate(self, prompt: str) -> str:
        return f"Gemini response for: {prompt}"