"""
Executor for Model Composition Protocol (MCP) workflows.
"""

from typing import Dict, Any, List, Optional
from multimind.models.base import BaseLLM
from multimind.mcp.parser import MCPParser

class MCPExecutor:
    """Executes MCP workflows."""
    
    def __init__(
        self,
        parser: Optional[MCPParser] = None,
        model_registry: Optional[Dict[str, BaseLLM]] = None
    ):
        self.parser = parser or MCPParser()
        self.model_registry = model_registry or {}
        self.workflow_state: Dict[str, Any] = {}
        
    def register_model(self, name: str, model: BaseLLM) -> None:
        """Register a model for use in workflows."""
        self.model_registry[name] = model
        
    async def execute(
        self,
        spec: Dict[str, Any],
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute an MCP workflow."""
        # Parse and validate spec
        validated_spec = self.parser.parse(spec)
        
        # Initialize workflow state
        self.workflow_state = initial_context or {}
        
        # Execute workflow steps in order
        for step in validated_spec["workflow"]["steps"]:
            await self._execute_step(step, validated_spec)
            
        return self.workflow_state
        
    async def _execute_step(
        self,
        step: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> None:
        """Execute a single workflow step."""
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
        else:
            raise ValueError(f"Unsupported step type: {step_type}")
            
        # Update workflow state
        self.workflow_state[step_id] = result
        
    def _get_step_inputs(
        self,
        step: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get inputs for a step from workflow state."""
        inputs = {}
        
        # Find incoming connections
        for conn in spec["workflow"]["connections"]:
            if conn["to"] == step["id"]:
                from_step = conn["from"]
                if from_step in self.workflow_state:
                    inputs[from_step] = self.workflow_state[from_step]
                    
        return inputs
        
    async def _execute_model_step(
        self,
        step_id: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Any:
        """Execute a model step."""
        model_name = config["model"]
        if model_name not in self.model_registry:
            raise ValueError(f"Model not registered: {model_name}")
            
        model = self.model_registry[model_name]
        
        # Prepare prompt from inputs
        prompt = self._prepare_model_prompt(config, inputs)
        
        # Generate response
        response = await model.generate(prompt)
        
        return response
        
    async def _execute_transform_step(
        self,
        step_id: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Any:
        """Execute a transform step."""
        transform_type = config["type"]
        
        if transform_type == "join":
            # Join multiple inputs into a single string
            separator = config.get("separator", " ")
            return separator.join(str(v) for v in inputs.values())
            
        elif transform_type == "extract":
            # Extract specific fields from input
            field = config["field"]
            input_key = list(inputs.keys())[0]  # Use first input
            return inputs[input_key].get(field)
            
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")
            
    async def _execute_condition_step(
        self,
        step_id: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> bool:
        """Execute a condition step."""
        condition_type = config["type"]
        input_key = list(inputs.keys())[0]  # Use first input
        value = inputs[input_key]
        
        if condition_type == "equals":
            return value == config["value"]
        elif condition_type == "contains":
            return config["value"] in value
        elif condition_type == "greater_than":
            return value > config["value"]
        elif condition_type == "less_than":
            return value < config["value"]
        else:
            raise ValueError(f"Unsupported condition type: {condition_type}")
            
    def _prepare_model_prompt(
        self,
        config: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> str:
        """Prepare prompt for model step."""
        template = config["prompt_template"]
        
        # Replace placeholders with input values
        prompt = template
        for key, value in inputs.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
                
        return prompt 