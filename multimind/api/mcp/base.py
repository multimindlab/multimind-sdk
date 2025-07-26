"""
Base API class for MCP workflows.

This module provides a base class that users can extend to create their own MCP workflows.
It includes common functionality and utilities for workflow management.
"""

from typing import Any, Dict, List, Optional, Union, Type
from multimind.mcp.advanced_executor import AdvancedMCPExecutor
from multimind.models.base import BaseLLM
from multimind.integrations.base import IntegrationHandler
from multimind.observability.metrics import MetricsCollector

class MCPWorkflowAPI:
    """Base class for MCP workflow APIs."""
    
    def __init__(
        self,
        name: str,
        description: str,
        models: Dict[str, BaseLLM],
        integrations: Dict[str, IntegrationHandler],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the MCP workflow API.
        
        Args:
            name: Name of the workflow
            description: Description of the workflow
            models: Dictionary of model instances to use
            integrations: Dictionary of integration handlers
            max_retries: Maximum number of retries for failed steps
            retry_delay: Delay between retries in seconds
            metrics_collector: Optional metrics collector for monitoring
        """
        self.name = name
        self.description = description
        self.models = models
        self.integrations = integrations
        self.executor = AdvancedMCPExecutor(
            model_registry=models,
            max_retries=max_retries,
            retry_delay=retry_delay,
            metrics_collector=metrics_collector
        )
        
    async def execute(
        self,
        initial_context: Dict[str, Any],
        callbacks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            initial_context: Initial context for the workflow
            callbacks: Optional callbacks for workflow events
            
        Returns:
            Dict containing workflow results
        """
        # Validate context
        if not self._validate_context(initial_context):
            raise ValueError("Invalid workflow context")
            
        # Build workflow spec
        workflow_spec = self._build_workflow_spec()
        
        # Execute workflow
        return await self.executor.execute(
            spec=workflow_spec,
            initial_context=initial_context,
            callbacks=callbacks
        )
    
    def _build_workflow_spec(self) -> Dict[str, Any]:
        """Build the workflow specification. Must be implemented in subclass."""
        raise NotImplementedError("_build_workflow_spec must be implemented in a subclass of MCPWorkflowAPI.")
    
    def _validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate the workflow context. Must be implemented in subclass."""
        raise NotImplementedError("_validate_context must be implemented in a subclass of MCPWorkflowAPI.")
    
    @classmethod
    def _get_required_integrations(cls) -> List[str]:
        """Get list of required integrations. Must be implemented in subclass."""
        raise NotImplementedError("_get_required_integrations must be implemented in a subclass of MCPWorkflowAPI.")
    
    @classmethod
    def _get_required_models(cls) -> List[str]:
        """Get list of required models. Must be implemented in subclass."""
        raise NotImplementedError("_get_required_models must be implemented in a subclass of MCPWorkflowAPI.")
    
    @classmethod
    def get_workflow_info(cls) -> Dict[str, Any]:
        """
        Get information about the workflow.
        
        Returns:
            Dict containing workflow information
        """
        return {
            "name": cls.__name__,
            "description": cls.__doc__,
            "required_integrations": cls._get_required_integrations(),
            "required_models": cls._get_required_models()
        }
        
    @classmethod
    def create_workflow(
        cls,
        models: Dict[str, BaseLLM],
        integrations: Dict[str, IntegrationHandler],
        **kwargs
    ) -> 'MCPWorkflowAPI':
        """
        Create a new workflow instance.
        
        Args:
            models: Dictionary of model instances
            integrations: Dictionary of integration handlers
            **kwargs: Additional arguments for workflow initialization
            
        Returns:
            New workflow instance
            
        Raises:
            ValueError: If required models or integrations are missing
        """
        # Validate required models
        required_models = cls._get_required_models()
        missing_models = [m for m in required_models if m not in models]
        if missing_models:
            raise ValueError(f"Missing required models: {missing_models}")
            
        # Validate required integrations
        required_integrations = cls._get_required_integrations()
        missing_integrations = [i for i in required_integrations if i not in integrations]
        if missing_integrations:
            raise ValueError(f"Missing required integrations: {missing_integrations}")
            
        return cls(
            name=cls.__name__,
            description=cls.__doc__ or "",
            models=models,
            integrations=integrations,
            **kwargs
        ) 

class ExampleMCPWorkflowAPI(MCPWorkflowAPI):
    """Example concrete implementation of MCPWorkflowAPI."""
    def _build_workflow_spec(self) -> dict:
        # Minimal example spec
        return {"steps": ["step1", "step2"], "description": "Example workflow spec"}

    def _validate_context(self, context: dict) -> bool:
        # Accept any context with a 'user' key
        return isinstance(context, dict) and "user" in context

    @classmethod
    def _get_required_integrations(cls) -> list:
        return ["example_integration"]

    @classmethod
    def _get_required_models(cls) -> list:
        return ["example_model"]

class DefaultMCPWorkflowAPI(MCPWorkflowAPI):
    """Default implementation of MCPWorkflowAPI for basic workflows."""
    def _build_workflow_spec(self) -> dict:
        return {"steps": ["default_step"], "description": "Default workflow spec"}

    def _validate_context(self, context: dict) -> bool:
        return isinstance(context, dict)

    @classmethod
    def _get_required_integrations(cls) -> list:
        return []

    @classmethod
    def _get_required_models(cls) -> list:
        return [] 