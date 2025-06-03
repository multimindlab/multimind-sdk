"""
Base API class for MCP workflows.

This module provides a base class that users can extend to create their own MCP workflows.
It includes common functionality and utilities for workflow management.
"""

from typing import Any, Dict, List, Optional, Union
from multimind.mcp.advanced_executor import AdvancedMCPExecutor
from multimind.models.base import BaseLLM
from multimind.integrations.base import IntegrationHandler

class MCPWorkflowAPI:
    """Base class for MCP workflow APIs."""
    
    def __init__(
        self,
        name: str,
        description: str,
        models: Dict[str, BaseLLM],
        integrations: Dict[str, IntegrationHandler],
        max_retries: int = 3,
        retry_delay: float = 1.0
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
        """
        self.name = name
        self.description = description
        self.models = models
        self.integrations = integrations
        self.executor = AdvancedMCPExecutor(
            model_registry=models,
            max_retries=max_retries,
            retry_delay=retry_delay
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
        workflow_spec = self._build_workflow_spec()
        return await self.executor.execute(
            spec=workflow_spec,
            initial_context=initial_context,
            callbacks=callbacks
        )
    
    def _build_workflow_spec(self) -> Dict[str, Any]:
        """
        Build the workflow specification.
        
        Returns:
            Dict containing workflow specification
        """
        raise NotImplementedError("Subclasses must implement _build_workflow_spec")
    
    def _validate_context(self, context: Dict[str, Any]) -> bool:
        """
        Validate the workflow context.
        
        Args:
            context: Context to validate
            
        Returns:
            True if context is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement _validate_context")
    
    def _get_required_integrations(self) -> List[str]:
        """
        Get list of required integrations.
        
        Returns:
            List of required integration names
        """
        raise NotImplementedError("Subclasses must implement _get_required_integrations")
    
    def _get_required_models(self) -> List[str]:
        """
        Get list of required models.
        
        Returns:
            List of required model names
        """
        raise NotImplementedError("Subclasses must implement _get_required_models")
    
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