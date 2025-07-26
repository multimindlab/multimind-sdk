"""
Workflow registry for managing available MCP workflows.

This module provides a registry for managing and discovering available MCP workflows.
"""

from typing import Dict, List, Type, Any, Optional
from .base import MCPWorkflowAPI
from multimind.models.base import BaseLLM
from multimind.integrations.base import IntegrationHandler

class WorkflowRegistry:
    """Registry for managing MCP workflows."""
    
    _workflows: Dict[str, Type[MCPWorkflowAPI]] = {}
    _workflow_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        workflow_class: Type[MCPWorkflowAPI],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Type[MCPWorkflowAPI]:
        """
        Register a workflow class.
        
        Args:
            workflow_class: The workflow class to register
            metadata: Optional metadata about the workflow
            
        Returns:
            The registered workflow class
        """
        cls._workflows[workflow_class.__name__] = workflow_class
        cls._workflow_metadata[workflow_class.__name__] = {
            "info": workflow_class.get_workflow_info(),
            "metadata": metadata or {}
        }
        return workflow_class
    
    @classmethod
    def get_workflow(cls, name: str) -> Type[MCPWorkflowAPI]:
        """
        Get a workflow class by name.
        
        Args:
            name: Name of the workflow
            
        Returns:
            The workflow class
            
        Raises:
            KeyError: If workflow not found
        """
        if name not in cls._workflows:
            raise KeyError(f"Workflow '{name}' not found")
        return cls._workflows[name]
    
    @classmethod
    def create_workflow(
        cls,
        name: str,
        models: Dict[str, BaseLLM],
        integrations: Dict[str, IntegrationHandler],
        **kwargs
    ) -> MCPWorkflowAPI:
        """
        Create a workflow instance by name.
        
        Args:
            name: Name of the workflow
            models: Dictionary of model instances
            integrations: Dictionary of integration handlers
            **kwargs: Additional arguments for workflow initialization
            
        Returns:
            New workflow instance
            
        Raises:
            KeyError: If workflow not found
        """
        workflow_class = cls.get_workflow(name)
        return workflow_class.create_workflow(
            models=models,
            integrations=integrations,
            **kwargs
        )
    
    @classmethod
    def list_workflows(cls) -> List[Dict[str, Any]]:
        """
        List all registered workflows.
        
        Returns:
            List of workflow information dictionaries
        """
        return [
            {
                "name": name,
                **metadata["info"],
                "metadata": metadata["metadata"]
            }
            for name, metadata in cls._workflow_metadata.items()
        ]
    
    @classmethod
    def get_workflows_by_integration(cls, integration_name: str) -> List[Dict[str, Any]]:
        """
        Get workflows that use a specific integration.
        
        Args:
            integration_name: Name of the integration
            
        Returns:
            List of workflow information dictionaries
        """
        return [
            {
                "name": name,
                **metadata["info"],
                "metadata": metadata["metadata"]
            }
            for name, metadata in cls._workflow_metadata.items()
            if integration_name in metadata["info"]["required_integrations"]
        ]
    
    @classmethod
    def get_workflows_by_model(cls, model_name: str) -> List[Dict[str, Any]]:
        """
        Get workflows that use a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of workflow information dictionaries
        """
        return [
            {
                "name": name,
                **metadata["info"],
                "metadata": metadata["metadata"]
            }
            for name, metadata in cls._workflow_metadata.items()
            if model_name in metadata["info"]["required_models"]
        ]
        
    @classmethod
    def get_workflow_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific workflow.
        
        Args:
            name: Name of the workflow
            
        Returns:
            Workflow metadata dictionary
            
        Raises:
            KeyError: If workflow not found
        """
        if name not in cls._workflow_metadata:
            raise KeyError(f"Workflow '{name}' not found")
        return cls._workflow_metadata[name]
        
    @classmethod
    def update_workflow_metadata(
        cls,
        name: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update metadata for a specific workflow.
        
        Args:
            name: Name of the workflow
            metadata: New metadata dictionary
            
        Raises:
            KeyError: If workflow not found
        """
        if name not in cls._workflow_metadata:
            raise KeyError(f"Workflow '{name}' not found")
        cls._workflow_metadata[name]["metadata"].update(metadata)
        
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a workflow.
        
        Args:
            name: Name of the workflow
            
        Raises:
            KeyError: If workflow not found
        """
        if name not in cls._workflows:
            raise KeyError(f"Workflow '{name}' not found")
        del cls._workflows[name]
        del cls._workflow_metadata[name] 