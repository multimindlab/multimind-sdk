"""
Workflow registry for managing available MCP workflows.

This module provides a registry for managing and discovering available MCP workflows.
"""

from typing import Dict, List, Type
from .base import MCPWorkflowAPI

class WorkflowRegistry:
    """Registry for managing MCP workflows."""
    
    _workflows: Dict[str, Type[MCPWorkflowAPI]] = {}
    
    @classmethod
    def register(cls, workflow_class: Type[MCPWorkflowAPI]) -> Type[MCPWorkflowAPI]:
        """
        Register a workflow class.
        
        Args:
            workflow_class: The workflow class to register
            
        Returns:
            The registered workflow class
        """
        cls._workflows[workflow_class.__name__] = workflow_class
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
    def list_workflows(cls) -> List[Dict[str, Any]]:
        """
        List all registered workflows.
        
        Returns:
            List of workflow information dictionaries
        """
        return [
            workflow_class.get_workflow_info()
            for workflow_class in cls._workflows.values()
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
            workflow_class.get_workflow_info()
            for workflow_class in cls._workflows.values()
            if integration_name in workflow_class._get_required_integrations()
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
            workflow_class.get_workflow_info()
            for workflow_class in cls._workflows.values()
            if model_name in workflow_class._get_required_models()
        ] 