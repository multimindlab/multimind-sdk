"""
Base tool class for agent tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """Base class for all agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Run the tool with given parameters."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters()
        }
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get tool parameters schema."""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate tool parameters."""
        required_params = self.get_parameters().get("required", [])
        return all(param in kwargs for param in required_params) 