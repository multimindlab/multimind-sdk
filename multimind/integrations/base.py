"""
Base integration handler for MCP workflows.
"""

from typing import Dict, Any, Optional, Protocol, List
from abc import ABC, abstractmethod
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class IntegrationHandler(ABC):
    """Base class for all integration handlers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize integration handler with configuration."""
        self.config = config
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "error_count": 0,
            "success_count": 0
        }

    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integration operation.
        
        Args:
            inputs: Dictionary containing operation inputs
            
        Returns:
            Dictionary containing operation results
        """
        pass

    def _update_metadata(self, success: bool = True) -> None:
        """Update handler metadata."""
        self.metadata["last_used"] = datetime.utcnow().isoformat()
        if success:
            self.metadata["success_count"] += 1
        else:
            self.metadata["error_count"] += 1

    def get_metadata(self) -> Dict[str, Any]:
        """Get handler metadata."""
        return self.metadata

    def validate_config(self, required_fields: List[str]) -> None:
        """Validate configuration has required fields.
        
        Args:
            required_fields: List of required configuration field names
            
        Raises:
            ValueError: If any required field is missing
        """
        missing_fields = [field for field in required_fields if field not in self.config]
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing_fields)}"
            )

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

class AsyncContextManager(Protocol):
    """Protocol for async context managers."""
    
    async def __aenter__(self) -> 'AsyncContextManager':
        """Enter async context."""
        ...
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        ... 