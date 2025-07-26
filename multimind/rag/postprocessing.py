"""
Post-processing module for RAG results.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing."""
    enabled: bool = True
    max_results: int = 10
    threshold: float = 0.5


class PostProcessor:
    """Base class for post-processing RAG results."""
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        self.config = config or PostProcessingConfig()
    
    def process(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process RAG results."""
        if not self.config.enabled:
            return results
        
        # Basic filtering by threshold
        filtered_results = [
            result for result in results 
            if result.get('score', 0) >= self.config.threshold
        ]
        
        # Limit results
        return filtered_results[:self.config.max_results] 