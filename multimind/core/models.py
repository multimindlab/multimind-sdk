"""
Core model functionality for MultiMind
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Standardized response from any model"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    timestamp: str = datetime.now().isoformat()

class ModelHandler(ABC):
    """Abstract base class for model handlers"""

    def __init__(self, model_config):
        self.config = model_config
        self._client = None

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Send a chat message to the model"""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text from a prompt"""
        pass 