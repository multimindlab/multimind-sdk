"""
Entity memory implementation for tracking entities in conversations.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from pathlib import Path
from ..models.base import BaseLLM
from .base import BaseMemory

class EntityMemory(BaseMemory):
    """Memory that tracks entities mentioned in conversations."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        entity_types: Optional[List[str]] = None,
        storage_path: Optional[str] = None
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.entity_types = entity_types or [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
            "MONEY", "PERCENT", "PRODUCT", "EVENT"
        ]
        self.storage_path = Path(storage_path) if storage_path else None
        self.messages: List[Dict[str, str]] = []
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.load()

    def add_message(self, message: Dict[str, str]) -> None:
        """Add a message and extract entities."""
        self.messages.append({
            **message,
            "timestamp": datetime.now().isoformat()
        })
        self._extract_entities(message["content"])
        self.save()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages with entity information."""
        return self.messages

    def get_entities(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked entities."""
        return self.entities

    def get_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific entity."""
        return self.entities.get(entity_name)

    def clear(self) -> None:
        """Clear all messages and entities."""
        self.messages.clear()
        self.entities.clear()
        self.save()

    def save(self) -> None:
        """Save messages and entities to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "messages": self.messages,
                    "entities": self.entities
                }, f)

    def load(self) -> None:
        """Load messages and entities from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                self.entities = data.get("entities", {})

    async def _extract_entities(self, text: str) -> None:
        """Extract entities from text using the LLM."""
        prompt = f"""Extract entities from the following text. For each entity, provide:
1. Entity name
2. Entity type (one of: {', '.join(self.entity_types)})
3. Context or additional information

Text: {text}

Format the response as a JSON object with entity names as keys and their details as values."""

        try:
            response = await self.llm.generate(prompt)
            new_entities = json.loads(response)
            
            # Update entities with new information
            for entity_name, entity_info in new_entities.items():
                if entity_name in self.entities:
                    # Merge new information with existing
                    self.entities[entity_name].update(entity_info)
                else:
                    self.entities[entity_name] = entity_info
        except Exception as e:
            print(f"Error extracting entities: {e}")

    def get_entity_context(self, entity_name: str) -> Optional[str]:
        """Get the context in which an entity was mentioned."""
        entity = self.entities.get(entity_name)
        if not entity:
            return None
        
        # Find messages mentioning the entity
        mentions = []
        for msg in self.messages:
            if entity_name.lower() in msg["content"].lower():
                mentions.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(mentions) if mentions else None 