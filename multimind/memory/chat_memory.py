"""
Chat memory implementation for managing conversation history with advanced features.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from .buffer import BufferMemory
from .token_buffer import TokenBufferMemory

class ChatMemory(BufferMemory):
    """Memory that manages chat history with advanced features."""

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_messages: Optional[int] = None,
        token_model: str = "gpt-3.5-turbo",
        roles: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Initialize chat memory."""
        super().__init__(max_messages=max_messages, **kwargs)
        
        # Token management
        self.max_tokens = max_tokens
        self.token_model = token_model
        self.token_buffer = TokenBufferMemory(
            max_tokens=max_tokens,
            token_model=token_model
        ) if max_tokens else None
        
        # Chat configuration
        self.roles = roles or ["system", "user", "assistant"]
        self.system_prompt = system_prompt
        
        # Chat state
        self.current_role: Optional[str] = None
        self.conversation_start = datetime.now()
        self.metadata.update({
            "conversation_id": None,
            "participants": set(),
            "topics": set(),
            "sentiment": None
        })

    async def add_message(
        self,
        message: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to chat history."""
        # Validate role
        role = message.get("role")
        if role not in self.roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {self.roles}")
            
        # Update current role
        self.current_role = role
        
        # Update metadata
        if metadata:
            if "participant" in metadata:
                self.metadata["participants"].add(metadata["participant"])
            if "topic" in metadata:
                self.metadata["topics"].add(metadata["topic"])
                
        # Add to token buffer if enabled
        if self.token_buffer:
            await self.token_buffer.add_message(message, metadata)
            
        # Add to main buffer
        await super().add_message(message, metadata)

    async def get_messages(
        self,
        role: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include_system: bool = True
    ) -> List[Dict[str, Any]]:
        """Get messages from chat history."""
        messages = await super().get_messages(limit, offset)
        
        # Filter by role if specified
        if role:
            messages = [
                m for m in messages
                if m.get("role") == role
            ]
            
        # Handle system prompt
        if not include_system:
            messages = [
                m for m in messages
                if m.get("role") != "system"
            ]
        elif self.system_prompt and not any(
            m.get("role") == "system" for m in messages
        ):
            messages.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })
            
        return messages

    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation."""
        return {
            "start_time": self.conversation_start,
            "duration": datetime.now() - self.conversation_start,
            "message_count": await self.get_message_count(),
            "participants": list(self.metadata["participants"]),
            "topics": list(self.metadata["topics"]),
            "current_role": self.current_role,
            "token_count": self.token_buffer.total_tokens if self.token_buffer else None,
            "metadata": self.metadata
        }

    async def get_messages_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get messages related to a specific topic."""
        return [
            m["message"] for m in self.messages
            if m["metadata"].get("topic") == topic
        ]

    async def get_messages_by_participant(
        self,
        participant: str
    ) -> List[Dict[str, Any]]:
        """Get messages from a specific participant."""
        return [
            m["message"] for m in self.messages
            if m["metadata"].get("participant") == participant
        ]

    async def get_recent_messages(
        self,
        n: int = 5,
        role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get the n most recent messages."""
        messages = await self.get_messages(role=role)
        return messages[-n:]

    async def get_messages_in_timeframe(
        self,
        start_time: datetime,
        end_time: datetime,
        role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get messages within a timeframe."""
        messages = await super().get_messages_in_timeframe(start_time, end_time)
        if role:
            messages = [
                m for m in messages
                if m.get("role") == role
            ]
        return messages

    async def clear(self) -> None:
        """Clear chat history."""
        await super().clear()
        if self.token_buffer:
            await self.token_buffer.clear()
        self.current_role = None
        self.conversation_start = datetime.now()
        self.metadata.update({
            "conversation_id": None,
            "participants": set(),
            "topics": set(),
            "sentiment": None
        })

    async def set_system_prompt(self, prompt: Optional[str]) -> None:
        """Set or clear the system prompt."""
        self.system_prompt = prompt

    async def add_participant(self, participant: str) -> None:
        """Add a participant to the conversation."""
        self.metadata["participants"].add(participant)

    async def add_topic(self, topic: str) -> None:
        """Add a topic to the conversation."""
        self.metadata["topics"].add(topic)

    async def set_sentiment(self, sentiment: str) -> None:
        """Set the conversation sentiment."""
        self.metadata["sentiment"] = sentiment

    async def get_token_count(self) -> Optional[int]:
        """Get the current token count."""
        return self.token_buffer.total_tokens if self.token_buffer else None 