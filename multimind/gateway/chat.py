"""
Chat session management for the MultiMind Gateway
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    """A single chat message"""
    role: str
    content: str
    model: str
    timestamp: datetime = datetime.now()
    metadata: Dict = {}

class ChatSession(BaseModel):
    """A chat session with history and metadata"""
    session_id: str
    model: str
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    messages: List[ChatMessage] = []
    metadata: Dict = {}
    system_prompt: Optional[str] = None

    def add_message(self, role: str, content: str, model: str, metadata: Optional[Dict[str, Union[str, int, float]]] = None) -> None:
        """Add a message to the session"""
        if metadata is None:
            metadata = {}
        self.messages.append(ChatMessage(
            role=role,
            content=content,
            model=model,
            metadata=metadata
        ))
        self.updated_at = datetime.now()

    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent messages for context"""
        messages = self.messages[-max_messages:] if max_messages else self.messages
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def export(self, format: str = "json") -> Union[str, Dict]:
        """Export session to different formats"""
        if format == "json":
            return json.dumps(asdict(self), default=str)
        elif format == "dict":
            return asdict(self)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ChatSession":
        """Load session from file"""
        with open(file_path, "r") as f:
            data = json.load(f)
            # Convert string timestamps back to datetime
            for key in ["created_at", "updated_at"]:
                data[key] = datetime.fromisoformat(data[key])
            for msg in data["messages"]:
                msg["timestamp"] = datetime.fromisoformat(msg["timestamp"])
            return cls(**data)

    def save(self, directory: Union[str, Path]) -> Path:
        """Save session to file"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"chat_{self.session_id}.json"
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, default=str, indent=2)
        return file_path

class ChatManager:
    """Manage chat sessions and persistence"""

    def __init__(self, storage_dir: Union[str, Path] = "chat_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, ChatSession] = {}

    def create_session(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        metadata: Dict = None
    ) -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            model=model,
            system_prompt=system_prompt,
            metadata=metadata or {}
        )
        self.active_sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an active session by ID"""
        return self.active_sessions.get(session_id)

    def list_sessions(self) -> List[Dict]:
        """List all active sessions"""
        return [
            {
                "session_id": session.session_id,
                "model": session.model,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages)
            }
            for session in self.active_sessions.values()
        ]

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session from storage"""
        file_path = self.storage_dir / f"chat_{session_id}.json"
        if file_path.exists():
            try:
                session = ChatSession.from_file(file_path)
                self.active_sessions[session.session_id] = session
                return session
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
        return None

    def save_session(self, session_id: str) -> Optional[Path]:
        """Save a session to storage"""
        session = self.active_sessions.get(session_id)
        if session:
            try:
                return session.save(self.storage_dir)
            except Exception as e:
                logger.error(f"Error saving session {session_id}: {e}")
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            file_path = self.storage_dir / f"chat_{session_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

# Global chat manager instance
chat_manager = ChatManager()