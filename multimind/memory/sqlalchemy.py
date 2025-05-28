"""
SQLAlchemy-based memory implementation.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .base import BaseMemory

Base = declarative_base()

class Message(Base):
    """SQLAlchemy model for storing messages."""
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    role = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)

class SQLAlchemyMemory(BaseMemory):
    """Memory that uses SQLAlchemy for database storage."""

    def __init__(
        self,
        database_url: str,
        memory_key: str = "chat_history",
        table_name: str = "messages"
    ):
        super().__init__(memory_key)
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def add_message(self, message: Dict[str, str]) -> None:
        """Add message to database."""
        session = self.Session()
        try:
            db_message = Message(
                role=message["role"],
                content=message["content"],
                metadata=message.get("metadata", {})
            )
            session.add(db_message)
            session.commit()
        finally:
            session.close()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from database."""
        session = self.Session()
        try:
            messages = session.query(Message).order_by(Message.timestamp).all()
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        finally:
            session.close()

    def clear(self) -> None:
        """Clear all messages from database."""
        session = self.Session()
        try:
            session.query(Message).delete()
            session.commit()
        finally:
            session.close()

    def save(self) -> None:
        """Save is handled automatically by SQLAlchemy."""
        pass

    def load(self) -> None:
        """Load is handled automatically by SQLAlchemy."""
        pass

    def get_messages_by_role(self, role: str) -> List[Dict[str, str]]:
        """Get messages by role."""
        session = self.Session()
        try:
            messages = session.query(Message).filter_by(role=role).all()
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        finally:
            session.close()

    def get_messages_since(self, timestamp: datetime) -> List[Dict[str, str]]:
        """Get messages since a specific timestamp."""
        session = self.Session()
        try:
            messages = session.query(Message).filter(
                Message.timestamp > timestamp
            ).all()
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        finally:
            session.close()

    def get_message_count(self) -> int:
        """Get the number of messages in memory."""
        session = self.Session()
        try:
            return session.query(Message).count()
        finally:
            session.close() 