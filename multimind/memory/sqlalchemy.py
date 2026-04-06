"""
SQLAlchemy-based memory implementation.
"""

from typing import List, Dict, Any
from datetime import datetime
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from .base import BaseMemory

class SQLAlchemyMemory(BaseMemory):
    """Memory that uses SQLAlchemy for database storage."""

    def __init__(
        self,
        database_url: str,
        memory_key: str = "chat_history",
        table_name: str = "messages"
    ):
        super().__init__(memory_key)
        engine_kwargs = {}
        if database_url.startswith("sqlite"):
            # Allow DB access from worker threads used by asyncio.to_thread.
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(database_url, **engine_kwargs)
        self.Session = sessionmaker(bind=self.engine)
        self.Base = declarative_base()
        self.MessageModel = self._create_message_model(self.Base, table_name)
        self.Base.metadata.create_all(self.engine)

    @staticmethod
    def _create_message_model(base, table_name: str):
        """Create a SQLAlchemy message model bound to the configured table name."""
        safe_table_name = table_name.strip() or "messages"
        return type(
            f"Message_{safe_table_name}_{id(base)}",
            (base,),
            {
                "__tablename__": safe_table_name,
                "__table_args__": {"extend_existing": True},
                "id": Column(Integer, primary_key=True),
                "role": Column(String),
                "content": Column(String),
                "timestamp": Column(DateTime, default=datetime.utcnow),
                "metadata": Column(JSON),
            },
        )

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message to database."""
        await asyncio.to_thread(self._add_message_sync, message)

    def _add_message_sync(self, message: Dict[str, str]) -> None:
        """Synchronous helper for inserting a message."""
        session = self.Session()
        try:
            db_message = self.MessageModel(
                role=message["role"],
                content=message["content"],
                metadata=message.get("metadata", {})
            )
            session.add(db_message)
            session.commit()
        finally:
            session.close()

    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from database."""
        return await asyncio.to_thread(self._get_messages_sync)

    def _get_messages_sync(self) -> List[Dict[str, str]]:
        """Synchronous helper for fetching all messages."""
        session = self.Session()
        try:
            messages = session.query(self.MessageModel).order_by(self.MessageModel.timestamp).all()
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

    async def clear(self) -> None:
        """Clear all messages from database."""
        await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        """Synchronous helper for clearing all messages."""
        session = self.Session()
        try:
            session.query(self.MessageModel).delete()
            session.commit()
        finally:
            session.close()

    async def save(self) -> None:
        """Save is handled automatically by SQLAlchemy."""
        pass

    async def load(self) -> None:
        """Load is handled automatically by SQLAlchemy."""
        pass

    def get_messages_by_role(self, role: str) -> List[Dict[str, str]]:
        """Get messages by role."""
        session = self.Session()
        try:
            messages = session.query(self.MessageModel).filter_by(role=role).all()
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
            messages = session.query(self.MessageModel).filter(
                self.MessageModel.timestamp > timestamp
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
            return session.query(self.MessageModel).count()
        finally:
            session.close() 