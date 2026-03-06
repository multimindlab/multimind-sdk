"""
Base memory class for all memory implementations.
"""

from abc import ABC, abstractmethod
import inspect
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

class BaseMemory(ABC):
    """Abstract base class for all memory implementations."""

    def __init__(self, memory_key: str = "chat_history"):
        self.memory_key = memory_key
        self.created_at = datetime.now()

    def __init_subclass__(cls, **kwargs):
        """Auto-wrap sync overrides so BaseMemory API stays async."""
        super().__init_subclass__(**kwargs)
        for method_name in ("add_message", "get_messages", "clear", "save", "load"):
            method = cls.__dict__.get(method_name)
            if method is None:
                continue
            if inspect.iscoroutinefunction(method):
                continue
            setattr(cls, method_name, BaseMemory._make_async_wrapper(method))

    @abstractmethod
    async def add_message(self, message: Dict[str, str]) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from memory."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all messages from memory."""
        pass

    @abstractmethod
    async def save(self) -> None:
        """Save memory to persistent storage."""
        pass

    @abstractmethod
    async def load(self) -> None:
        """Load memory from persistent storage."""
        pass 

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """Await value when needed; otherwise return directly."""
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _make_async_wrapper(sync_method):
        """Wrap a synchronous method with an async callable."""
        async def _wrapped(self, *args, **kwargs):
            return sync_method(self, *args, **kwargs)
        _wrapped.__name__ = sync_method.__name__
        _wrapped.__doc__ = sync_method.__doc__
        return _wrapped

    async def add_message_async(self, message: Dict[str, str]) -> None:
        """Backward-compatible alias for async add_message()."""
        await self.add_message(message)

    async def get_messages_async(self) -> List[Dict[str, str]]:
        """Backward-compatible alias for async get_messages()."""
        return await self.get_messages()

    async def clear_async(self) -> None:
        """Backward-compatible alias for async clear()."""
        await self.clear()

    async def save_async(self) -> None:
        """Backward-compatible alias for async save()."""
        await self.save()

    async def load_async(self) -> None:
        """Backward-compatible alias for async load()."""
        await self.load()