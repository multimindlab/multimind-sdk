"""
Chat session management for the MultiMind Gateway API
"""

from ..core.chat import ChatManager, ChatSession, ChatMessage

# Re-export the chat manager for API use
chat_manager = ChatManager()