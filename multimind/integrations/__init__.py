"""
Integrations module for MultiMind SDK.

This module provides integrations with external services and platforms.
"""

from .base import IntegrationHandler
from .discord import DiscordIntegrationHandler
from .github import GitHubIntegrationHandler

__all__ = [
    "IntegrationHandler",
    "DiscordIntegrationHandler",
    "GitHubIntegrationHandler"
] 