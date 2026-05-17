"""
Integrations module for MultiMind SDK.

This module provides integrations with external services and platforms.
"""

from .base import IntegrationHandler
from .discord import DiscordIntegrationHandler
from .github import GitHubIntegrationHandler
from .jira import JiraIntegrationHandler
from .slack import SlackIntegrationHandler

__all__ = [
    "IntegrationHandler",
    "DiscordIntegrationHandler",
    "GitHubIntegrationHandler",
    "JiraIntegrationHandler",
    "SlackIntegrationHandler",
]
