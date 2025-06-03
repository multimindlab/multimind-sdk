"""
Example workflows for the MCP system.

This package contains example workflows demonstrating various use cases of the MCP system:
- CI/CD automation
- Code review automation
- Documentation generation
- Multi-platform issue management
- Basic workflow with Slack and Jira integrations
"""

from multimind.mcp.advanced_executor import AdvancedMCPExecutor
from multimind.models.base import BaseLLM
from multimind.integrations.github import GitHubIntegrationHandler
from multimind.integrations.jira import JiraIntegrationHandler
from multimind.integrations.slack import SlackIntegrationHandler
from multimind.integrations.discord import DiscordIntegrationHandler

from .ci_cd_workflow import main as ci_cd_workflow
from .code_review_workflow import main as code_review_workflow
from .mcp_workflow import main as mcp_workflow
from .multi_integration_workflow import main as multi_integration_workflow
from .documentation_workflow import main as documentation_workflow

__all__ = [
    'ci_cd_workflow',
    'code_review_workflow',
    'mcp_workflow',
    'multi_integration_workflow',
    'documentation_workflow',
    'AdvancedMCPExecutor',
    'BaseLLM',
    'GitHubIntegrationHandler',
    'JiraIntegrationHandler',
    'SlackIntegrationHandler',
    'DiscordIntegrationHandler'
] 