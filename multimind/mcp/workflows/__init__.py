"""
Pre-built MCP workflows.

This package contains pre-built workflows for common tasks:
- Code review automation
- CI/CD pipeline management
- Documentation generation
- Multi-platform issue management
"""

from .code_review import CodeReviewWorkflow
from .ci_cd import CICDWorkflow
from .documentation import DocumentationWorkflow

__all__ = [
    'CodeReviewWorkflow',
    'CICDWorkflow',
    'DocumentationWorkflow',
] 