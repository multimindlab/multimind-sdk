"""
Documentation Workflow

This workflow automates the documentation generation process using AI-powered analysis and multi-platform publishing.
"""

from typing import Any, Dict, List
from ...api.mcp.base import MCPWorkflowAPI
from ...api.mcp.registry import WorkflowRegistry

@WorkflowRegistry.register
class DocumentationWorkflow(MCPWorkflowAPI):
    """Documentation workflow implementation."""
    
    def __init__(
        self,
        models: Dict[str, Any],
        integrations: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the documentation workflow."""
        super().__init__(
            name="Documentation Generator",
            description="Automated documentation generation with AI analysis and multi-platform publishing",
            models=models,
            integrations=integrations,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def _build_workflow_spec(self) -> Dict[str, Any]:
        """Build the workflow specification."""
        return {
            "workflow": {
                "name": "Documentation Generator",
                "parallel": True,
                "steps": [
                    {
                        "name": "analyze_code",
                        "model": "gpt4",
                        "inputs": {
                            "code": "{{context.code}}",
                            "requirements": "{{context.requirements}}"
                        },
                        "prompt": """
                        Analyze the following code and requirements:
                        
                        Code:
                        {{inputs.code}}
                        
                        Requirements:
                        {{inputs.requirements}}
                        
                        Provide a detailed analysis including:
                        1. Code structure and architecture
                        2. Key components and their relationships
                        3. API endpoints and interfaces
                        4. Data models and schemas
                        5. Configuration and environment setup
                        """
                    },
                    {
                        "name": "generate_api_docs",
                        "model": "claude",
                        "inputs": {
                            "analysis": "{{steps.analyze_code.output}}"
                        },
                        "prompt": """
                        Based on the following code analysis, generate API documentation:
                        
                        Analysis:
                        {{inputs.analysis}}
                        
                        The documentation should include:
                        1. API overview
                        2. Endpoint specifications
                        3. Request/response formats
                        4. Authentication details
                        5. Rate limiting and quotas
                        6. Error handling
                        7. Code examples
                        """
                    },
                    {
                        "name": "generate_architecture_docs",
                        "model": "claude",
                        "inputs": {
                            "analysis": "{{steps.analyze_code.output}}"
                        },
                        "prompt": """
                        Based on the following code analysis, generate architecture documentation:
                        
                        Analysis:
                        {{inputs.analysis}}
                        
                        The documentation should include:
                        1. System overview
                        2. Component diagrams
                        3. Data flow diagrams
                        4. Deployment architecture
                        5. Security considerations
                        6. Performance characteristics
                        7. Scaling strategies
                        """
                    },
                    {
                        "name": "generate_user_guide",
                        "model": "claude",
                        "inputs": {
                            "analysis": "{{steps.analyze_code.output}}",
                            "api_docs": "{{steps.generate_api_docs.output}}"
                        },
                        "prompt": """
                        Based on the following analysis and API docs, generate a user guide:
                        
                        Analysis:
                        {{inputs.analysis}}
                        
                        API Documentation:
                        {{inputs.api_docs}}
                        
                        The guide should include:
                        1. Getting started
                        2. Installation instructions
                        3. Basic usage examples
                        4. Advanced features
                        5. Troubleshooting
                        6. Best practices
                        7. FAQ
                        """
                    },
                    {
                        "name": "publish_to_github",
                        "integration": "github",
                        "operation": "publish_docs",
                        "inputs": {
                            "api_docs": "{{steps.generate_api_docs.output}}",
                            "architecture_docs": "{{steps.generate_architecture_docs.output}}",
                            "user_guide": "{{steps.generate_user_guide.output}}",
                            "repo": "{{context.github_repo}}",
                            "branch": "{{context.github_branch}}"
                        }
                    },
                    {
                        "name": "send_slack_notification",
                        "integration": "slack",
                        "operation": "send_message",
                        "inputs": {
                            "channel": "{{context.slack_channel}}",
                            "message": """
                            *Documentation Update*
                            
                            Repository: {{context.github_repo}}
                            Branch: {{context.github_branch}}
                            
                            Documentation has been generated and published:
                            1. API Documentation
                            2. Architecture Documentation
                            3. User Guide
                            
                            View the docs at: {{steps.publish_to_github.output.docs_url}}
                            """
                        }
                    },
                    {
                        "name": "send_discord_notification",
                        "integration": "discord",
                        "operation": "send_message",
                        "inputs": {
                            "channel_id": "{{context.discord_channel}}",
                            "message": """
                            **Documentation Update**
                            
                            Repository: {{context.github_repo}}
                            Branch: {{context.github_branch}}
                            
                            Documentation has been generated and published:
                            1. API Documentation
                            2. Architecture Documentation
                            3. User Guide
                            
                            View the docs at: {{steps.publish_to_github.output.docs_url}}
                            """
                        }
                    }
                ],
                "connections": [
                    {
                        "from": "analyze_code",
                        "to": ["generate_api_docs", "generate_architecture_docs", "generate_user_guide"]
                    },
                    {
                        "from": "generate_api_docs",
                        "to": ["generate_user_guide", "publish_to_github"]
                    },
                    {
                        "from": "generate_architecture_docs",
                        "to": ["publish_to_github"]
                    },
                    {
                        "from": "generate_user_guide",
                        "to": ["publish_to_github"]
                    },
                    {
                        "from": "publish_to_github",
                        "to": ["send_slack_notification", "send_discord_notification"]
                    }
                ]
            }
        }
    
    def _validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate the workflow context."""
        required_fields = [
            "code",
            "requirements",
            "github_repo",
            "github_branch",
            "slack_channel",
            "discord_channel"
        ]
        return all(field in context for field in required_fields)
    
    @classmethod
    def _get_required_integrations(cls) -> List[str]:
        """Get required integrations."""
        return ["github", "slack", "discord"]
    
    @classmethod
    def _get_required_models(cls) -> List[str]:
        """Get required models."""
        return ["gpt4", "claude"] 