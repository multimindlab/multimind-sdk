"""
Code Review Workflow

This workflow automates the code review process using AI-powered analysis and multi-platform notifications.
"""

from typing import Any, Dict, List
from ..api.base import MCPWorkflowAPI
from ..api.registry import WorkflowRegistry

@WorkflowRegistry.register
class CodeReviewWorkflow(MCPWorkflowAPI):
    """Code review workflow implementation."""
    
    def __init__(
        self,
        models: Dict[str, Any],
        integrations: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the code review workflow."""
        super().__init__(
            name="Code Review",
            description="Automated code review with AI analysis and multi-platform notifications",
            models=models,
            integrations=integrations,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def _build_workflow_spec(self) -> Dict[str, Any]:
        """Build the workflow specification."""
        return {
            "workflow": {
                "name": "Code Review",
                "parallel": True,
                "steps": [
                    {
                        "name": "analyze_code",
                        "model": "gpt4",
                        "inputs": {
                            "code_changes": "{{context.code_changes}}",
                            "pr_description": "{{context.pr_description}}"
                        },
                        "prompt": """
                        Analyze the following code changes and PR description:
                        
                        Code Changes:
                        {{inputs.code_changes}}
                        
                        PR Description:
                        {{inputs.pr_description}}
                        
                        Provide a detailed analysis including:
                        1. Code quality assessment
                        2. Potential bugs or issues
                        3. Security concerns
                        4. Performance implications
                        5. Suggested improvements
                        """
                    },
                    {
                        "name": "generate_review_comment",
                        "model": "claude",
                        "inputs": {
                            "analysis": "{{steps.analyze_code.output}}"
                        },
                        "prompt": """
                        Based on the following code analysis, generate a constructive review comment:
                        
                        {{inputs.analysis}}
                        
                        The comment should:
                        1. Be clear and actionable
                        2. Highlight both positive aspects and areas for improvement
                        3. Provide specific suggestions
                        4. Be professional and constructive
                        """
                    },
                    {
                        "name": "post_github_review",
                        "integration": "github",
                        "operation": "post_review",
                        "inputs": {
                            "review_comment": "{{steps.generate_review_comment.output}}",
                            "pr_number": "{{context.pr_number}}"
                        }
                    },
                    {
                        "name": "send_slack_notification",
                        "integration": "slack",
                        "operation": "send_message",
                        "inputs": {
                            "channel": "{{context.slack_channel}}",
                            "message": """
                            *Code Review Completed*
                            
                            PR: #{{context.pr_number}}
                            Analysis: {{steps.analyze_code.output}}
                            Review Comment: {{steps.generate_review_comment.output}}
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
                            **Code Review Completed**
                            
                            PR: #{{context.pr_number}}
                            Analysis: {{steps.analyze_code.output}}
                            Review Comment: {{steps.generate_review_comment.output}}
                            """
                        }
                    }
                ],
                "connections": [
                    {
                        "from": "analyze_code",
                        "to": "generate_review_comment"
                    },
                    {
                        "from": "generate_review_comment",
                        "to": "post_github_review"
                    },
                    {
                        "from": "analyze_code",
                        "to": ["send_slack_notification", "send_discord_notification"]
                    },
                    {
                        "from": "generate_review_comment",
                        "to": ["send_slack_notification", "send_discord_notification"]
                    }
                ]
            }
        }
    
    def _validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate the workflow context."""
        required_fields = [
            "code_changes",
            "pr_description",
            "pr_number",
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