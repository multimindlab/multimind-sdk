"""
GitHub integration handler for MCP workflows.
"""

from typing import Dict, Any, Optional, List
import aiohttp
import logging
from datetime import datetime
from .base import IntegrationHandler, AsyncContextManager

logger = logging.getLogger(__name__)

class GitHubIntegrationHandler(IntegrationHandler, AsyncContextManager):
    """Handler for GitHub integration operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize GitHub integration handler."""
        super().__init__(config)
        self.validate_config(["token"])
        
        self.token = config["token"]
        self.api_base = "https://api.github.com"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Set up aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up aiohttp session."""
        if self.session:
            await self.session.close()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub integration operation."""
        try:
            operation = inputs.get("operation", "create_issue")
            
            if operation == "create_issue":
                result = await self.create_issue(inputs)
            elif operation == "create_pull_request":
                result = await self.create_pull_request(inputs)
            elif operation == "list_repositories":
                result = await self.list_repositories(inputs)
            elif operation == "create_repository":
                result = await self.create_repository(inputs)
            else:
                raise ValueError(f"Unsupported GitHub operation: {operation}")
            
            self._update_metadata(success=True)
            return result
            
        except Exception as e:
            self._update_metadata(success=False)
            raise

    async def create_issue(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new GitHub issue."""
        owner = inputs["owner"]
        repo = inputs["repo"]
        title = inputs["title"]
        body = inputs.get("body", "")
        
        payload = {
            "title": title,
            "body": body
        }
        
        if "labels" in inputs:
            payload["labels"] = inputs["labels"]
        if "assignees" in inputs:
            payload["assignees"] = inputs["assignees"]

        async with self.session.post(
            f"{self.api_base}/repos/{owner}/{repo}/issues",
            json=payload
        ) as response:
            result = await response.json()
            
            if response.status != 201:
                raise Exception(f"Failed to create GitHub issue: {result.get('message', 'Unknown error')}")
            
            return {
                "issue_number": result["number"],
                "html_url": result["html_url"],
                "state": result["state"]
            }

    async def create_pull_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new pull request."""
        owner = inputs["owner"]
        repo = inputs["repo"]
        title = inputs["title"]
        head = inputs["head"]
        base = inputs.get("base", "main")
        body = inputs.get("body", "")
        
        payload = {
            "title": title,
            "head": head,
            "base": base,
            "body": body
        }

        async with self.session.post(
            f"{self.api_base}/repos/{owner}/{repo}/pulls",
            json=payload
        ) as response:
            result = await response.json()
            
            if response.status != 201:
                raise Exception(f"Failed to create pull request: {result.get('message', 'Unknown error')}")
            
            return {
                "pr_number": result["number"],
                "html_url": result["html_url"],
                "state": result["state"]
            }

    async def list_repositories(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List repositories for a user or organization."""
        owner = inputs["owner"]
        repo_type = inputs.get("type", "all")
        sort = inputs.get("sort", "updated")
        direction = inputs.get("direction", "desc")
        
        params = {
            "type": repo_type,
            "sort": sort,
            "direction": direction
        }

        async with self.session.get(
            f"{self.api_base}/users/{owner}/repos",
            params=params
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                raise Exception(f"Failed to list repositories: {result.get('message', 'Unknown error')}")
            
            return {
                "repositories": [
                    {
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "description": repo["description"],
                        "html_url": repo["html_url"],
                        "stars": repo["stargazers_count"],
                        "forks": repo["forks_count"]
                    }
                    for repo in result
                ]
            }

    async def create_repository(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new repository."""
        name = inputs["name"]
        description = inputs.get("description", "")
        private = inputs.get("private", False)
        
        payload = {
            "name": name,
            "description": description,
            "private": private,
            "auto_init": inputs.get("auto_init", True)
        }

        async with self.session.post(
            f"{self.api_base}/user/repos",
            json=payload
        ) as response:
            result = await response.json()
            
            if response.status != 201:
                raise Exception(f"Failed to create repository: {result.get('message', 'Unknown error')}")
            
            return {
                "name": result["name"],
                "full_name": result["full_name"],
                "html_url": result["html_url"],
                "clone_url": result["clone_url"],
                "private": result["private"]
            } 