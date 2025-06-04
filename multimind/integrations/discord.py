"""
Discord integration handler for MCP workflows.
"""

from typing import Dict, Any, Optional, List
import aiohttp
import logging
from datetime import datetime
from .base import IntegrationHandler, AsyncContextManager

logger = logging.getLogger(__name__)

class DiscordIntegrationHandler(IntegrationHandler, AsyncContextManager):
    """Handler for Discord integration operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Discord integration handler."""
        super().__init__(config)
        self.validate_config(["token"])
        
        self.token = config["token"]
        self.api_base = "https://discord.com/api/v10"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Set up aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bot {self.token}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up aiohttp session."""
        if self.session:
            await self.session.close()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Discord integration operation."""
        try:
            operation = inputs.get("operation", "send_message")
            
            if operation == "send_message":
                result = await self.send_message(inputs)
            elif operation == "create_channel":
                result = await self.create_channel(inputs)
            elif operation == "list_channels":
                result = await self.list_channels(inputs)
            elif operation == "create_role":
                result = await self.create_role(inputs)
            else:
                raise ValueError(f"Unsupported Discord operation: {operation}")
            
            self._update_metadata(success=True)
            return result
            
        except Exception as e:
            self._update_metadata(success=False)
            raise

    async def send_message(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to a Discord channel."""
        channel_id = inputs["channel_id"]
        content = inputs.get("content", "")
        embeds = inputs.get("embeds", [])
        
        payload = {}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds

        async with self.session.post(
            f"{self.api_base}/channels/{channel_id}/messages",
            json=payload
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                raise Exception(f"Failed to send Discord message: {result.get('message', 'Unknown error')}")
            
            return {
                "message_id": result["id"],
                "channel_id": result["channel_id"],
                "timestamp": result["timestamp"]
            }

    async def create_channel(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Discord channel."""
        guild_id = inputs["guild_id"]
        name = inputs["name"]
        channel_type = inputs.get("type", 0)  # 0 for text channel
        
        payload = {
            "name": name,
            "type": channel_type
        }
        
        if "topic" in inputs:
            payload["topic"] = inputs["topic"]
        if "parent_id" in inputs:
            payload["parent_id"] = inputs["parent_id"]

        async with self.session.post(
            f"{self.api_base}/guilds/{guild_id}/channels",
            json=payload
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                raise Exception(f"Failed to create Discord channel: {result.get('message', 'Unknown error')}")
            
            return {
                "channel_id": result["id"],
                "name": result["name"],
                "type": result["type"],
                "guild_id": result["guild_id"]
            }

    async def list_channels(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List channels in a Discord guild."""
        guild_id = inputs["guild_id"]
        
        async with self.session.get(
            f"{self.api_base}/guilds/{guild_id}/channels"
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                raise Exception(f"Failed to list Discord channels: {result.get('message', 'Unknown error')}")
            
            return {
                "channels": [
                    {
                        "id": channel["id"],
                        "name": channel["name"],
                        "type": channel["type"],
                        "position": channel["position"],
                        "parent_id": channel.get("parent_id")
                    }
                    for channel in result
                ]
            }

    async def create_role(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Discord role."""
        guild_id = inputs["guild_id"]
        name = inputs["name"]
        
        payload = {
            "name": name,
            "color": inputs.get("color", 0),
            "hoist": inputs.get("hoist", False),
            "mentionable": inputs.get("mentionable", False)
        }
        
        if "permissions" in inputs:
            payload["permissions"] = inputs["permissions"]

        async with self.session.post(
            f"{self.api_base}/guilds/{guild_id}/roles",
            json=payload
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                raise Exception(f"Failed to create Discord role: {result.get('message', 'Unknown error')}")
            
            return {
                "role_id": result["id"],
                "name": result["name"],
                "color": result["color"],
                "position": result["position"],
                "permissions": result["permissions"]
            } 