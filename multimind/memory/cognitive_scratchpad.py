"""
Cognitive scratchpad memory implementation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class CognitiveScratchpadMemory(BaseMemory):
    """Memory that implements cognitive scratchpad/chain-of-thought memory."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_items: int = 1000,
        max_steps: int = 100,
        step_retention_days: int = 30,
        enable_step_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_reasoning_tracking: bool = True,
        reasoning_threshold: float = 0.7,
        enable_optimization: bool = True,
        optimization_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_items = max_items
        self.max_steps = max_steps
        self.step_retention_days = step_retention_days
        self.enable_step_analysis = enable_step_analysis
        self.analysis_interval = analysis_interval
        self.enable_reasoning_tracking = enable_reasoning_tracking
        self.reasoning_threshold = reasoning_threshold
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        
        # Initialize storage
        self.items: List[Dict[str, Any]] = []
        self.reasoning_steps: List[Dict[str, Any]] = []  # Chain of thought steps
        self.reasoning_chains: Dict[str, List[Dict[str, Any]]] = {}  # chain_id -> chain data
        self.last_analysis = datetime.now()
        self.last_optimization = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and track reasoning steps."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "created_at": datetime.now().isoformat(),
                "modified_at": datetime.now().isoformat(),
                "step_count": 0,
                "chain_count": 0
            }
        }
        
        # Add to storage
        self.items.append(new_item)
        
        # Track reasoning steps
        await self._track_reasoning_steps(item_id, new_item)
        
        # Analyze steps if needed
        if self.enable_step_analysis and (
            datetime.now() - self.last_analysis
        ).total_seconds() >= self.analysis_interval:
            await self._analyze_steps()
        
        # Maintain item limit
        await self._maintain_item_limit()
        
        await self.save()

    async def _track_reasoning_steps(self, item_id: str, item: Dict[str, Any]) -> None:
        """Track reasoning steps for a new item."""
        try:
            # Generate reasoning steps prompt
            prompt = f"""
            Break down the reasoning process for this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. steps: list of strings (each step in the reasoning process)
            2. step_types: list of strings (type of each step)
            3. confidence: list of floats (confidence in each step)
            """
            response = await self.llm.generate(prompt)
            steps = json.loads(response)
            
            # Create reasoning steps
            chain_id = f"chain_{len(self.reasoning_chains)}"
            self.reasoning_chains[chain_id] = []
            
            for i, step in enumerate(steps["steps"]):
                reasoning_step = {
                    "id": f"step_{len(self.reasoning_steps)}",
                    "item_id": item_id,
                    "chain_id": chain_id,
                    "content": step,
                    "step_type": steps["step_types"][i],
                    "confidence": steps["confidence"][i],
                    "timestamp": datetime.now().isoformat()
                }
                self.reasoning_steps.append(reasoning_step)
                self.reasoning_chains[chain_id].append(reasoning_step)
            
            # Update item metadata
            item["metadata"]["step_count"] = len(steps["steps"])
            item["metadata"]["chain_count"] = 1
            
        except Exception as e:
            print(f"Error tracking reasoning steps: {e}")

    async def _analyze_steps(self) -> None:
        """Analyze reasoning steps."""
        # Group steps by chain
        chain_steps = {}
        for step in self.reasoning_steps:
            if step["chain_id"] not in chain_steps:
                chain_steps[step["chain_id"]] = []
            chain_steps[step["chain_id"]].append(step)
        
        # Analyze each chain
        for chain_id, steps in chain_steps.items():
            try:
                # Generate chain analysis prompt
                prompt = f"""
                Analyze this reasoning chain:
                
                {json.dumps(steps, indent=2)}
                
                Return a JSON object with:
                1. chain_quality: float (0-1)
                2. missing_steps: list of strings
                3. improvement_suggestions: list of strings
                """
                response = await self.llm.generate(prompt)
                analysis = json.loads(response)
                
                # Update chain metadata
                if chain_id in self.reasoning_chains:
                    self.reasoning_chains[chain_id].append({
                        "type": "chain_analysis",
                        "quality": analysis["chain_quality"],
                        "missing_steps": analysis["missing_steps"],
                        "suggestions": analysis["improvement_suggestions"],
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                print(f"Error analyzing steps: {e}")
        
        # Update last analysis time
        self.last_analysis = datetime.now()

    async def _maintain_item_limit(self) -> None:
        """Maintain item and step limits."""
        # Check item limit
        if len(self.items) > self.max_items:
            # Sort items by timestamp
            sorted_items = sorted(
                self.items,
                key=lambda x: datetime.fromisoformat(x["timestamp"])
            )
            
            # Remove oldest items
            items_to_remove = sorted_items[:len(self.items) - self.max_items]
            for item in items_to_remove:
                await self._remove_item(item["id"])
        
        # Check step limit
        if len(self.reasoning_steps) > self.max_steps:
            # Sort steps by timestamp
            sorted_steps = sorted(
                self.reasoning_steps,
                key=lambda x: datetime.fromisoformat(x["timestamp"])
            )
            
            # Remove oldest steps
            self.reasoning_steps = sorted_steps[len(self.reasoning_steps) - self.max_steps:]

    async def _remove_item(self, item_id: str) -> None:
        """Remove an item and its associated steps."""
        # Remove from items
        self.items = [i for i in self.items if i["id"] != item_id]
        
        # Remove associated steps
        self.reasoning_steps = [s for s in self.reasoning_steps if s["item_id"] != item_id]
        
        # Remove from chains
        for chain_id, chain_data in self.reasoning_chains.items():
            self.reasoning_chains[chain_id] = [
                s for s in chain_data if s["item_id"] != item_id
            ]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all items."""
        messages = []
        for item in self.items:
            messages.append({
                "role": "cognitive_scratchpad",
                "content": item["content"],
                "timestamp": item["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all items and steps."""
        self.items = []
        self.reasoning_steps = []
        self.reasoning_chains = {}
        await self.save()

    async def save(self) -> None:
        """Save items and steps to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "items": self.items,
                    "reasoning_steps": self.reasoning_steps,
                    "reasoning_chains": self.reasoning_chains,
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_optimization": self.last_optimization.isoformat()
                }, f)

    def load(self) -> None:
        """Load items and steps from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.reasoning_steps = data.get("reasoning_steps", [])
                self.reasoning_chains = data.get("reasoning_chains", {})
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )

    async def get_cognitive_scratchpad_stats(self) -> Dict[str, Any]:
        """Get statistics about cognitive scratchpad memory."""
        stats = {
            "total_items": len(self.items),
            "step_stats": {
                "total_steps": len(self.reasoning_steps),
                "step_types": len(set(s["step_type"] for s in self.reasoning_steps)),
                "average_steps_per_item": len(self.reasoning_steps) / len(self.items) if self.items else 0
            },
            "chain_stats": {
                "total_chains": len(self.reasoning_chains),
                "average_chain_length": sum(
                    len(chain) for chain in self.reasoning_chains.values()
                ) / len(self.reasoning_chains) if self.reasoning_chains else 0
            }
        }
        
        return stats

    async def get_cognitive_scratchpad_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for cognitive scratchpad memory optimization."""
        suggestions = []
        
        # Check item count
        if len(self.items) > self.max_items * 0.8:
            suggestions.append({
                "type": "item_limit",
                "suggestion": "Consider increasing max_items or removing older items"
            })
        
        # Check step count
        stats = await self.get_cognitive_scratchpad_stats()
        if stats["step_stats"]["total_steps"] > self.max_steps * 0.8:
            suggestions.append({
                "type": "step_limit",
                "suggestion": "Consider increasing max_steps or compressing steps"
            })
        
        # Check step coverage
        if stats["step_stats"]["average_steps_per_item"] < 2:
            suggestions.append({
                "type": "step_coverage",
                "suggestion": "Consider improving step tracking"
            })
        
        # Check chain quality
        if stats["chain_stats"]["average_chain_length"] < 2:
            suggestions.append({
                "type": "chain_quality",
                "suggestion": "Consider improving chain analysis"
            })
        
        return suggestions 