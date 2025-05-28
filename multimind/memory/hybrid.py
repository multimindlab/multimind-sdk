"""
Hybrid memory implementation that combines multiple memory types with intelligent routing.
"""

from typing import List, Dict, Any, Optional, Type, Set, Tuple
from datetime import datetime, timedelta
import json
import zlib
import base64
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory
from .vector_store import VectorStoreMemory
from .knowledge_graph import KnowledgeGraphMemory
from .time_weighted import TimeWeightedMemory
from .token_buffer import TokenBufferMemory
from .dnc import DNCMemory

class HybridMemory(BaseMemory):
    """Memory that combines multiple memory types with intelligent routing."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        memory_types: Optional[List[Type[BaseMemory]]] = None,
        routing_threshold: float = 0.7,
        max_memories: int = 4,
        sync_interval: int = 300,  # 5 minutes
        priority_weights: Optional[Dict[str, float]] = None,
        adaptive_routing: bool = True,
        compression_enabled: bool = True,
        compression_level: int = 6,
        backup_interval: int = 3600,  # 1 hour
        max_backups: int = 24,  # 24 hours of backups
        routing_strategy: str = "adaptive",  # adaptive, priority, or hybrid
        enable_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_optimization: bool = True,
        optimization_interval: int = 3600,  # 1 hour
        enable_learning: bool = True,
        learning_rate: float = 0.01,
        enable_metadata: bool = True,
        metadata_interval: int = 3600,  # 1 hour
        enable_cross_memory: bool = True,
        cross_memory_interval: int = 3600,  # 1 hour
        enable_consolidation: bool = True,
        consolidation_interval: int = 3600,  # 1 hour
        enable_validation: bool = True,
        validation_interval: int = 3600,  # 1 hour
        enable_evolution: bool = True,
        evolution_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.routing_threshold = routing_threshold
        self.max_memories = max_memories
        self.sync_interval = sync_interval
        self.adaptive_routing = adaptive_routing
        self.compression_enabled = compression_enabled
        self.compression_level = compression_level
        self.backup_interval = backup_interval
        self.max_backups = max_backups
        self.routing_strategy = routing_strategy
        self.enable_analysis = enable_analysis
        self.analysis_interval = analysis_interval
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.enable_metadata = enable_metadata
        self.metadata_interval = metadata_interval
        self.enable_cross_memory = enable_cross_memory
        self.cross_memory_interval = cross_memory_interval
        self.enable_consolidation = enable_consolidation
        self.consolidation_interval = consolidation_interval
        self.enable_validation = enable_validation
        self.validation_interval = validation_interval
        self.enable_evolution = enable_evolution
        self.evolution_interval = evolution_interval
        
        # Initialize default memory types if none provided
        self.memory_types = memory_types or [
            VectorStoreMemory,
            KnowledgeGraphMemory,
            TimeWeightedMemory,
            TokenBufferMemory,
            DNCMemory
        ]
        
        # Initialize memory instances and configurations
        self.memories: Dict[str, BaseMemory] = {}
        self.memory_configs: Dict[str, Dict[str, Any]] = {}
        self.priority_weights = priority_weights or {
            "VectorStoreMemory": 1.0,
            "KnowledgeGraphMemory": 1.0,
            "TimeWeightedMemory": 1.0,
            "TokenBufferMemory": 1.0,
            "DNCMemory": 1.0
        }
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}
        self.metadata_history: Dict[str, List[Dict[str, Any]]] = {}
        self.cross_memory_links: Dict[str, Dict[str, List[str]]] = {}
        self.consolidation_history: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Timestamps
        self.last_sync = datetime.now()
        self.last_backup = datetime.now()
        self.last_analysis = datetime.now()
        self.last_optimization = datetime.now()
        self.last_metadata = datetime.now()
        self.last_cross_memory = datetime.now()
        self.last_consolidation = datetime.now()
        self.last_validation = datetime.now()
        self.last_evolution = datetime.now()
        
        # Initialize memories
        self._initialize_memories()
        self.load()

    def _initialize_memories(self) -> None:
        """Initialize memory instances."""
        for memory_type in self.memory_types[:self.max_memories]:
            memory_name = memory_type.__name__
            memory_instance = memory_type(
                llm=self.llm,
                memory_key=f"{self.memory_key}_{memory_name}",
                storage_path=str(self.storage_path / f"{memory_name}.json") if self.storage_path else None
            )
            self.memories[memory_name] = memory_instance
            self.memory_configs[memory_name] = {
                "type": memory_name,
                "priority": self.priority_weights.get(memory_name, 1.0),
                "performance": {
                    "hits": 0,
                    "misses": 0,
                    "latency": 0.0
                }
            }

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message to appropriate memory types."""
        current_time = datetime.now()

        # Route message to appropriate memories
        routed_memories = await self._route_message(message)

        # Add to routed memories
        for memory_name in routed_memories:
            await self.memories[memory_name].add_message(message)
            self.memory_configs[memory_name]["performance"]["hits"] += 1

        # Update learning if enabled
        if self.enable_learning:
            await self._update_learning(message, routed_memories)

        # Check for analysis
        if self.enable_analysis:
            if (current_time - self.last_analysis).total_seconds() > self.analysis_interval:
                await self._analyze_memories()

        # Check for optimization
        if self.enable_optimization:
            if (current_time - self.last_optimization).total_seconds() > self.optimization_interval:
                await self._optimize_memories()

        # Check for metadata update
        if self.enable_metadata:
            if (current_time - self.last_metadata).total_seconds() > self.metadata_interval:
                await self._update_metadata()

        # Check for cross-memory analysis
        if self.enable_cross_memory:
            if (current_time - self.last_cross_memory).total_seconds() > self.cross_memory_interval:
                await self._analyze_cross_memory()

        # Check for consolidation
        if self.enable_consolidation:
            if (current_time - self.last_consolidation).total_seconds() > self.consolidation_interval:
                await self._consolidate_memories()

        # Check for validation
        if self.enable_validation:
            if (current_time - self.last_validation).total_seconds() > self.validation_interval:
                await self._validate_memories()

        # Check for evolution
        if self.enable_evolution:
            if (current_time - self.last_evolution).total_seconds() > self.evolution_interval:
                await self._evolve_memories()

        # Check for backup
        if (current_time - self.last_backup).total_seconds() > self.backup_interval:
            await self._create_backup()

        await self.save()

    async def _route_message(self, message: Dict[str, str]) -> List[str]:
        """Route message to appropriate memory types."""
        try:
            # Generate routing prompt
            prompt = f"""
            Route message to appropriate memory types:
            
            Message: {message['content']}
            
            Available memory types:
            {json.dumps(self.memory_configs, indent=2)}
            
            Return a JSON object with:
            1. selected_memories: list of memory type names
            2. routing_reason: string
            3. confidence: float
            """
            response = await self.llm.generate(prompt)
            routing = json.loads(response)

            # Record routing decision
            self.routing_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "selected_memories": routing["selected_memories"],
                "routing_reason": routing["routing_reason"],
                "confidence": routing["confidence"]
            })

            return routing["selected_memories"]

        except Exception as e:
            print(f"Error routing message: {e}")
            return list(self.memories.keys())

    async def _update_learning(self, message: Dict[str, str], routed_memories: List[str]) -> None:
        """Update learning based on routing performance."""
        try:
            # Generate learning prompt
            prompt = f"""
            Analyze routing performance:
            
            Message: {message['content']}
            Routed to: {routed_memories}
            
            Memory configurations:
            {json.dumps(self.memory_configs, indent=2)}
            
            Return a JSON object with:
            1. learning_updates: dict of memory_name -> update_data
            2. learning_reason: string
            """
            response = await self.llm.generate(prompt)
            learning = json.loads(response)

            # Update learning history
            for memory_name, update_data in learning["learning_updates"].items():
                if memory_name not in self.learning_history:
                    self.learning_history[memory_name] = []
                self.learning_history[memory_name].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "update_data": update_data,
                    "learning_reason": learning["learning_reason"]
                })

            # Update memory configurations
            for memory_name, update_data in learning["learning_updates"].items():
                if memory_name in self.memory_configs:
                    self.memory_configs[memory_name].update(update_data)

        except Exception as e:
            print(f"Error updating learning: {e}")

    async def _analyze_memories(self) -> None:
        """Analyze memory performance and patterns."""
        try:
            # Generate analysis prompt
            prompt = f"""
            Analyze memory performance:
            
            Memory configurations:
            {json.dumps(self.memory_configs, indent=2)}
            
            Routing history:
            {json.dumps(self.routing_history[-10:], indent=2)}
            
            Return a JSON object with:
            1. analysis: dict of string -> any
            2. suggestions: list of string
            3. metrics: dict of string -> float
            """
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)

            # Record analysis
            self.performance_metrics["analysis"] = {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis["analysis"],
                "suggestions": analysis["suggestions"],
                "metrics": analysis["metrics"]
            }

            self.last_analysis = datetime.now()

        except Exception as e:
            print(f"Error analyzing memories: {e}")

    async def _optimize_memories(self) -> None:
        """Optimize memory configurations."""
        try:
            # Generate optimization prompt
            prompt = f"""
            Optimize memory configurations:
            
            Current configurations:
            {json.dumps(self.memory_configs, indent=2)}
            
            Performance metrics:
            {json.dumps(self.performance_metrics, indent=2)}
            
            Return a JSON object with:
            1. optimizations: dict of memory_name -> optimization_data
            2. optimization_reason: string
            """
            response = await self.llm.generate(prompt)
            optimization = json.loads(response)

            # Apply optimizations
            for memory_name, optimization_data in optimization["optimizations"].items():
                if memory_name in self.memory_configs:
                    self.memory_configs[memory_name].update(optimization_data)

            self.last_optimization = datetime.now()

        except Exception as e:
            print(f"Error optimizing memories: {e}")

    async def _update_metadata(self) -> None:
        """Update memory metadata."""
        try:
            # Generate metadata prompt
            prompt = f"""
            Update memory metadata:
            
            Current metadata:
            {json.dumps(self.metadata_history, indent=2)}
            
            Memory configurations:
            {json.dumps(self.memory_configs, indent=2)}
            
            Return a JSON object with:
            1. metadata_updates: dict of memory_name -> metadata
            2. update_reason: string
            """
            response = await self.llm.generate(prompt)
            metadata = json.loads(response)

            # Update metadata history
            for memory_name, metadata_update in metadata["metadata_updates"].items():
                if memory_name not in self.metadata_history:
                    self.metadata_history[memory_name] = []
                self.metadata_history[memory_name].append({
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata_update,
                    "update_reason": metadata["update_reason"]
                })

            self.last_metadata = datetime.now()

        except Exception as e:
            print(f"Error updating metadata: {e}")

    async def _analyze_cross_memory(self) -> None:
        """Analyze relationships between different memory types."""
        try:
            # Generate cross-memory analysis prompt
            prompt = f"""
            Analyze cross-memory relationships:
            
            Memory configurations:
            {json.dumps(self.memory_configs, indent=2)}
            
            Cross-memory links:
            {json.dumps(self.cross_memory_links, indent=2)}
            
            Return a JSON object with:
            1. relationships: dict of memory_pair -> relationship_data
            2. analysis_reason: string
            """
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)

            # Update cross-memory links
            for memory_pair, relationship_data in analysis["relationships"].items():
                memory1, memory2 = memory_pair.split("_")
                if memory1 not in self.cross_memory_links:
                    self.cross_memory_links[memory1] = {}
                if memory2 not in self.cross_memory_links[memory1]:
                    self.cross_memory_links[memory1][memory2] = []
                self.cross_memory_links[memory1][memory2].append({
                    "timestamp": datetime.now().isoformat(),
                    "relationship_data": relationship_data,
                    "analysis_reason": analysis["analysis_reason"]
                })

            self.last_cross_memory = datetime.now()

        except Exception as e:
            print(f"Error analyzing cross-memory: {e}")

    async def _consolidate_memories(self) -> None:
        """Consolidate memory contents."""
        try:
            # Generate consolidation prompt
            prompt = f"""
            Consolidate memory contents:
            
            Memory configurations:
            {json.dumps(self.memory_configs, indent=2)}
            
            Consolidation history:
            {json.dumps(self.consolidation_history[-5:], indent=2)}
            
            Return a JSON object with:
            1. consolidation_plan: dict of memory_name -> consolidation_data
            2. consolidation_reason: string
            """
            response = await self.llm.generate(prompt)
            consolidation = json.loads(response)

            # Record consolidation
            self.consolidation_history.append({
                "timestamp": datetime.now().isoformat(),
                "consolidation_plan": consolidation["consolidation_plan"],
                "consolidation_reason": consolidation["consolidation_reason"]
            })

            self.last_consolidation = datetime.now()

        except Exception as e:
            print(f"Error consolidating memories: {e}")

    async def _validate_memories(self) -> None:
        """Validate memory contents and configurations."""
        try:
            # Generate validation prompt
            prompt = f"""
            Validate memory contents:
            
            Memory configurations:
            {json.dumps(self.memory_configs, indent=2)}
            
            Validation history:
            {json.dumps(self.validation_history[-5:], indent=2)}
            
            Return a JSON object with:
            1. validation_results: dict of memory_name -> validation_data
            2. validation_reason: string
            """
            response = await self.llm.generate(prompt)
            validation = json.loads(response)

            # Record validation
            self.validation_history.append({
                "timestamp": datetime.now().isoformat(),
                "validation_results": validation["validation_results"],
                "validation_reason": validation["validation_reason"]
            })

            self.last_validation = datetime.now()

        except Exception as e:
            print(f"Error validating memories: {e}")

    async def _evolve_memories(self) -> None:
        """Evolve memory configurations and relationships."""
        try:
            # Generate evolution prompt
            prompt = f"""
            Evolve memory system:
            
            Current state:
            {json.dumps(self.memory_configs, indent=2)}
            
            Evolution history:
            {json.dumps(self.evolution_history[-5:], indent=2)}
            
            Return a JSON object with:
            1. evolution_plan: dict of memory_name -> evolution_data
            2. evolution_reason: string
            """
            response = await self.llm.generate(prompt)
            evolution = json.loads(response)

            # Record evolution
            self.evolution_history.append({
                "timestamp": datetime.now().isoformat(),
                "evolution_plan": evolution["evolution_plan"],
                "evolution_reason": evolution["evolution_reason"]
            })

            self.last_evolution = datetime.now()

        except Exception as e:
            print(f"Error evolving memories: {e}")

    async def _create_backup(self) -> None:
        """Create backup of memory state."""
        try:
            backup = {
                "timestamp": datetime.now().isoformat(),
                "memory_configs": self.memory_configs,
                "performance_metrics": self.performance_metrics,
                "routing_history": self.routing_history,
                "learning_history": self.learning_history,
                "metadata_history": self.metadata_history,
                "cross_memory_links": self.cross_memory_links,
                "consolidation_history": self.consolidation_history,
                "validation_history": self.validation_history,
                "evolution_history": self.evolution_history
            }

            # Compress backup if enabled
            if self.compression_enabled:
                backup_str = json.dumps(backup)
                compressed = zlib.compress(backup_str.encode(), level=self.compression_level)
                backup = {
                    "compressed": True,
                    "data": base64.b64encode(compressed).decode()
                }

            # Save backup
            if self.storage_path:
                backup_path = self.storage_path / f"backup_{datetime.now().isoformat()}.json"
                with open(backup_path, 'w') as f:
                    json.dump(backup, f)

            self.last_backup = datetime.now()

        except Exception as e:
            print(f"Error creating backup: {e}")

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all memory types."""
        all_messages = []
        for memory in self.memories.values():
            all_messages.extend(memory.get_messages())
        return sorted(all_messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all memories."""
        for memory in self.memories.values():
            await memory.clear()
        self.memory_configs = {}
        self.performance_metrics = {}
        self.routing_history = []
        self.learning_history = {}
        self.metadata_history = {}
        self.cross_memory_links = {}
        self.consolidation_history = []
        self.validation_history = []
        self.evolution_history = []
        await self.save()

    async def save(self) -> None:
        """Save memory state to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path / "hybrid_memory.json", 'w') as f:
                json.dump({
                    "memory_configs": self.memory_configs,
                    "performance_metrics": self.performance_metrics,
                    "routing_history": self.routing_history,
                    "learning_history": self.learning_history,
                    "metadata_history": self.metadata_history,
                    "cross_memory_links": self.cross_memory_links,
                    "consolidation_history": self.consolidation_history,
                    "validation_history": self.validation_history,
                    "evolution_history": self.evolution_history,
                    "last_sync": self.last_sync.isoformat(),
                    "last_backup": self.last_backup.isoformat(),
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_optimization": self.last_optimization.isoformat(),
                    "last_metadata": self.last_metadata.isoformat(),
                    "last_cross_memory": self.last_cross_memory.isoformat(),
                    "last_consolidation": self.last_consolidation.isoformat(),
                    "last_validation": self.last_validation.isoformat(),
                    "last_evolution": self.last_evolution.isoformat()
                }, f)

    def load(self) -> None:
        """Load memory state from persistent storage."""
        if self.storage_path and (self.storage_path / "hybrid_memory.json").exists():
            with open(self.storage_path / "hybrid_memory.json", 'r') as f:
                data = json.load(f)
                self.memory_configs = data.get("memory_configs", {})
                self.performance_metrics = data.get("performance_metrics", {})
                self.routing_history = data.get("routing_history", [])
                self.learning_history = data.get("learning_history", {})
                self.metadata_history = data.get("metadata_history", {})
                self.cross_memory_links = data.get("cross_memory_links", {})
                self.consolidation_history = data.get("consolidation_history", [])
                self.validation_history = data.get("validation_history", [])
                self.evolution_history = data.get("evolution_history", [])
                self.last_sync = datetime.fromisoformat(
                    data.get("last_sync", datetime.now().isoformat())
                )
                self.last_backup = datetime.fromisoformat(
                    data.get("last_backup", datetime.now().isoformat())
                )
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )
                self.last_metadata = datetime.fromisoformat(
                    data.get("last_metadata", datetime.now().isoformat())
                )
                self.last_cross_memory = datetime.fromisoformat(
                    data.get("last_cross_memory", datetime.now().isoformat())
                )
                self.last_consolidation = datetime.fromisoformat(
                    data.get("last_consolidation", datetime.now().isoformat())
                )
                self.last_validation = datetime.fromisoformat(
                    data.get("last_validation", datetime.now().isoformat())
                )
                self.last_evolution = datetime.fromisoformat(
                    data.get("last_evolution", datetime.now().isoformat())
                )

    async def get_hybrid_stats(self) -> Dict[str, Any]:
        """Get statistics about hybrid memory."""
        stats = {
            "memory_stats": {
                "total_memories": len(self.memories),
                "memory_types": list(self.memories.keys()),
                "total_messages": sum(
                    len(memory.get_messages())
                    for memory in self.memories.values()
                )
            },
            "routing_stats": {
                "total_routes": len(self.routing_history),
                "routing_strategy": self.routing_strategy,
                "adaptive_routing": self.adaptive_routing
            },
            "performance_stats": {
                "total_hits": sum(
                    config["performance"]["hits"]
                    for config in self.memory_configs.values()
                ),
                "total_misses": sum(
                    config["performance"]["misses"]
                    for config in self.memory_configs.values()
                ),
                "average_latency": sum(
                    config["performance"]["latency"]
                    for config in self.memory_configs.values()
                ) / len(self.memory_configs) if self.memory_configs else 0
            },
            "learning_stats": {
                "total_learning_records": sum(
                    len(records)
                    for records in self.learning_history.values()
                ),
                "learning_enabled": self.enable_learning,
                "learning_rate": self.learning_rate
            },
            "optimization_stats": {
                "total_optimizations": len(self.performance_metrics.get("analysis", [])),
                "optimization_enabled": self.enable_optimization,
                "optimization_interval": self.optimization_interval
            },
            "cross_memory_stats": {
                "total_links": sum(
                    len(links)
                    for memory_links in self.cross_memory_links.values()
                    for links in memory_links.values()
                ),
                "cross_memory_enabled": self.enable_cross_memory,
                "cross_memory_interval": self.cross_memory_interval
            },
            "consolidation_stats": {
                "total_consolidations": len(self.consolidation_history),
                "consolidation_enabled": self.enable_consolidation,
                "consolidation_interval": self.consolidation_interval
            },
            "validation_stats": {
                "total_validations": len(self.validation_history),
                "validation_enabled": self.enable_validation,
                "validation_interval": self.validation_interval
            },
            "evolution_stats": {
                "total_evolutions": len(self.evolution_history),
                "evolution_enabled": self.enable_evolution,
                "evolution_interval": self.evolution_interval
            }
        }
        return stats

    async def get_hybrid_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for hybrid memory optimization."""
        suggestions = []
        
        # Check memory count
        if len(self.memories) > self.max_memories:
            suggestions.append({
                "type": "memory_count",
                "suggestion": "Consider reducing number of memory types or increasing max_memories"
            })
        
        # Check routing performance
        if self.routing_history:
            hit_rate = sum(
                1 for route in self.routing_history
                if len(route["selected_memories"]) > 0
            ) / len(self.routing_history)
            if hit_rate < 0.7:
                suggestions.append({
                    "type": "routing_performance",
                    "suggestion": "Consider adjusting routing strategy or threshold"
                })
        
        # Check learning progress
        if self.learning_history:
            avg_learning = sum(
                len(records)
                for records in self.learning_history.values()
            ) / len(self.learning_history)
            if avg_learning < 10:
                suggestions.append({
                    "type": "learning_rate",
                    "suggestion": "Consider increasing learning rate or improving learning mechanisms"
                })
        
        # Check optimization frequency
        if len(self.performance_metrics.get("analysis", [])) < 2:
            suggestions.append({
                "type": "optimization_frequency",
                "suggestion": "Consider adjusting optimization interval"
            })
        
        # Check cross-memory coverage
        if self.cross_memory_links:
            coverage = len(self.cross_memory_links) / (len(self.memories) * (len(self.memories) - 1) / 2)
            if coverage < 0.5:
                suggestions.append({
                    "type": "cross_memory_coverage",
                    "suggestion": "Consider improving cross-memory analysis"
                })
        
        # Check consolidation frequency
        if len(self.consolidation_history) < 2:
            suggestions.append({
                "type": "consolidation_frequency",
                "suggestion": "Consider adjusting consolidation interval"
            })
        
        # Check validation coverage
        if len(self.validation_history) < 2:
            suggestions.append({
                "type": "validation_frequency",
                "suggestion": "Consider adjusting validation interval"
            })
        
        # Check evolution progress
        if len(self.evolution_history) < 2:
            suggestions.append({
                "type": "evolution_frequency",
                "suggestion": "Consider adjusting evolution interval"
            })
        
        return suggestions 