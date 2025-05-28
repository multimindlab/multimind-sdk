"""
Procedural memory implementation that stores and retrieves procedural knowledge with step-by-step instructions.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class ProceduralMemory(BaseMemory):
    """Memory that stores and retrieves procedural knowledge with step-by-step instructions."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_procedures: int = 1000,
        similarity_threshold: float = 0.7,
        execution_tracking: bool = True,
        success_threshold: float = 0.8,
        min_confidence: float = 0.6,
        enable_optimization: bool = True,
        optimization_interval: int = 3600,  # 1 hour
        enable_validation: bool = True,
        validation_interval: int = 3600,  # 1 hour
        enable_adaptation: bool = True,
        adaptation_rate: float = 0.1,
        enable_chaining: bool = True,
        chain_depth: int = 3,
        enable_monitoring: bool = True,
        monitoring_interval: int = 300,  # 5 minutes
        enable_learning: bool = True,
        learning_rate: float = 0.1
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_procedures = max_procedures
        self.similarity_threshold = similarity_threshold
        self.execution_tracking = execution_tracking
        self.success_threshold = success_threshold
        self.min_confidence = min_confidence
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        self.enable_validation = enable_validation
        self.validation_interval = validation_interval
        self.enable_adaptation = enable_adaptation
        self.adaptation_rate = adaptation_rate
        self.enable_chaining = enable_chaining
        self.chain_depth = chain_depth
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        
        # Initialize procedure storage
        self.procedures: List[Dict[str, Any]] = []
        self.procedure_embeddings: List[List[float]] = []
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}  # procedure_id -> execution records
        self.procedure_weights: Dict[str, float] = {}  # procedure_id -> weight
        self.procedure_metadata: Dict[str, Dict[str, Any]] = {}  # procedure_id -> metadata
        self.optimization_cache: Dict[str, List[Dict[str, Any]]] = {}  # procedure_id -> optimization suggestions
        self.procedure_chains: Dict[str, List[str]] = {}  # procedure_id -> chain of related procedures
        self.monitoring_metrics: Dict[str, Dict[str, Any]] = {}  # procedure_id -> monitoring metrics
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}  # procedure_id -> learning records
        self.last_optimization = datetime.now()
        self.last_validation = datetime.now()
        self.last_monitoring = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message as new procedural knowledge."""
        # Extract procedure from message
        procedure = await self._extract_procedure(message["content"])
        
        if procedure:
            # Create procedure
            procedure_id = f"proc_{len(self.procedures)}"
            new_procedure = {
                "id": procedure_id,
                "content": procedure["content"],
                "timestamp": datetime.now().isoformat(),
                "steps": procedure["steps"],
                "metadata": {
                    "category": procedure["category"],
                    "prerequisites": procedure["prerequisites"],
                    "expected_outcome": procedure["expected_outcome"],
                    "confidence": procedure["confidence"],
                    "validated": False,
                    "optimized": False,
                    "success_rate": 0.0,
                    "execution_count": 0,
                    "average_duration": 0.0,
                    "chain_position": 0,
                    "learning_progress": 0.0
                }
            }
            
            # Add to storage
            self.procedures.append(new_procedure)
            self.procedure_weights[procedure_id] = 1.0
            self.procedure_metadata[procedure_id] = new_procedure["metadata"]
            
            # Get procedure embedding
            embedding = await self.llm.embeddings(procedure["content"])
            self.procedure_embeddings.append(embedding)
            
            # Initialize execution history and chains
            self.execution_history[procedure_id] = []
            self.procedure_chains[procedure_id] = []
            self.monitoring_metrics[procedure_id] = {
                "performance": 0.0,
                "reliability": 1.0,
                "efficiency": 1.0,
                "complexity": len(procedure["steps"])
            }
            self.learning_history[procedure_id] = []
            
            # Update procedure chains
            if self.enable_chaining:
                await self._update_procedure_chains(procedure_id)
            
            # Check for optimization
            if self.enable_optimization:
                current_time = datetime.now()
                if (current_time - self.last_optimization).total_seconds() > self.optimization_interval:
                    await self._optimize_procedures()
            
            # Check for validation
            if self.enable_validation:
                current_time = datetime.now()
                if (current_time - self.last_validation).total_seconds() > self.validation_interval:
                    await self._validate_procedures()
            
            # Check for monitoring
            if self.enable_monitoring:
                current_time = datetime.now()
                if (current_time - self.last_monitoring).total_seconds() > self.monitoring_interval:
                    await self._monitor_procedures()
            
            # Maintain procedure limit
            await self._maintain_procedure_limit()
            
            await self.save()

    async def _extract_procedure(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract procedure and its steps from content."""
        try:
            prompt = f"""
            Extract a procedure and its steps from the following content:
            
            Content: {content}
            
            Determine:
            1. Procedure content
            2. Category
            3. Prerequisites
            4. Expected outcome
            5. Step-by-step instructions
            6. Confidence in extraction (0-1)
            
            Return in format:
            Procedure: <procedure content>
            Category: <category>
            Prerequisites: <comma-separated prerequisites>
            Expected Outcome: <expected outcome>
            Steps:
            1. <step 1>
            2. <step 2>
            ...
            Confidence: <confidence score>
            """
            response = await self.llm.generate(prompt)
            
            procedure = {
                "content": None,
                "category": None,
                "prerequisites": set(),
                "expected_outcome": None,
                "steps": [],
                "confidence": 1.0
            }
            
            current_step = None
            for line in response.split('\n'):
                if line.startswith('Procedure:'):
                    procedure["content"] = line.split(':', 1)[1].strip()
                elif line.startswith('Category:'):
                    procedure["category"] = line.split(':', 1)[1].strip()
                elif line.startswith('Prerequisites:'):
                    prerequisites = line.split(':', 1)[1].strip().split(',')
                    procedure["prerequisites"] = {p.strip() for p in prerequisites}
                elif line.startswith('Expected Outcome:'):
                    procedure["expected_outcome"] = line.split(':', 1)[1].strip()
                elif line.startswith('Steps:'):
                    continue
                elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    step = line.split('.', 1)[1].strip()
                    procedure["steps"].append(step)
                elif line.startswith('Confidence:'):
                    confidence = float(line.split(':', 1)[1].strip())
                    procedure["confidence"] = confidence
            
            if procedure["content"] and procedure["steps"]:
                return procedure
            
            return None
            
        except Exception as e:
            print(f"Error extracting procedure: {e}")
            return None

    async def record_execution(
        self,
        procedure_id: str,
        success: bool,
        duration: float,
        notes: Optional[str] = None
    ) -> None:
        """Record the execution of a procedure."""
        if procedure_id not in self.execution_history:
            return
        
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "duration": duration,
            "notes": notes
        }
        
        self.execution_history[procedure_id].append(execution_record)
        
        # Update procedure metadata
        metadata = self.procedure_metadata[procedure_id]
        metadata["execution_count"] += 1
        
        # Update success rate
        success_count = sum(1 for record in self.execution_history[procedure_id] if record["success"])
        metadata["success_rate"] = success_count / metadata["execution_count"]
        
        # Update average duration
        total_duration = sum(record["duration"] for record in self.execution_history[procedure_id])
        metadata["average_duration"] = total_duration / metadata["execution_count"]
        
        # Adapt procedure if enabled
        if self.enable_adaptation and not success:
            await self._adapt_procedure(procedure_id, execution_record)
        
        await self.save()

    async def _adapt_procedure(
        self,
        procedure_id: str,
        execution_record: Dict[str, Any]
    ) -> None:
        """Adapt procedure based on execution failure."""
        try:
            procedure = next(p for p in self.procedures if p["id"] == procedure_id)
            
            prompt = f"""
            Adapt this procedure based on the failed execution:
            
            Procedure: {procedure['content']}
            Steps:
            {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(procedure['steps']))}
            
            Failed Execution:
            Duration: {execution_record['duration']}
            Notes: {execution_record['notes']}
            
            Return adapted steps in format:
            Steps:
            1. <adapted step 1>
            2. <adapted step 2>
            ...
            """
            response = await self.llm.generate(prompt)
            
            # Parse adapted steps
            adapted_steps = []
            for line in response.split('\n'):
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    step = line.split('.', 1)[1].strip()
                    adapted_steps.append(step)
            
            if adapted_steps:
                # Blend original and adapted steps
                for i, (original, adapted) in enumerate(zip(procedure["steps"], adapted_steps)):
                    procedure["steps"][i] = f"{original} (Adapted: {adapted})"
            
        except Exception as e:
            print(f"Error adapting procedure: {e}")

    async def _optimize_procedures(self) -> None:
        """Optimize procedures based on execution history."""
        for procedure in self.procedures:
            if procedure["metadata"]["optimized"]:
                continue
            
            try:
                # Generate optimization prompt
                prompt = f"""
                Optimize this procedure based on its execution history:
                
                Procedure: {procedure['content']}
                Steps:
                {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(procedure['steps']))}
                
                Execution History:
                Success Rate: {procedure['metadata']['success_rate']}
                Average Duration: {procedure['metadata']['average_duration']}
                Total Executions: {procedure['metadata']['execution_count']}
                
                Return optimized steps in format:
                Steps:
                1. <optimized step 1>
                2. <optimized step 2>
                ...
                """
                response = await self.llm.generate(prompt)
                
                # Parse optimized steps
                optimized_steps = []
                for line in response.split('\n'):
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        step = line.split('.', 1)[1].strip()
                        optimized_steps.append(step)
                
                if optimized_steps:
                    procedure["steps"] = optimized_steps
                    procedure["metadata"]["optimized"] = True
                
            except Exception as e:
                print(f"Error optimizing procedure: {e}")
        
        self.last_optimization = datetime.now()

    async def _validate_procedures(self) -> None:
        """Validate procedures and their steps."""
        for procedure in self.procedures:
            if procedure["metadata"]["validated"]:
                continue
            
            try:
                # Generate validation prompt
                prompt = f"""
                Validate this procedure and its steps:
                
                Procedure: {procedure['content']}
                Category: {procedure['metadata']['category']}
                Prerequisites: {procedure['metadata']['prerequisites']}
                Expected Outcome: {procedure['metadata']['expected_outcome']}
                Steps:
                {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(procedure['steps']))}
                
                Return validation results in format:
                Valid: <true/false>
                Confidence: <confidence score>
                Issues: <comma-separated issues>
                """
                response = await self.llm.generate(prompt)
                
                # Parse validation results
                lines = response.split('\n')
                for line in lines:
                    if line.startswith('Valid:'):
                        is_valid = line.split(':', 1)[1].strip().lower() == 'true'
                    elif line.startswith('Confidence:'):
                        confidence = float(line.split(':', 1)[1].strip())
                    elif line.startswith('Issues:'):
                        issues = line.split(':', 1)[1].strip().split(',')
                
                if is_valid and confidence >= self.min_confidence:
                    procedure["metadata"]["validated"] = True
                    procedure["metadata"]["confidence"] = confidence
                else:
                    # Remove invalid procedure
                    await self._remove_procedure(procedure["id"])
            
            except Exception as e:
                print(f"Error validating procedure: {e}")
        
        self.last_validation = datetime.now()

    async def _maintain_procedure_limit(self) -> None:
        """Maintain procedure limit by removing least important procedures."""
        if len(self.procedures) > self.max_procedures:
            # Sort procedures by weight
            sorted_procedures = sorted(
                self.procedures,
                key=lambda x: self.procedure_weights[x["id"]]
            )
            
            # Remove procedures with lowest weights
            procedures_to_remove = sorted_procedures[:len(self.procedures) - self.max_procedures]
            for procedure in procedures_to_remove:
                await self._remove_procedure(procedure["id"])

    async def _remove_procedure(self, procedure_id: str) -> None:
        """Remove a procedure and its execution history."""
        # Remove from procedures
        procedure_idx = next(i for i, p in enumerate(self.procedures) if p["id"] == procedure_id)
        self.procedures.pop(procedure_idx)
        self.procedure_embeddings.pop(procedure_idx)
        
        # Remove execution history
        if procedure_id in self.execution_history:
            del self.execution_history[procedure_id]
        
        # Remove metadata and weights
        del self.procedure_metadata[procedure_id]
        del self.procedure_weights[procedure_id]
        
        # Remove from optimization cache
        if procedure_id in self.optimization_cache:
            del self.optimization_cache[procedure_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all procedures."""
        messages = []
        for procedure in self.procedures:
            messages.append({
                "role": "procedure",
                "content": procedure["content"],
                "timestamp": procedure["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all procedures."""
        self.procedures = []
        self.procedure_embeddings = []
        self.execution_history = {}
        self.procedure_weights = {}
        self.procedure_metadata = {}
        self.optimization_cache = {}
        self.procedure_chains = {}
        self.monitoring_metrics = {}
        self.learning_history = {}
        await self.save()

    async def save(self) -> None:
        """Save procedures to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "procedures": self.procedures,
                    "execution_history": self.execution_history,
                    "procedure_weights": self.procedure_weights,
                    "procedure_metadata": {
                        k: {
                            **v,
                            "prerequisites": list(v["prerequisites"])
                        }
                        for k, v in self.procedure_metadata.items()
                    },
                    "optimization_cache": self.optimization_cache,
                    "procedure_chains": self.procedure_chains,
                    "monitoring_metrics": self.monitoring_metrics,
                    "learning_history": self.learning_history,
                    "last_optimization": self.last_optimization.isoformat(),
                    "last_validation": self.last_validation.isoformat(),
                    "last_monitoring": self.last_monitoring.isoformat()
                }, f)

    def load(self) -> None:
        """Load procedures from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.procedures = data.get("procedures", [])
                self.execution_history = data.get("execution_history", {})
                self.procedure_weights = data.get("procedure_weights", {})
                self.procedure_metadata = {
                    k: {
                        **v,
                        "prerequisites": set(v["prerequisites"])
                    }
                    for k, v in data.get("procedure_metadata", {}).items()
                }
                self.optimization_cache = data.get("optimization_cache", {})
                self.procedure_chains = data.get("procedure_chains", {})
                self.monitoring_metrics = data.get("monitoring_metrics", {})
                self.learning_history = data.get("learning_history", {})
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )
                self.last_validation = datetime.fromisoformat(
                    data.get("last_validation", datetime.now().isoformat())
                )
                self.last_monitoring = datetime.fromisoformat(
                    data.get("last_monitoring", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.procedure_embeddings = []
                for procedure in self.procedures:
                    self.procedure_embeddings.append(
                        self.llm.embeddings(procedure["content"])
                    )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    async def get_procedure_by_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Get a procedure by its ID."""
        try:
            return next(p for p in self.procedures if p["id"] == procedure_id)
        except StopIteration:
            return None

    async def get_execution_history(
        self,
        procedure_id: str,
        min_success_rate: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get execution history of a procedure."""
        if procedure_id not in self.execution_history:
            return []
        
        if min_success_rate is None:
            return self.execution_history[procedure_id]
        
        return [
            record for record in self.execution_history[procedure_id]
            if record["success"] >= min_success_rate
        ]

    async def get_procedure_stats(self) -> Dict[str, Any]:
        """Get statistics about procedures."""
        stats = {
            "total_procedures": len(self.procedures),
            "category_distribution": {},
            "success_rate_distribution": {
                "high": 0,  # > 0.8
                "medium": 0,  # 0.5-0.8
                "low": 0  # < 0.5
            },
            "execution_stats": {
                "total_executions": sum(
                    metadata["execution_count"]
                    for metadata in self.procedure_metadata.values()
                ),
                "average_duration": sum(
                    metadata["average_duration"]
                    for metadata in self.procedure_metadata.values()
                ) / len(self.procedure_metadata) if self.procedure_metadata else 0
            },
            "validation_stats": {
                "validated": 0,
                "unvalidated": 0
            },
            "optimization_stats": {
                "optimized": 0,
                "unoptimized": 0
            }
        }
        
        for procedure in self.procedures:
            # Count categories
            category = procedure["metadata"]["category"]
            if category:
                stats["category_distribution"][category] = \
                    stats["category_distribution"].get(category, 0) + 1
            
            # Count success rates
            success_rate = procedure["metadata"]["success_rate"]
            if success_rate > 0.8:
                stats["success_rate_distribution"]["high"] += 1
            elif success_rate > 0.5:
                stats["success_rate_distribution"]["medium"] += 1
            else:
                stats["success_rate_distribution"]["low"] += 1
            
            # Count validation status
            if procedure["metadata"]["validated"]:
                stats["validation_stats"]["validated"] += 1
            else:
                stats["validation_stats"]["unvalidated"] += 1
            
            # Count optimization status
            if procedure["metadata"]["optimized"]:
                stats["optimization_stats"]["optimized"] += 1
            else:
                stats["optimization_stats"]["unoptimized"] += 1
        
        return stats

    async def get_procedure_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for procedure optimization."""
        suggestions = []
        
        # Check procedure count
        if len(self.procedures) > self.max_procedures * 0.8:
            suggestions.append({
                "type": "procedure_limit",
                "suggestion": "Consider increasing max_procedures or removing less important procedures"
            })
        
        # Check success rate distribution
        stats = await self.get_procedure_stats()
        if stats["success_rate_distribution"]["low"] > len(self.procedures) * 0.3:
            suggestions.append({
                "type": "success_rate",
                "suggestion": "Consider improving procedure success rates"
            })
        
        # Check validation status
        if stats["validation_stats"]["unvalidated"] > len(self.procedures) * 0.5:
            suggestions.append({
                "type": "validation",
                "suggestion": "Consider running procedure validation"
            })
        
        # Check optimization status
        if stats["optimization_stats"]["unoptimized"] > len(self.procedures) * 0.5:
            suggestions.append({
                "type": "optimization",
                "suggestion": "Consider running procedure optimization"
            })
        
        # Check execution coverage
        if stats["execution_stats"]["total_executions"] < len(self.procedures) * 5:
            suggestions.append({
                "type": "execution_coverage",
                "suggestion": "Consider executing more procedures for better optimization"
            })
        
        return suggestions

    async def _update_procedure_chains(self, procedure_id: str) -> None:
        """Update procedure chains based on relationships."""
        procedure = next(p for p in self.procedures if p["id"] == procedure_id)
        procedure_idx = self.procedures.index(procedure)
        
        # Find related procedures
        related_procedures = []
        for i, other_procedure in enumerate(self.procedures):
            if other_procedure["id"] == procedure_id:
                continue
            
            similarity = self._cosine_similarity(
                self.procedure_embeddings[procedure_idx],
                self.procedure_embeddings[i]
            )
            
            if similarity >= self.similarity_threshold:
                related_procedures.append((other_procedure["id"], similarity))
        
        # Sort by similarity
        related_procedures.sort(key=lambda x: x[1], reverse=True)
        
        # Update chains
        self.procedure_chains[procedure_id] = [
            proc_id for proc_id, _ in related_procedures[:self.chain_depth]
        ]
        
        # Update chain positions
        for i, chain_id in enumerate(self.procedure_chains[procedure_id]):
            self.procedure_metadata[chain_id]["chain_position"] = i + 1

    async def _monitor_procedures(self) -> None:
        """Monitor procedure performance and update metrics."""
        for procedure in self.procedures:
            procedure_id = procedure["id"]
            metrics = self.monitoring_metrics[procedure_id]
            
            # Calculate performance metrics
            execution_records = self.execution_history.get(procedure_id, [])
            if execution_records:
                # Performance (success rate weighted by execution count)
                success_rate = sum(1 for r in execution_records if r["success"]) / len(execution_records)
                metrics["performance"] = success_rate * (1 + len(execution_records) / 100)
                
                # Reliability (consistency of execution duration)
                durations = [r["duration"] for r in execution_records]
                avg_duration = sum(durations) / len(durations)
                duration_variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
                metrics["reliability"] = 1 / (1 + duration_variance)
                
                # Efficiency (inverse of average duration)
                metrics["efficiency"] = 1 / (1 + avg_duration)
            
            # Update learning progress
            if self.enable_learning:
                await self._update_learning_progress(procedure_id)
        
        self.last_monitoring = datetime.now()

    async def _update_learning_progress(self, procedure_id: str) -> None:
        """Update learning progress based on execution history."""
        execution_records = self.execution_history.get(procedure_id, [])
        if not execution_records:
            return
        
        # Calculate learning metrics
        recent_records = execution_records[-10:]  # Last 10 executions
        success_rate = sum(1 for r in recent_records if r["success"]) / len(recent_records)
        avg_duration = sum(r["duration"] for r in recent_records) / len(recent_records)
        
        # Update learning progress
        progress = (
            self.learning_rate * success_rate +
            self.learning_rate * (1 / (1 + avg_duration))
        )
        
        self.procedure_metadata[procedure_id]["learning_progress"] = min(
            1.0,
            self.procedure_metadata[procedure_id]["learning_progress"] + progress
        )
        
        # Record learning update
        self.learning_history[procedure_id].append({
            "timestamp": datetime.now().isoformat(),
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "progress": progress
        })

    async def get_procedure_chain(self, procedure_id: str) -> List[Dict[str, Any]]:
        """Get the chain of related procedures."""
        if procedure_id not in self.procedure_chains:
            return []
        
        chain = []
        for chain_id in self.procedure_chains[procedure_id]:
            procedure = await self.get_procedure_by_id(chain_id)
            if procedure:
                chain.append(procedure)
        
        return chain

    async def get_monitoring_metrics(self, procedure_id: str) -> Dict[str, Any]:
        """Get monitoring metrics for a procedure."""
        return self.monitoring_metrics.get(procedure_id, {})

    async def get_learning_history(
        self,
        procedure_id: str,
        min_progress: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get learning history of a procedure."""
        if procedure_id not in self.learning_history:
            return []
        
        if min_progress is None:
            return self.learning_history[procedure_id]
        
        return [
            record for record in self.learning_history[procedure_id]
            if record["progress"] >= min_progress
        ]

    async def get_procedure_stats(self) -> Dict[str, Any]:
        """Get statistics about procedures."""
        stats = await super().get_procedure_stats()
        
        # Add chain statistics
        stats["chain_stats"] = {
            "total_chains": len(self.procedure_chains),
            "average_chain_length": sum(len(chain) for chain in self.procedure_chains.values()) / len(self.procedure_chains) if self.procedure_chains else 0,
            "max_chain_length": max(len(chain) for chain in self.procedure_chains.values()) if self.procedure_chains else 0
        }
        
        # Add monitoring statistics
        stats["monitoring_stats"] = {
            "average_performance": sum(m["performance"] for m in self.monitoring_metrics.values()) / len(self.monitoring_metrics) if self.monitoring_metrics else 0,
            "average_reliability": sum(m["reliability"] for m in self.monitoring_metrics.values()) / len(self.monitoring_metrics) if self.monitoring_metrics else 0,
            "average_efficiency": sum(m["efficiency"] for m in self.monitoring_metrics.values()) / len(self.monitoring_metrics) if self.monitoring_metrics else 0
        }
        
        # Add learning statistics
        stats["learning_stats"] = {
            "average_progress": sum(p["metadata"]["learning_progress"] for p in self.procedures) / len(self.procedures) if self.procedures else 0,
            "procedures_with_progress": sum(1 for p in self.procedures if p["metadata"]["learning_progress"] > 0)
        }
        
        return stats

    async def get_procedure_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for procedure optimization."""
        suggestions = await super().get_procedure_suggestions()
        
        # Add chain-related suggestions
        stats = await self.get_procedure_stats()
        if stats["chain_stats"]["average_chain_length"] < 2:
            suggestions.append({
                "type": "chain_development",
                "suggestion": "Consider developing more procedure chains for better knowledge organization"
            })
        
        # Add monitoring-related suggestions
        if stats["monitoring_stats"]["average_performance"] < 0.7:
            suggestions.append({
                "type": "performance_improvement",
                "suggestion": "Consider improving procedure performance through optimization"
            })
        
        # Add learning-related suggestions
        if stats["learning_stats"]["average_progress"] < 0.5:
            suggestions.append({
                "type": "learning_enhancement",
                "suggestion": "Consider enhancing learning mechanisms for procedures"
            })
        
        return suggestions 