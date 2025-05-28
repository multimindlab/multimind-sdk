"""
Declarative memory implementation that manages factual knowledge with verification and confidence scoring.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class DeclarativeMemory(BaseMemory):
    """Memory that manages factual knowledge with verification and confidence scoring."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_facts: int = 1000,
        verification_threshold: float = 0.8,
        confidence_threshold: float = 0.7,
        enable_verification: bool = True,
        verification_interval: int = 3600,  # 1 hour
        enable_consistency: bool = True,
        consistency_interval: int = 3600,  # 1 hour
        enable_learning: bool = True,
        learning_rate: float = 0.1,
        enable_history: bool = True,
        history_window: int = 100,
        enable_evolution: bool = True,
        evolution_interval: int = 3600,  # 1 hour
        enable_validation: bool = True,
        validation_interval: int = 3600,  # 1 hour
        enable_knowledge_integration: bool = True,
        integration_interval: int = 3600,  # 1 hour
        enable_semantic_reasoning: bool = True,
        reasoning_interval: int = 3600,  # 1 hour
        enable_uncertainty: bool = True,
        uncertainty_interval: int = 3600,  # 1 hour
        enable_contradiction_detection: bool = True,
        contradiction_interval: int = 3600,  # 1 hour
        enable_temporal_reasoning: bool = True,
        temporal_interval: int = 3600,  # 1 hour
        enable_causal_analysis: bool = True,
        causal_interval: int = 3600,  # 1 hour
        enable_knowledge_graph: bool = True,
        graph_update_interval: int = 3600,  # 1 hour
        relationship_types: Set[str] = None
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_facts = max_facts
        self.verification_threshold = verification_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_verification = enable_verification
        self.verification_interval = verification_interval
        self.enable_consistency = enable_consistency
        self.consistency_interval = consistency_interval
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.enable_history = enable_history
        self.history_window = history_window
        self.enable_evolution = enable_evolution
        self.evolution_interval = evolution_interval
        self.enable_validation = enable_validation
        self.validation_interval = validation_interval
        self.enable_knowledge_integration = enable_knowledge_integration
        self.integration_interval = integration_interval
        self.enable_semantic_reasoning = enable_semantic_reasoning
        self.reasoning_interval = reasoning_interval
        self.enable_uncertainty = enable_uncertainty
        self.uncertainty_interval = uncertainty_interval
        self.enable_contradiction_detection = enable_contradiction_detection
        self.contradiction_interval = contradiction_interval
        self.enable_temporal_reasoning = enable_temporal_reasoning
        self.temporal_interval = temporal_interval
        self.enable_causal_analysis = enable_causal_analysis
        self.causal_interval = causal_interval
        self.enable_knowledge_graph = enable_knowledge_graph
        self.graph_update_interval = graph_update_interval
        self.relationship_types = relationship_types or {
            "implies",
            "contradicts",
            "supports",
            "refutes",
            "elaborates",
            "generalizes",
            "specializes",
            "analogous_to",
            "causes",
            "prevents",
            "temporally_before",
            "temporally_after",
            "temporally_during",
            "temporally_overlaps",
            "causally_affects",
            "causally_inhibits",
            "causally_enables",
            "causally_triggers"
        }
        
        # Initialize declarative memory storage
        self.facts: List[Dict[str, Any]] = []
        self.fact_embeddings: List[List[float]] = []
        self.relationships: Dict[str, Dict[str, List[str]]] = {}  # fact_id -> {relationship_type -> target_ids}
        self.verification_history: Dict[str, List[Dict[str, Any]]] = {}  # fact_id -> verification records
        self.consistency_history: Dict[str, List[Dict[str, Any]]] = {}  # fact_id -> consistency records
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}
        self.fact_history: List[Dict[str, Any]] = []  # Recent fact updates
        self.evolution_history: Dict[str, List[Dict[str, Any]]] = {}  # fact_id -> evolution records
        self.validation_history: Dict[str, List[Dict[str, Any]]] = {}  # fact_id -> validation records
        self.integrated_knowledge: Dict[str, Dict[str, Any]] = {}  # integration_id -> integrated knowledge
        self.semantic_reasoning: Dict[str, Dict[str, Any]] = {}  # reasoning_id -> reasoning results
        self.uncertainty_measures: Dict[str, Dict[str, Any]] = {}  # fact_id -> uncertainty data
        self.contradictions: Dict[str, List[Dict[str, Any]]] = {}  # fact_id -> contradiction records
        self.temporal_relations: Dict[str, Dict[str, Any]] = {}  # fact_id -> temporal data
        self.causal_chains: Dict[str, List[Dict[str, Any]]] = {}  # fact_id -> causal chain data
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}  # node_id -> node data
        self.last_verification = datetime.now()
        self.last_consistency = datetime.now()
        self.last_evolution = datetime.now()
        self.last_validation = datetime.now()
        self.last_integration = datetime.now()
        self.last_reasoning = datetime.now()
        self.last_uncertainty = datetime.now()
        self.last_contradiction = datetime.now()
        self.last_temporal = datetime.now()
        self.last_causal = datetime.now()
        self.last_graph_update = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and analyze factual information."""
        # Create new fact
        fact_id = f"fact_{len(self.facts)}"
        new_fact = {
            "id": fact_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "verification_score": 0.0,
                "confidence_score": 0.0,
                "consistency_score": 0.0,
                "learning_progress": 0.0,
                "evolution_stage": 0,
                "validation_score": 0.0,
                "uncertainty_score": 0.0,
                "verification_results": {},
                "consistency_results": {},
                "validation_results": {},
                "uncertainty_results": {},
                "contradiction_results": {},
                "integration_data": {},
                "reasoning_data": {},
                "temporal_data": {},
                "causal_data": {},
                "graph_data": {}
            }
        }
        
        # Add to storage
        self.facts.append(new_fact)
        
        # Get fact embedding
        embedding = await self.llm.embeddings(message["content"])
        self.fact_embeddings.append(embedding)
        
        # Perform all analyses
        if self.enable_verification:
            current_time = datetime.now()
            if (current_time - self.last_verification).total_seconds() > self.verification_interval:
                await self._verify_fact(fact_id)
        
        if self.enable_consistency:
            current_time = datetime.now()
            if (current_time - self.last_consistency).total_seconds() > self.consistency_interval:
                await self._check_consistency(fact_id)
        
        if self.enable_learning:
            await self._update_learning_progress(fact_id)
        
        if self.enable_evolution:
            current_time = datetime.now()
            if (current_time - self.last_evolution).total_seconds() > self.evolution_interval:
                await self._update_evolution(fact_id)
        
        if self.enable_validation:
            current_time = datetime.now()
            if (current_time - self.last_validation).total_seconds() > self.validation_interval:
                await self._validate_fact(fact_id)
        
        if self.enable_knowledge_integration:
            current_time = datetime.now()
            if (current_time - self.last_integration).total_seconds() > self.integration_interval:
                await self._integrate_knowledge(fact_id)
        
        if self.enable_semantic_reasoning:
            current_time = datetime.now()
            if (current_time - self.last_reasoning).total_seconds() > self.reasoning_interval:
                await self._perform_semantic_reasoning(fact_id)
        
        if self.enable_uncertainty:
            current_time = datetime.now()
            if (current_time - self.last_uncertainty).total_seconds() > self.uncertainty_interval:
                await self._update_uncertainty_measures(fact_id)
        
        if self.enable_contradiction_detection:
            current_time = datetime.now()
            if (current_time - self.last_contradiction).total_seconds() > self.contradiction_interval:
                await self._detect_contradictions(fact_id)
        
        if self.enable_temporal_reasoning:
            current_time = datetime.now()
            if (current_time - self.last_temporal).total_seconds() > self.temporal_interval:
                await self._analyze_temporal_relations(fact_id)
        
        if self.enable_causal_analysis:
            current_time = datetime.now()
            if (current_time - self.last_causal).total_seconds() > self.causal_interval:
                await self._analyze_causal_chains(fact_id)
        
        if self.enable_knowledge_graph:
            current_time = datetime.now()
            if (current_time - self.last_graph_update).total_seconds() > self.graph_update_interval:
                await self._update_knowledge_graph(fact_id)
        
        # Update fact history
        if self.enable_history:
            self.fact_history.append({
                "fact_id": fact_id,
                "timestamp": new_fact["timestamp"],
                "content": new_fact["content"],
                "verification_score": new_fact["metadata"]["verification_score"],
                "confidence_score": new_fact["metadata"]["confidence_score"],
                "consistency_score": new_fact["metadata"]["consistency_score"]
            })
            if len(self.fact_history) > self.history_window:
                self.fact_history.pop(0)
        
        # Maintain fact limit
        await self._maintain_fact_limit()
        
        await self.save()

    async def _verify_fact(self, fact_id: str) -> None:
        """Verify a fact using multiple sources and methods."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        try:
            # Generate verification prompt
            prompt = f"""
            Verify this fact using multiple methods:
            
            {fact['content']}
            
            Return a JSON object with:
            1. verification_score: float (0-1)
            2. verification_methods: list of strings
            3. supporting_evidence: list of strings
            4. conflicting_evidence: list of strings
            5. confidence_level: string (high/medium/low)
            6. verification_notes: string
            """
            response = await self.llm.generate(prompt)
            verification = json.loads(response)
            
            # Update fact metadata
            fact["metadata"]["verification_score"] = verification["verification_score"]
            fact["metadata"]["verification_results"] = verification
            
            # Record verification
            self.verification_history[fact_id].append({
                "timestamp": datetime.now().isoformat(),
                "score": verification["verification_score"],
                "methods": verification["verification_methods"],
                "supporting_evidence": verification["supporting_evidence"],
                "conflicting_evidence": verification["conflicting_evidence"],
                "confidence_level": verification["confidence_level"],
                "notes": verification["verification_notes"]
            })
            
        except Exception as e:
            print(f"Error verifying fact: {e}")
        
        self.last_verification = datetime.now()

    async def _check_consistency(self, fact_id: str) -> None:
        """Check consistency of a fact with other facts."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        try:
            # Generate consistency check prompt
            prompt = f"""
            Check consistency of this fact with other facts:
            
            {fact['content']}
            
            Return a JSON object with:
            1. consistency_score: float (0-1)
            2. consistent_facts: list of strings
            3. inconsistent_facts: list of strings
            4. consistency_reason: string
            5. resolution_suggestions: list of strings
            """
            response = await self.llm.generate(prompt)
            consistency = json.loads(response)
            
            # Update fact metadata
            fact["metadata"]["consistency_score"] = consistency["consistency_score"]
            fact["metadata"]["consistency_results"] = consistency
            
            # Record consistency check
            self.consistency_history[fact_id].append({
                "timestamp": datetime.now().isoformat(),
                "score": consistency["consistency_score"],
                "consistent_facts": consistency["consistent_facts"],
                "inconsistent_facts": consistency["inconsistent_facts"],
                "reason": consistency["consistency_reason"],
                "suggestions": consistency["resolution_suggestions"]
            })
            
        except Exception as e:
            print(f"Error checking consistency: {e}")
        
        self.last_consistency = datetime.now()

    async def _integrate_knowledge(self, fact_id: str) -> None:
        """Integrate new knowledge with existing knowledge."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        try:
            # Generate integration prompt
            prompt = f"""
            Integrate this fact with existing knowledge:
            
            {fact['content']}
            
            Return a JSON object with:
            1. integration_score: float (0-1)
            2. integrated_concepts: list of strings
            3. knowledge_gaps: list of strings
            4. integration_notes: string
            5. related_domains: list of strings
            """
            response = await self.llm.generate(prompt)
            integration = json.loads(response)
            
            # Create integration record
            integration_id = f"integration_{len(self.integrated_knowledge)}"
            self.integrated_knowledge[integration_id] = {
                "fact_id": fact_id,
                "timestamp": datetime.now().isoformat(),
                "score": integration["integration_score"],
                "concepts": integration["integrated_concepts"],
                "gaps": integration["knowledge_gaps"],
                "notes": integration["integration_notes"],
                "domains": integration["related_domains"]
            }
            
            # Update fact metadata
            fact["metadata"]["integration_data"][integration_id] = {
                "score": integration["integration_score"],
                "concepts": integration["integrated_concepts"],
                "domains": integration["related_domains"]
            }
            
        except Exception as e:
            print(f"Error integrating knowledge: {e}")
        
        self.last_integration = datetime.now()

    async def _perform_semantic_reasoning(self, fact_id: str) -> None:
        """Perform semantic reasoning on a fact."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        try:
            # Generate reasoning prompt
            prompt = f"""
            Perform semantic reasoning on this fact:
            
            {fact['content']}
            
            Return a JSON object with:
            1. reasoning_score: float (0-1)
            2. logical_consequences: list of strings
            3. assumptions: list of strings
            4. reasoning_chain: list of strings
            5. reasoning_type: string
            6. reasoning_notes: string
            """
            response = await self.llm.generate(prompt)
            reasoning = json.loads(response)
            
            # Create reasoning record
            reasoning_id = f"reasoning_{len(self.semantic_reasoning)}"
            self.semantic_reasoning[reasoning_id] = {
                "fact_id": fact_id,
                "timestamp": datetime.now().isoformat(),
                "score": reasoning["reasoning_score"],
                "consequences": reasoning["logical_consequences"],
                "assumptions": reasoning["assumptions"],
                "chain": reasoning["reasoning_chain"],
                "type": reasoning["reasoning_type"],
                "notes": reasoning["reasoning_notes"]
            }
            
            # Update fact metadata
            fact["metadata"]["reasoning_data"][reasoning_id] = {
                "score": reasoning["reasoning_score"],
                "consequences": reasoning["logical_consequences"],
                "type": reasoning["reasoning_type"]
            }
            
        except Exception as e:
            print(f"Error performing semantic reasoning: {e}")
        
        self.last_reasoning = datetime.now()

    async def _update_uncertainty_measures(self, fact_id: str) -> None:
        """Update uncertainty measures for a fact."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        try:
            # Generate uncertainty prompt
            prompt = f"""
            Assess uncertainty in this fact:
            
            {fact['content']}
            
            Return a JSON object with:
            1. uncertainty_score: float (0-1)
            2. uncertainty_sources: list of strings
            3. confidence_factors: list of strings
            4. reliability_indicators: list of strings
            5. uncertainty_type: string
            6. uncertainty_notes: string
            """
            response = await self.llm.generate(prompt)
            uncertainty = json.loads(response)
            
            # Update uncertainty measures
            self.uncertainty_measures[fact_id] = {
                "timestamp": datetime.now().isoformat(),
                "score": uncertainty["uncertainty_score"],
                "sources": uncertainty["uncertainty_sources"],
                "confidence_factors": uncertainty["confidence_factors"],
                "reliability": uncertainty["reliability_indicators"],
                "type": uncertainty["uncertainty_type"],
                "notes": uncertainty["uncertainty_notes"]
            }
            
            # Update fact metadata
            fact["metadata"]["uncertainty_score"] = uncertainty["uncertainty_score"]
            fact["metadata"]["uncertainty_results"] = uncertainty
            
        except Exception as e:
            print(f"Error updating uncertainty measures: {e}")
        
        self.last_uncertainty = datetime.now()

    async def _detect_contradictions(self, fact_id: str) -> None:
        """Detect contradictions with a fact."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        try:
            # Generate contradiction detection prompt
            prompt = f"""
            Detect contradictions with this fact:
            
            {fact['content']}
            
            Return a JSON object with:
            1. contradiction_score: float (0-1)
            2. contradictory_facts: list of strings
            3. contradiction_type: string
            4. resolution_strategies: list of strings
            5. contradiction_notes: string
            """
            response = await self.llm.generate(prompt)
            contradiction = json.loads(response)
            
            # Record contradiction
            self.contradictions[fact_id].append({
                "timestamp": datetime.now().isoformat(),
                "score": contradiction["contradiction_score"],
                "contradictory_facts": contradiction["contradictory_facts"],
                "type": contradiction["contradiction_type"],
                "strategies": contradiction["resolution_strategies"],
                "notes": contradiction["contradiction_notes"]
            })
            
            # Update fact metadata
            fact["metadata"]["contradiction_results"] = contradiction
            
        except Exception as e:
            print(f"Error detecting contradictions: {e}")
        
        self.last_contradiction = datetime.now()

    async def _update_learning_progress(self, fact_id: str) -> None:
        """Update learning progress for a fact."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        # Calculate learning metrics
        verification_score = fact["metadata"]["verification_score"]
        consistency_score = fact["metadata"]["consistency_score"]
        validation_score = fact["metadata"]["validation_score"]
        
        # Update learning progress
        progress = (
            self.learning_rate * verification_score +
            self.learning_rate * consistency_score +
            self.learning_rate * validation_score
        )
        
        fact["metadata"]["learning_progress"] = min(
            1.0,
            fact["metadata"]["learning_progress"] + progress
        )
        
        # Record learning update
        self.learning_history[fact_id].append({
            "timestamp": datetime.now().isoformat(),
            "verification_score": verification_score,
            "consistency_score": consistency_score,
            "validation_score": validation_score,
            "progress": progress
        })

    async def _update_evolution(self, fact_id: str) -> None:
        """Update evolution stage for a fact."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        # Calculate evolution metrics
        learning_progress = fact["metadata"]["learning_progress"]
        verification_score = fact["metadata"]["verification_score"]
        consistency_score = fact["metadata"]["consistency_score"]
        
        # Determine evolution stage
        if learning_progress >= 0.8 and verification_score >= 0.8 and consistency_score >= 0.8:
            stage = 3  # Mature
        elif learning_progress >= 0.5 or verification_score >= 0.5 or consistency_score >= 0.5:
            stage = 2  # Developing
        elif verification_score > 0 or consistency_score > 0:
            stage = 1  # Emerging
        else:
            stage = 0  # New
        
        # Update evolution stage
        fact["metadata"]["evolution_stage"] = stage
        
        # Record evolution
        self.evolution_history[fact_id].append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "learning_progress": learning_progress,
            "verification_score": verification_score,
            "consistency_score": consistency_score
        })

    async def _validate_fact(self, fact_id: str) -> None:
        """Validate a fact."""
        fact = next(f for f in self.facts if f["id"] == fact_id)
        
        try:
            # Generate validation prompt
            prompt = f"""
            Validate this fact:
            
            {fact['content']}
            
            Return a JSON object with:
            1. validation_score: float (0-1)
            2. validation_reason: string
            3. inconsistencies: list of strings
            4. suggestions: list of strings
            """
            response = await self.llm.generate(prompt)
            validation = json.loads(response)
            
            # Update fact metadata
            fact["metadata"]["validation_score"] = validation["validation_score"]
            fact["metadata"]["validation_results"] = validation
            
            # Record validation
            self.validation_history[fact_id].append({
                "timestamp": datetime.now().isoformat(),
                "score": validation["validation_score"],
                "reason": validation["validation_reason"],
                "inconsistencies": validation["inconsistencies"],
                "suggestions": validation["suggestions"]
            })
            
        except Exception as e:
            print(f"Error validating fact: {e}")

    async def _maintain_fact_limit(self) -> None:
        """Maintain fact limit by removing least important facts."""
        if len(self.facts) > self.max_facts:
            # Sort facts by learning progress and validation score
            sorted_facts = sorted(
                self.facts,
                key=lambda x: (
                    x["metadata"]["learning_progress"] +
                    x["metadata"]["validation_score"]
                )
            )
            
            # Remove facts with lowest scores
            facts_to_remove = sorted_facts[:len(self.facts) - self.max_facts]
            for fact in facts_to_remove:
                await self._remove_fact(fact["id"])

    async def _remove_fact(self, fact_id: str) -> None:
        """Remove a fact and its associated data."""
        # Remove from facts
        fact_idx = next(i for i, f in enumerate(self.facts) if f["id"] == fact_id)
        self.facts.pop(fact_idx)
        self.fact_embeddings.pop(fact_idx)
        
        # Remove from history
        if self.enable_history:
            self.fact_history = [
                f for f in self.fact_history
                if f["fact_id"] != fact_id
            ]
        
        # Remove verification history
        if fact_id in self.verification_history:
            del self.verification_history[fact_id]
        
        # Remove consistency history
        if fact_id in self.consistency_history:
            del self.consistency_history[fact_id]
        
        # Remove learning history
        if fact_id in self.learning_history:
            del self.learning_history[fact_id]
        
        # Remove evolution history
        if fact_id in self.evolution_history:
            del self.evolution_history[fact_id]
        
        # Remove validation history
        if fact_id in self.validation_history:
            del self.validation_history[fact_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all facts."""
        messages = []
        for fact in self.facts:
            messages.append({
                "role": "declarative_memory",
                "content": fact["content"],
                "timestamp": fact["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all facts."""
        self.facts = []
        self.fact_embeddings = []
        self.verification_history = {}
        self.consistency_history = {}
        self.learning_history = {}
        self.fact_history = []
        self.evolution_history = {}
        self.validation_history = {}
        await self.save()

    async def save(self) -> None:
        """Save facts to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "facts": self.facts,
                    "relationships": self.relationships,
                    "verification_history": self.verification_history,
                    "consistency_history": self.consistency_history,
                    "learning_history": self.learning_history,
                    "fact_history": self.fact_history,
                    "evolution_history": self.evolution_history,
                    "validation_history": self.validation_history,
                    "integrated_knowledge": self.integrated_knowledge,
                    "semantic_reasoning": self.semantic_reasoning,
                    "uncertainty_measures": self.uncertainty_measures,
                    "contradictions": self.contradictions,
                    "temporal_relations": self.temporal_relations,
                    "causal_chains": self.causal_chains,
                    "knowledge_graph": self.knowledge_graph,
                    "last_verification": self.last_verification.isoformat(),
                    "last_consistency": self.last_consistency.isoformat(),
                    "last_evolution": self.last_evolution.isoformat(),
                    "last_validation": self.last_validation.isoformat(),
                    "last_integration": self.last_integration.isoformat(),
                    "last_reasoning": self.last_reasoning.isoformat(),
                    "last_uncertainty": self.last_uncertainty.isoformat(),
                    "last_contradiction": self.last_contradiction.isoformat(),
                    "last_temporal": self.last_temporal.isoformat(),
                    "last_causal": self.last_causal.isoformat(),
                    "last_graph_update": self.last_graph_update.isoformat()
                }, f)

    def load(self) -> None:
        """Load facts from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.facts = data.get("facts", [])
                self.relationships = data.get("relationships", {})
                self.verification_history = data.get("verification_history", {})
                self.consistency_history = data.get("consistency_history", {})
                self.learning_history = data.get("learning_history", {})
                self.fact_history = data.get("fact_history", [])
                self.evolution_history = data.get("evolution_history", {})
                self.validation_history = data.get("validation_history", {})
                self.integrated_knowledge = data.get("integrated_knowledge", {})
                self.semantic_reasoning = data.get("semantic_reasoning", {})
                self.uncertainty_measures = data.get("uncertainty_measures", {})
                self.contradictions = data.get("contradictions", {})
                self.temporal_relations = data.get("temporal_relations", {})
                self.causal_chains = data.get("causal_chains", {})
                self.knowledge_graph = data.get("knowledge_graph", {})
                self.last_verification = datetime.fromisoformat(
                    data.get("last_verification", datetime.now().isoformat())
                )
                self.last_consistency = datetime.fromisoformat(
                    data.get("last_consistency", datetime.now().isoformat())
                )
                self.last_evolution = datetime.fromisoformat(
                    data.get("last_evolution", datetime.now().isoformat())
                )
                self.last_validation = datetime.fromisoformat(
                    data.get("last_validation", datetime.now().isoformat())
                )
                self.last_integration = datetime.fromisoformat(
                    data.get("last_integration", datetime.now().isoformat())
                )
                self.last_reasoning = datetime.fromisoformat(
                    data.get("last_reasoning", datetime.now().isoformat())
                )
                self.last_uncertainty = datetime.fromisoformat(
                    data.get("last_uncertainty", datetime.now().isoformat())
                )
                self.last_contradiction = datetime.fromisoformat(
                    data.get("last_contradiction", datetime.now().isoformat())
                )
                self.last_temporal = datetime.fromisoformat(
                    data.get("last_temporal", datetime.now().isoformat())
                )
                self.last_causal = datetime.fromisoformat(
                    data.get("last_causal", datetime.now().isoformat())
                )
                self.last_graph_update = datetime.fromisoformat(
                    data.get("last_graph_update", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.fact_embeddings = []
                for fact in self.facts:
                    self.fact_embeddings.append(
                        self.llm.embeddings(fact["content"])
                    )

    async def get_declarative_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about declarative memory."""
        stats = {
            "total_facts": len(self.facts),
            "verification_stats": {
                "average_score": sum(
                    f["metadata"]["verification_score"]
                    for f in self.facts
                ) / len(self.facts) if self.facts else 0,
                "verified_facts": sum(
                    1 for f in self.facts
                    if f["metadata"]["verification_score"] >= self.verification_threshold
                )
            },
            "consistency_stats": {
                "average_score": sum(
                    f["metadata"]["consistency_score"]
                    for f in self.facts
                ) / len(self.facts) if self.facts else 0,
                "consistent_facts": sum(
                    1 for f in self.facts
                    if f["metadata"]["consistency_score"] >= 0.8
                )
            },
            "learning_stats": {
                "average_progress": sum(
                    f["metadata"]["learning_progress"]
                    for f in self.facts
                ) / len(self.facts) if self.facts else 0,
                "facts_with_progress": sum(
                    1 for f in self.facts
                    if f["metadata"]["learning_progress"] > 0
                )
            },
            "evolution_stats": {
                "stage_distribution": {
                    stage: sum(1 for f in self.facts if f["metadata"]["evolution_stage"] == stage)
                    for stage in range(4)
                },
                "average_stage": sum(f["metadata"]["evolution_stage"] for f in self.facts) / len(self.facts) if self.facts else 0
            },
            "validation_stats": {
                "average_score": sum(
                    f["metadata"]["validation_score"]
                    for f in self.facts
                ) / len(self.facts) if self.facts else 0,
                "validated_facts": sum(
                    1 for f in self.facts
                    if f["metadata"]["validation_score"] >= 0.8
                )
            }
        }
        
        # Add knowledge integration statistics
        if self.enable_knowledge_integration:
            stats["integration_stats"] = {
                "total_integrations": len(self.integrated_knowledge),
                "average_score": sum(
                    integration["score"]
                    for integration in self.integrated_knowledge.values()
                ) / len(self.integrated_knowledge) if self.integrated_knowledge else 0,
                "domain_distribution": {
                    domain: sum(
                        1 for integration in self.integrated_knowledge.values()
                        if domain in integration["domains"]
                    )
                    for domain in set(
                        domain
                        for integration in self.integrated_knowledge.values()
                        for domain in integration["domains"]
                    )
                }
            }
        
        # Add semantic reasoning statistics
        if self.enable_semantic_reasoning:
            stats["reasoning_stats"] = {
                "total_reasonings": len(self.semantic_reasoning),
                "average_score": sum(
                    reasoning["score"]
                    for reasoning in self.semantic_reasoning.values()
                ) / len(self.semantic_reasoning) if self.semantic_reasoning else 0,
                "reasoning_types": {
                    reasoning["type"]: sum(
                        1 for r in self.semantic_reasoning.values()
                        if r["type"] == reasoning["type"]
                    )
                    for reasoning in self.semantic_reasoning.values()
                }
            }
        
        # Add uncertainty statistics
        if self.enable_uncertainty:
            stats["uncertainty_stats"] = {
                "average_score": sum(
                    measures["score"]
                    for measures in self.uncertainty_measures.values()
                ) / len(self.uncertainty_measures) if self.uncertainty_measures else 0,
                "uncertainty_types": {
                    measures["type"]: sum(
                        1 for m in self.uncertainty_measures.values()
                        if m["type"] == measures["type"]
                    )
                    for measures in self.uncertainty_measures.values()
                }
            }
        
        # Add contradiction statistics
        if self.enable_contradiction_detection:
            stats["contradiction_stats"] = {
                "total_contradictions": sum(
                    len(contradictions)
                    for contradictions in self.contradictions.values()
                ),
                "contradiction_types": {
                    contradiction["type"]: sum(
                        1 for c in contradictions
                        if c["type"] == contradiction["type"]
                    )
                    for contradictions in self.contradictions.values()
                    for contradiction in contradictions
                }
            }
        
        # Add temporal reasoning statistics
        if self.enable_temporal_reasoning:
            stats["temporal_stats"] = {
                "total_relations": len(self.temporal_relations),
                "average_score": sum(
                    relation["score"]
                    for relation in self.temporal_relations.values()
                ) / len(self.temporal_relations) if self.temporal_relations else 0,
                "temporal_types": {
                    relation["type"]: sum(
                        1 for r in self.temporal_relations.values()
                        if r["type"] == relation["type"]
                    )
                    for relation in self.temporal_relations.values()
                }
            }
        
        # Add causal analysis statistics
        if self.enable_causal_analysis:
            stats["causal_stats"] = {
                "total_chains": sum(
                    len(chains["chains"])
                    for chains in self.causal_chains.values()
                ),
                "average_score": sum(
                    chains["score"]
                    for chains in self.causal_chains.values()
                ) / len(self.causal_chains) if self.causal_chains else 0,
                "chain_types": {
                    chain["chain_type"]: sum(
                        1 for c in chains["chains"]
                        if c["chain_type"] == chain["chain_type"]
                    )
                    for chains in self.causal_chains.values()
                    for chain in chains["chains"]
                }
            }
        
        # Add knowledge graph statistics
        if self.enable_knowledge_graph:
            stats["graph_stats"] = {
                "total_nodes": len(self.knowledge_graph),
                "node_types": {
                    node["type"]: sum(
                        1 for n in self.knowledge_graph.values()
                        if n["type"] == node["type"]
                    )
                    for node in self.knowledge_graph.values()
                }
            }
        
        return stats

    async def get_declarative_memory_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for declarative memory optimization."""
        suggestions = []
        
        # Check fact count
        if len(self.facts) > self.max_facts * 0.8:
            suggestions.append({
                "type": "fact_limit",
                "suggestion": "Consider increasing max_facts or removing less important facts"
            })
        
        # Check verification quality
        stats = await self.get_declarative_memory_stats()
        if stats["verification_stats"]["average_score"] < self.verification_threshold:
            suggestions.append({
                "type": "verification_improvement",
                "suggestion": "Consider improving fact verification mechanisms"
            })
        
        # Check consistency quality
        if stats["consistency_stats"]["average_score"] < 0.8:
            suggestions.append({
                "type": "consistency_improvement",
                "suggestion": "Consider improving fact consistency checks"
            })
        
        # Check learning progress
        if stats["learning_stats"]["average_progress"] < 0.5:
            suggestions.append({
                "type": "learning_enhancement",
                "suggestion": "Consider enhancing learning mechanisms for facts"
            })
        
        # Check evolution progress
        if stats["evolution_stats"]["average_stage"] < 1.5:
            suggestions.append({
                "type": "evolution_enhancement",
                "suggestion": "Consider enhancing evolution mechanisms for facts"
            })
        
        # Check validation quality
        if stats["validation_stats"]["average_score"] < 0.8:
            suggestions.append({
                "type": "validation_improvement",
                "suggestion": "Consider improving validation mechanisms"
            })
        
        # Add knowledge integration suggestions
        if self.enable_knowledge_integration:
            if stats["integration_stats"]["total_integrations"] < len(self.facts) * 0.1:
                suggestions.append({
                    "type": "integration_development",
                    "suggestion": "Consider developing more knowledge integrations"
                })
        
        # Add semantic reasoning suggestions
        if self.enable_semantic_reasoning:
            if stats["reasoning_stats"]["total_reasonings"] < len(self.facts) * 0.1:
                suggestions.append({
                    "type": "reasoning_development",
                    "suggestion": "Consider developing more semantic reasoning"
                })
        
        # Add uncertainty suggestions
        if self.enable_uncertainty:
            if stats["uncertainty_stats"]["average_score"] > 0.5:
                suggestions.append({
                    "type": "uncertainty_reduction",
                    "suggestion": "Consider reducing uncertainty in facts"
                })
        
        # Add contradiction suggestions
        if self.enable_contradiction_detection:
            if stats["contradiction_stats"]["total_contradictions"] > 0:
                suggestions.append({
                    "type": "contradiction_resolution",
                    "suggestion": "Consider resolving detected contradictions"
                })
        
        # Add temporal reasoning suggestions
        if self.enable_temporal_reasoning:
            if stats["temporal_stats"]["total_relations"] < len(self.facts) * 0.1:
                suggestions.append({
                    "type": "temporal_development",
                    "suggestion": "Consider developing more temporal relationships"
                })
        
        # Add causal analysis suggestions
        if self.enable_causal_analysis:
            if stats["causal_stats"]["total_chains"] < len(self.facts) * 0.1:
                suggestions.append({
                    "type": "causal_development",
                    "suggestion": "Consider developing more causal chains"
                })
        
        # Add knowledge graph suggestions
        if self.enable_knowledge_graph:
            stats = await self.get_declarative_memory_stats()
            if stats["graph_stats"]["total_nodes"] < len(self.facts) * 0.5:
                suggestions.append({
                    "type": "graph_development",
                    "suggestion": "Consider expanding the knowledge graph"
                })
        
        return suggestions 