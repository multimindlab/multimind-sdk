"""
Legal Assistant Example
This example demonstrates how to use MultiMind's memory capabilities for legal applications,
combining knowledge graph memory for legal knowledge and cognitive scratchpad for case analysis.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import (
    KnowledgeGraphMemory,
    CognitiveScratchpadMemory,
    HybridMemory
)
from multimind.models import OllamaLLM

class LegalAssistant:
    def __init__(self, model: str = "mistral"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize knowledge graph memory for legal knowledge
        self.legal_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path="legal_knowledge.json"
        )
        
        # Initialize cognitive scratchpad for case analysis
        self.case_analysis = CognitiveScratchpadMemory(
            llm=self.llm,
            max_steps=100,
            max_chains=50,
            chain_depth=10,
            enable_analysis=True,
            analysis_interval=3600,  # 1 hour
            storage_path="case_analysis.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.legal_knowledge,
                self.case_analysis
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path="legal_memory.json"
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a legal assistant that helps with legal knowledge management and case analysis.
            You maintain a knowledge graph of legal concepts and use step-by-step reasoning for case analysis.
            Always emphasize that you are an AI assistant and not a replacement for professional legal advice."""
        )

    async def add_legal_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add legal knowledge to the knowledge graph."""
        self.legal_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def start_case_analysis(self, case_description: str) -> str:
        """Start a new case analysis reasoning chain."""
        chain_id = self.case_analysis.start_chain(case_description)
        return chain_id

    async def add_analysis_step(self, chain_id: str, step: str, confidence: float) -> None:
        """Add a step to the case analysis reasoning chain."""
        self.case_analysis.add_step(
            chain_id=chain_id,
            step=step,
            confidence=confidence
        )

    async def query_legal_knowledge(self, query: str) -> Dict[str, Any]:
        """Query legal knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.legal_knowledge.get_related_concepts(query)
        inferences = self.legal_knowledge.get_inferences(query)
        return {
            "response": response,
            "related_concepts": related,
            "inferences": inferences
        }

    async def analyze_case(self, chain_id: str) -> Dict[str, Any]:
        """Analyze a case reasoning chain."""
        steps = self.case_analysis.get_chain_steps(chain_id)
        deps = self.case_analysis.get_step_dependencies(chain_id)
        analysis = self.case_analysis.get_reasoning_analysis(chain_id)
        return {
            "steps": steps,
            "dependencies": deps,
            "analysis": analysis
        }

async def main():
    # Initialize legal assistant
    assistant = LegalAssistant()
    
    # Add some example legal knowledge
    legal_facts = [
        ("Contract", "is_a", "Legal Agreement", 0.95),
        ("Contract", "requires", "Mutual Consent", 0.95),
        ("Breach of Contract", "is_a", "Legal Violation", 0.95),
        ("Breach of Contract", "remedy", "Damages", 0.90),
        ("Tort", "is_a", "Civil Wrong", 0.95),
        ("Tort", "remedy", "Compensation", 0.90)
    ]
    
    print("Adding legal knowledge...")
    for subject, predicate, object_, confidence in legal_facts:
        await assistant.add_legal_knowledge(subject, predicate, object_, confidence)
    
    # Example case analysis
    print("\nStarting case analysis...")
    case_description = "A company failed to deliver goods as per contract terms"
    chain_id = await assistant.start_case_analysis(case_description)
    
    # Add analysis steps
    analysis_steps = [
        ("Identify contract elements and terms", 0.9),
        ("Determine if breach occurred", 0.85),
        ("Assess damages and remedies", 0.8),
        ("Consider applicable legal precedents", 0.75)
    ]
    
    for step, confidence in analysis_steps:
        await assistant.add_analysis_step(chain_id, step, confidence)
    
    # Query legal knowledge
    print("\nQuerying legal knowledge...")
    query_result = await assistant.query_legal_knowledge("What are the remedies for breach of contract?")
    print(f"Response: {query_result['response']}")
    print("\nRelated Concepts:")
    for concept in query_result['related_concepts']:
        print(f"- {concept}")
    print("\nInferences:")
    for inference in query_result['inferences']:
        print(f"- {inference}")
    
    # Analyze case
    print("\nAnalyzing case...")
    analysis = await assistant.analyze_case(chain_id)
    print("\nAnalysis Steps:")
    for step in analysis['steps']:
        print(f"- {step}")
    
    print("\nStep Dependencies:")
    for step, deps in analysis['dependencies'].items():
        print(f"\nStep: {step}")
        print("Depends on:")
        for dep in deps:
            print(f"- {dep}")
    
    print("\nAnalysis Results:")
    for result in analysis['analysis']:
        print(f"- {result}")

if __name__ == "__main__":
    asyncio.run(main()) 