"""
Healthcare Assistant Example
This example demonstrates how to use MultiMind's memory capabilities for healthcare applications,
combining knowledge graph memory for medical knowledge and cognitive scratchpad for diagnosis reasoning.
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

class HealthcareAssistant:
    def __init__(self, model: str = "mistral"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize knowledge graph memory for medical knowledge
        self.medical_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path="medical_knowledge.json"
        )
        
        # Initialize cognitive scratchpad for diagnosis reasoning
        self.diagnosis_reasoning = CognitiveScratchpadMemory(
            llm=self.llm,
            max_steps=100,
            max_chains=50,
            chain_depth=10,
            enable_analysis=True,
            analysis_interval=3600,  # 1 hour
            storage_path="diagnosis_reasoning.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.medical_knowledge,
                self.diagnosis_reasoning
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path="healthcare_memory.json"
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a healthcare assistant that helps with medical knowledge management and diagnosis reasoning.
            You maintain a knowledge graph of medical concepts and use step-by-step reasoning for diagnosis.
            Always emphasize that you are an AI assistant and not a replacement for professional medical advice."""
        )

    async def add_medical_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add medical knowledge to the knowledge graph."""
        self.medical_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def start_diagnosis(self, symptoms: str) -> str:
        """Start a new diagnosis reasoning chain."""
        chain_id = self.diagnosis_reasoning.start_chain(symptoms)
        return chain_id

    async def add_diagnosis_step(self, chain_id: str, step: str, confidence: float) -> None:
        """Add a step to the diagnosis reasoning chain."""
        self.diagnosis_reasoning.add_step(
            chain_id=chain_id,
            step=step,
            confidence=confidence
        )

    async def query_medical_knowledge(self, query: str) -> Dict[str, Any]:
        """Query medical knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.medical_knowledge.get_related_concepts(query)
        return {
            "response": response,
            "related_concepts": related
        }

    async def analyze_diagnosis(self, chain_id: str) -> Dict[str, Any]:
        """Analyze a diagnosis reasoning chain."""
        steps = self.diagnosis_reasoning.get_chain_steps(chain_id)
        deps = self.diagnosis_reasoning.get_step_dependencies(chain_id)
        analysis = self.diagnosis_reasoning.get_reasoning_analysis(chain_id)
        return {
            "steps": steps,
            "dependencies": deps,
            "analysis": analysis
        }

async def main():
    # Initialize healthcare assistant
    assistant = HealthcareAssistant()
    
    # Add some example medical knowledge
    medical_facts = [
        ("Hypertension", "is_a", "Cardiovascular Disease", 0.95),
        ("Hypertension", "symptom", "High Blood Pressure", 0.95),
        ("Hypertension", "treatment", "ACE Inhibitors", 0.85),
        ("Diabetes", "is_a", "Metabolic Disorder", 0.95),
        ("Diabetes", "symptom", "High Blood Sugar", 0.95),
        ("Diabetes", "treatment", "Insulin", 0.90)
    ]
    
    print("Adding medical knowledge...")
    for subject, predicate, object_, confidence in medical_facts:
        await assistant.add_medical_knowledge(subject, predicate, object_, confidence)
    
    # Example diagnosis scenario
    print("\nStarting diagnosis scenario...")
    symptoms = "Patient presents with high blood pressure and frequent urination"
    chain_id = await assistant.start_diagnosis(symptoms)
    
    # Add diagnosis steps
    diagnosis_steps = [
        ("Consider hypertension as primary condition", 0.8),
        ("Check for diabetes due to frequent urination", 0.7),
        ("Review patient's medical history", 0.9),
        ("Recommend blood tests for confirmation", 0.85)
    ]
    
    for step, confidence in diagnosis_steps:
        await assistant.add_diagnosis_step(chain_id, step, confidence)
    
    # Query medical knowledge
    print("\nQuerying medical knowledge...")
    query_result = await assistant.query_medical_knowledge("What are the treatments for hypertension?")
    print(f"Response: {query_result['response']}")
    print("\nRelated Concepts:")
    for concept in query_result['related_concepts']:
        print(f"- {concept}")
    
    # Analyze diagnosis
    print("\nAnalyzing diagnosis...")
    analysis = await assistant.analyze_diagnosis(chain_id)
    print("\nDiagnosis Steps:")
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