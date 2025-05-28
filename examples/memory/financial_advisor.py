"""
Financial Advisor Example
This example demonstrates how to use MultiMind's memory capabilities for financial applications,
combining event-sourced memory for transaction tracking and knowledge graph memory for financial knowledge.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import (
    EventSourcedMemory,
    KnowledgeGraphMemory,
    HybridMemory
)
from multimind.models import OllamaLLM

class FinancialAdvisor:
    def __init__(self, model: str = "mistral"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize event-sourced memory for transaction tracking
        self.transaction_memory = EventSourcedMemory(
            llm=self.llm,
            max_events=10000,
            max_snapshots=1000,
            snapshot_interval=3600,  # 1 hour
            enable_causality=True,
            enable_patterns=True,
            pattern_interval=3600,  # 1 hour
            storage_path="financial_transactions.json"
        )
        
        # Initialize knowledge graph memory for financial knowledge
        self.financial_knowledge = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=5000,
            max_edges=20000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path="financial_knowledge.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.transaction_memory,
                self.financial_knowledge
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path="financial_memory.json"
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are a financial advisor that helps with transaction tracking and financial knowledge management.
            You maintain a knowledge graph of financial concepts and track financial events and patterns.
            Always emphasize that you are an AI assistant and not a replacement for professional financial advice."""
        )

    async def add_transaction(self, event_type: str, description: str, metadata: Dict[str, Any]) -> None:
        """Add a financial transaction event."""
        self.transaction_memory.add_event(
            event_type=event_type,
            description=description,
            metadata=metadata
        )

    async def add_financial_knowledge(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add financial knowledge to the knowledge graph."""
        self.financial_knowledge.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def analyze_transactions(self, query: str) -> Dict[str, Any]:
        """Analyze financial transactions and patterns."""
        patterns = self.transaction_memory.get_event_patterns(query)
        causality = self.transaction_memory.get_causality_analysis(query)
        timeline = self.transaction_memory.get_event_timeline()
        return {
            "patterns": patterns,
            "causality": causality,
            "timeline": timeline
        }

    async def query_financial_knowledge(self, query: str) -> Dict[str, Any]:
        """Query financial knowledge and get related concepts."""
        response = await self.mm.chat(query)
        related = self.financial_knowledge.get_related_concepts(query)
        inferences = self.financial_knowledge.get_inferences(query)
        return {
            "response": response,
            "related_concepts": related,
            "inferences": inferences
        }

    async def get_financial_insights(self) -> Dict[str, Any]:
        """Get comprehensive financial insights."""
        stats = self.transaction_memory.get_event_stats()
        patterns = self.transaction_memory.get_event_patterns("financial")
        knowledge_stats = self.financial_knowledge.get_graph_stats()
        return {
            "transaction_stats": stats,
            "patterns": patterns,
            "knowledge_stats": knowledge_stats
        }

async def main():
    # Initialize financial advisor
    advisor = FinancialAdvisor()
    
    # Add some example financial knowledge
    financial_facts = [
        ("Stock Market", "is_a", "Investment Vehicle", 0.95),
        ("Stock Market", "risk_level", "High", 0.85),
        ("Bonds", "is_a", "Fixed Income", 0.95),
        ("Bonds", "risk_level", "Low", 0.90),
        ("Diversification", "strategy", "Risk Management", 0.95),
        ("Diversification", "benefit", "Reduced Risk", 0.90)
    ]
    
    print("Adding financial knowledge...")
    for subject, predicate, object_, confidence in financial_facts:
        await advisor.add_financial_knowledge(subject, predicate, object_, confidence)
    
    # Add example transactions
    print("\nAdding financial transactions...")
    transactions = [
        {
            "event_type": "investment",
            "description": "Purchased 100 shares of AAPL",
            "metadata": {
                "amount": 15000.00,
                "date": "2024-03-15",
                "type": "stock_purchase"
            }
        },
        {
            "event_type": "dividend",
            "description": "Received dividend payment from AAPL",
            "metadata": {
                "amount": 230.00,
                "date": "2024-03-20",
                "type": "dividend_payment"
            }
        },
        {
            "event_type": "bond_purchase",
            "description": "Purchased government bonds",
            "metadata": {
                "amount": 50000.00,
                "date": "2024-03-25",
                "type": "bond_investment"
            }
        }
    ]
    
    for transaction in transactions:
        await advisor.add_transaction(
            transaction["event_type"],
            transaction["description"],
            transaction["metadata"]
        )
    
    # Query financial knowledge
    print("\nQuerying financial knowledge...")
    query_result = await advisor.query_financial_knowledge("What are the risk levels of different investment types?")
    print(f"Response: {query_result['response']}")
    print("\nRelated Concepts:")
    for concept in query_result['related_concepts']:
        print(f"- {concept}")
    print("\nInferences:")
    for inference in query_result['inferences']:
        print(f"- {inference}")
    
    # Analyze transactions
    print("\nAnalyzing transactions...")
    analysis = await advisor.analyze_transactions("investment")
    print("\nTransaction Patterns:")
    for pattern in analysis['patterns']:
        print(f"- {pattern}")
    
    print("\nCausality Analysis:")
    for cause in analysis['causality']:
        print(f"- {cause}")
    
    # Get financial insights
    print("\nGetting financial insights...")
    insights = await advisor.get_financial_insights()
    print("\nTransaction Statistics:")
    for key, value in insights['transaction_stats'].items():
        print(f"{key}: {value}")
    
    print("\nKnowledge Graph Statistics:")
    for key, value in insights['knowledge_stats'].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main()) 