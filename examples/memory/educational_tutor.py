"""
Educational Tutor Example
This example demonstrates how to use MultiMind's memory capabilities for educational applications,
combining active learning memory for student progress tracking and knowledge graph memory for educational content.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
from multimind import MultiMind
from multimind.memory import (
    ActiveLearningMemory,
    KnowledgeGraphMemory,
    HybridMemory
)
from multimind.models import OllamaLLM

class EducationalTutor:
    def __init__(self, model: str = "mistral"):
        self.llm = OllamaLLM(model=model)
        
        # Initialize active learning memory for student progress
        self.student_progress = ActiveLearningMemory(
            llm=self.llm,
            max_feedback=1000,
            max_improvements=100,
            improvement_interval=3600,  # 1 hour
            enable_learning=True,
            enable_analysis=True,
            storage_path="student_progress.json"
        )
        
        # Initialize knowledge graph memory for educational content
        self.educational_content = KnowledgeGraphMemory(
            llm=self.llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,  # 1 hour
            storage_path="educational_content.json"
        )
        
        # Initialize hybrid memory for overall context
        self.memory = HybridMemory(
            llm=self.llm,
            memory_types=[
                self.student_progress,
                self.educational_content
            ],
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            storage_path="educational_memory.json"
        )
        
        self.mm = MultiMind(
            llm=self.llm,
            memory=self.memory,
            system_prompt="""You are an educational tutor that helps with student progress tracking and educational content management.
            You maintain a knowledge graph of educational concepts and track student learning progress.
            Always emphasize that you are an AI assistant and not a replacement for human teachers."""
        )

    async def add_educational_content(self, subject: str, predicate: str, object_: str, confidence: float) -> None:
        """Add educational content to the knowledge graph."""
        self.educational_content.add_knowledge(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence
        )

    async def track_student_progress(self, student_id: str, topic: str, feedback: str, score: float) -> None:
        """Track student progress and learning feedback."""
        self.student_progress.add_feedback(
            student_id=student_id,
            topic=topic,
            feedback=feedback,
            score=score
        )

    async def get_student_insights(self, student_id: str) -> Dict[str, Any]:
        """Get insights about student progress and learning patterns."""
        progress = self.student_progress.get_student_progress(student_id)
        patterns = self.student_progress.get_learning_patterns(student_id)
        improvements = self.student_progress.get_improvements(student_id)
        return {
            "progress": progress,
            "patterns": patterns,
            "improvements": improvements
        }

    async def query_educational_content(self, query: str) -> Dict[str, Any]:
        """Query educational content and get related concepts."""
        response = await self.mm.chat(query)
        related = self.educational_content.get_related_concepts(query)
        inferences = self.educational_content.get_inferences(query)
        return {
            "response": response,
            "related_concepts": related,
            "inferences": inferences
        }

    async def get_learning_recommendations(self, student_id: str) -> Dict[str, Any]:
        """Get personalized learning recommendations."""
        progress = self.student_progress.get_student_progress(student_id)
        patterns = self.student_progress.get_learning_patterns(student_id)
        content_stats = self.educational_content.get_graph_stats()
        return {
            "progress": progress,
            "patterns": patterns,
            "content_stats": content_stats
        }

async def main():
    # Initialize educational tutor
    tutor = EducationalTutor()
    
    # Add some example educational content
    educational_facts = [
        ("Python", "is_a", "Programming Language", 0.95),
        ("Python", "difficulty", "Beginner Friendly", 0.90),
        ("Variables", "concept_in", "Python", 0.95),
        ("Variables", "difficulty", "Basic", 0.90),
        ("Functions", "concept_in", "Python", 0.95),
        ("Functions", "difficulty", "Intermediate", 0.85)
    ]
    
    print("Adding educational content...")
    for subject, predicate, object_, confidence in educational_facts:
        await tutor.add_educational_content(subject, predicate, object_, confidence)
    
    # Track student progress
    print("\nTracking student progress...")
    student_progress = [
        {
            "student_id": "student1",
            "topic": "Python Variables",
            "feedback": "Good understanding of basic variable concepts",
            "score": 0.85
        },
        {
            "student_id": "student1",
            "topic": "Python Functions",
            "feedback": "Needs more practice with function parameters",
            "score": 0.65
        },
        {
            "student_id": "student2",
            "topic": "Python Variables",
            "feedback": "Excellent grasp of variable scope",
            "score": 0.95
        }
    ]
    
    for progress in student_progress:
        await tutor.track_student_progress(
            progress["student_id"],
            progress["topic"],
            progress["feedback"],
            progress["score"]
        )
    
    # Query educational content
    print("\nQuerying educational content...")
    query_result = await tutor.query_educational_content("What are the basic concepts in Python?")
    print(f"Response: {query_result['response']}")
    print("\nRelated Concepts:")
    for concept in query_result['related_concepts']:
        print(f"- {concept}")
    print("\nInferences:")
    for inference in query_result['inferences']:
        print(f"- {inference}")
    
    # Get student insights
    print("\nGetting student insights...")
    insights = await tutor.get_student_insights("student1")
    print("\nStudent Progress:")
    for topic, score in insights['progress'].items():
        print(f"{topic}: {score}")
    
    print("\nLearning Patterns:")
    for pattern in insights['patterns']:
        print(f"- {pattern}")
    
    # Get learning recommendations
    print("\nGetting learning recommendations...")
    recommendations = await tutor.get_learning_recommendations("student1")
    print("\nContent Statistics:")
    for key, value in recommendations['content_stats'].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main()) 