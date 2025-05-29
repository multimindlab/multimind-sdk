"""
Advanced examples of the RAG system with agent integration.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
import json

from multimind.models import OpenAIModel, AnthropicModel
from multimind.rag import RAG, Document
from multimind.rag.embeddings import get_embedder
from multimind.agents import Agent, AgentMemory, AgentTool
from multimind.agents.tools import WebSearchTool, CalculatorTool

class RAGTool(AgentTool):
    """Tool for querying the RAG system."""
    
    def __init__(self, rag: RAG):
        """Initialize RAG tool.
        
        Args:
            rag: RAG instance to use
        """
        super().__init__(
            name="rag_query",
            description="Query the RAG system for information from documents",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The query to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve",
                    "default": 3
                }
            }
        )
        self.rag = rag
        
    async def execute(
        self,
        query: str,
        top_k: int = 3,
        **kwargs
    ) -> str:
        """Execute RAG query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Query results as formatted string
        """
        # Get results
        results = await self.rag.query(query, top_k=top_k)
        
        # Format results
        formatted = []
        for i, (doc, score) in enumerate(results, 1):
            formatted.append(
                f"Document {i} (Score: {score:.3f}):\n"
                f"Text: {doc.text}\n"
                f"Metadata: {json.dumps(doc.metadata, indent=2)}"
            )
            
        return "\n\n".join(formatted)

class RAGGeneratorTool(AgentTool):
    """Tool for generating responses using the RAG system."""
    
    def __init__(self, rag: RAG):
        """Initialize RAG generator tool.
        
        Args:
            rag: RAG instance to use
        """
        super().__init__(
            name="rag_generate",
            description="Generate a response using the RAG system",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The query to generate a response for"
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature for generation",
                    "default": 0.7
                }
            }
        )
        self.rag = rag
        
    async def execute(
        self,
        query: str,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Execute RAG generation.
        
        Args:
            query: Query text
            temperature: Generation temperature
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        return await self.rag.generate(
            query,
            temperature=temperature,
            **kwargs
        )

async def main():
    # Initialize models
    openai_model = OpenAIModel(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    anthropic_model = AnthropicModel(
        model="claude-3-sonnet-20240229",
        temperature=0.7
    )
    
    # Initialize embedder
    embedder = get_embedder(
        "openai",
        model="text-embedding-ada-002"
    )
    
    # Create RAG instance with multiple models
    rag = RAG(
        embedder=embedder,
        vector_store="faiss",
        model=openai_model,  # Default model
        chunk_size=1000,
        chunk_overlap=200,
        top_k=3
    )
    
    # Example 1: Process and index a documentation directory
    docs_dir = Path("docs")
    if docs_dir.exists():
        print("\nIndexing documentation...")
        for file_path in docs_dir.rglob("*.md"):
            print(f"Processing {file_path}...")
            await rag.add_file(
                file_path,
                metadata={
                    "category": "documentation",
                    "section": file_path.parent.name
                }
            )
            
    # Example 2: Create an agent with RAG tools
    memory = AgentMemory(
        max_tokens=2000,
        include_metadata=True
    )
    
    # Create RAG tools
    rag_tool = RAGTool(rag)
    rag_generator = RAGGeneratorTool(rag)
    
    # Create agent with multiple tools
    agent = Agent(
        model=anthropic_model,
        memory=memory,
        tools=[
            rag_tool,
            rag_generator,
            WebSearchTool(),
            CalculatorTool()
        ]
    )
    
    # Example 3: Use agent to answer questions about documentation
    questions = [
        "What are the main features of the MultiMind SDK?",
        "How do I use the RAG system?",
        "What models are supported?",
        "Can you explain the agent system?"
    ]
    
    print("\nAgent answering questions...")
    for question in questions:
        print(f"\nQuestion: {question}")
        response = await agent.run(question)
        print(f"Response: {response}")
        
    # Example 4: Use different models for different tasks
    print("\nUsing different models for different tasks...")
    
    # Use Claude for complex reasoning
    rag.model = anthropic_model
    complex_query = """
    Based on the documentation, explain how the RAG system integrates with 
    the agent system and what are the benefits of this integration?
    """
    response = await rag.generate(
        complex_query,
        temperature=0.7,
        max_tokens=1000
    )
    print(f"\nComplex reasoning (Claude):\n{response}")
    
    # Use GPT-3.5 for simpler tasks
    rag.model = openai_model
    simple_query = "List the main components of the RAG system."
    response = await rag.generate(
        simple_query,
        temperature=0.3,
        max_tokens=500
    )
    print(f"\nSimple task (GPT-3.5):\n{response}")
    
    # Example 5: Batch processing and parallel queries
    print("\nBatch processing and parallel queries...")
    
    # Create multiple queries
    queries = [
        "What is the RAG system?",
        "How do I use the agent system?",
        "What are the supported models?",
        "How do I process documents?"
    ]
    
    # Run queries in parallel
    tasks = [rag.query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    # Process results
    for query, query_results in zip(queries, results):
        print(f"\nQuery: {query}")
        for doc, score in query_results:
            print(f"Score: {score:.3f}")
            print(f"Text: {doc.text[:100]}...")
            
    # Example 6: Custom document processing
    print("\nCustom document processing...")
    
    # Create custom document processor
    from multimind.rag.document import DocumentProcessor
    
    processor = DocumentProcessor(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=100,
        tokenizer="cl100k_base"
    )
    
    # Process a long document with custom settings
    long_text = """
    # MultiMind SDK Architecture
    
    The MultiMind SDK is built with a modular architecture that allows for
    easy extension and customization. The main components are:
    
    1. Model Layer
       - Base model interface
       - Model implementations
       - Model management
    
    2. RAG System
       - Document processing
       - Vector storage
       - Semantic search
    
    3. Agent System
       - Tool integration
       - Memory management
       - Task orchestration
    
    4. Integration Layer
       - CLI interface
       - API endpoints
       - Framework integrations
    """
    
    # Process and add with custom metadata
    docs = processor.process_document(
        long_text,
        metadata={
            "type": "architecture",
            "importance": "high",
            "version": "1.0"
        }
    )
    
    await rag.add_documents(docs)
    
    # Query the processed document
    results = await rag.query("What is the architecture of the MultiMind SDK?")
    print("\nArchitecture query results:")
    for doc, score in results:
        print(f"\nScore: {score:.3f}")
        print(f"Text: {doc.text}")
        print(f"Metadata: {doc.metadata}")
        
    # Clean up
    await rag.clear()

if __name__ == "__main__":
    # Check for API keys
    required_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic"
    }
    
    missing_keys = [
        key for key, provider in required_keys.items()
        if not os.getenv(key)
    ]
    
    if missing_keys:
        print("Please set the following environment variables:")
        for key in missing_keys:
            print(f"- {key}")
        exit(1)
        
    # Run examples
    asyncio.run(main()) 