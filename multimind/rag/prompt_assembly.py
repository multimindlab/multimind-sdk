"""
Advanced prompt assembly module for structured prompt generation.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime
from ..models.base import BaseLLM
from .advanced_prompting import AdvancedPrompting, PromptType, PromptStrategy

@dataclass
class PromptAssemblyConfig:
    """Configuration for prompt assembly."""
    template_type: str
    max_context_length: int
    max_documents: int
    include_metadata: bool
    include_sources: bool
    custom_params: Dict[str, Any]

@dataclass
class AssembledPrompt:
    """Assembled prompt with metadata."""
    prompt: str
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]]
    context_length: int
    document_count: int

class PromptTemplateType(Enum):
    """Types of prompt templates."""
    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_CONSISTENCY = "self_consistency"
    TREE_OF_THOUGHT = "tree_of_thought"
    REFLEXION = "reflexion"
    REACT = "react"
    CUSTOM = "custom"

class PromptAssembly:
    """Advanced prompt assembly with multiple template types."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        config: Optional[PromptAssemblyConfig] = None,
        **kwargs
    ):
        """
        Initialize prompt assembly.
        
        Args:
            llm: Optional LLM for advanced features
            config: Optional assembly configuration
            **kwargs: Additional parameters
        """
        self.llm = llm
        self.config = config or self._get_default_config()
        self.kwargs = kwargs
        
        # Initialize advanced prompting
        self.prompting = AdvancedPrompting(llm=llm)
        
        # Initialize templates
        self.templates = self._initialize_templates()

    def _get_default_config(self) -> PromptAssemblyConfig:
        """Get default assembly configuration."""
        return PromptAssemblyConfig(
            template_type=PromptTemplateType.STANDARD.value,
            max_context_length=4000,
            max_documents=5,
            include_metadata=True,
            include_sources=True,
            custom_params={}
        )

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize prompt templates."""
        return {
            PromptTemplateType.STANDARD.value: """
            You are a helpful AI assistant. Use the following context to answer the question.
            If you cannot answer the question based on the context, say so.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """,
            
            PromptTemplateType.CHAIN_OF_THOUGHT.value: """
            You are a helpful AI assistant. Use the following context to answer the question.
            Think through the answer step by step.
            
            Context:
            {context}
            
            Question: {query}
            
            Let's think through this step by step:
            1.
            """,
            
            PromptTemplateType.SELF_CONSISTENCY.value: """
            You are a helpful AI assistant. Use the following context to answer the question.
            Generate multiple reasoning paths and then combine them into a final answer.
            
            Context:
            {context}
            
            Question: {query}
            
            Let's generate multiple reasoning paths:
            
            Path 1:
            1.
            
            Path 2:
            1.
            
            Path 3:
            1.
            
            Now, let's combine these paths into a final answer:
            """,
            
            PromptTemplateType.TREE_OF_THOUGHT.value: """
            You are a helpful AI assistant. Use the following context to answer the question.
            Explore different reasoning paths in a tree structure.
            
            Context:
            {context}
            
            Question: {query}
            
            Let's explore different reasoning paths:
            
            Branch 1:
            - Step 1:
            - Step 2:
            - Evaluation:
            
            Branch 2:
            - Step 1:
            - Step 2:
            - Evaluation:
            
            Branch 3:
            - Step 1:
            - Step 2:
            - Evaluation:
            
            Now, let's combine the best paths into a final answer:
            """,
            
            PromptTemplateType.REFLEXION.value: """
            You are a helpful AI assistant. Use the following context to answer the question.
            Reflect on your reasoning process and improve it iteratively.
            
            Context:
            {context}
            
            Question: {query}
            
            Initial Answer:
            
            Reflection:
            - What assumptions did I make?
            - What could I have missed?
            - How can I improve my reasoning?
            
            Improved Answer:
            """,
            
            PromptTemplateType.REACT.value: """
            You are a helpful AI assistant. Use the following context to answer the question.
            Follow the ReAct framework: Reason, Act, Observe, and Think.
            
            Context:
            {context}
            
            Question: {query}
            
            Let's follow the ReAct framework:
            
            Thought: What do I need to do to answer this question?
            Action: What specific information should I look for?
            Observation: What did I find in the context?
            Thought: How does this help me answer the question?
            
            Final Answer:
            """
        }

    async def assemble_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        template_type: Optional[str] = None,
        **kwargs
    ) -> AssembledPrompt:
        """
        Assemble prompt with retrieved documents.
        
        Args:
            query: Query string
            documents: Retrieved documents
            template_type: Optional template type
            **kwargs: Additional parameters
            
        Returns:
            Assembled prompt
        """
        # Select template
        template = self.templates.get(
            template_type or self.config.template_type,
            self.templates[PromptTemplateType.STANDARD.value]
        )
        
        # Process documents
        processed_docs = await self._process_documents(
            documents,
            max_docs=self.config.max_documents,
            include_metadata=self.config.include_metadata
        )
        
        # Format context
        context = self._format_context(processed_docs)
        
        # Check context length
        if len(context) > self.config.max_context_length:
            context = await self._truncate_context(
                context,
                query,
                max_length=self.config.max_context_length
            )
        
        # Format sources
        sources = []
        if self.config.include_sources:
            sources = [
                {
                    "id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "metadata": doc.get("metadata", {})
                }
                for doc in processed_docs
            ]
        
        # Generate prompt
        prompt = template.format(
            context=context,
            query=query,
            **kwargs
        )
        
        return AssembledPrompt(
            prompt=prompt,
            metadata={
                "template_type": template_type or self.config.template_type,
                "context_length": len(context),
                "document_count": len(processed_docs)
            },
            sources=sources,
            context_length=len(context),
            document_count=len(processed_docs)
        )

    async def _process_documents(
        self,
        documents: List[Dict[str, Any]],
        max_docs: int,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Process and filter documents."""
        processed_docs = []
        
        for doc in documents[:max_docs]:
            # Extract content
            content = doc.get("content", "")
            if not content:
                continue
            
            # Process document
            processed_doc = {
                "id": doc.get("id", ""),
                "content": content,
                "title": doc.get("title", ""),
                "url": doc.get("url", "")
            }
            
            # Add metadata if requested
            if include_metadata:
                processed_doc["metadata"] = doc.get("metadata", {})
            
            processed_docs.append(processed_doc)
        
        return processed_docs

    def _format_context(
        self,
        documents: List[Dict[str, Any]]
    ) -> str:
        """Format documents into context string."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Format document
            doc_text = f"Document {i}:\n"
            
            # Add title if available
            if doc.get("title"):
                doc_text += f"Title: {doc['title']}\n"
            
            # Add content
            doc_text += f"Content: {doc['content']}\n"
            
            # Add metadata if available
            if doc.get("metadata"):
                metadata_str = json.dumps(doc["metadata"], indent=2)
                doc_text += f"Metadata: {metadata_str}\n"
            
            context_parts.append(doc_text)
        
        return "\n\n".join(context_parts)

    async def _truncate_context(
        self,
        context: str,
        query: str,
        max_length: int
    ) -> str:
        """Truncate context while preserving relevance."""
        if not self.llm:
            # Simple truncation if no LLM available
            return context[:max_length]
        
        # Use LLM to identify most relevant parts
        prompt = f"""
        Given the following context and query, identify the most relevant parts that should be kept.
        The total length should not exceed {max_length} characters.
        
        Query: {query}
        
        Context:
        {context}
        
        Most relevant parts to keep (in order of importance):
        1.
        """
        
        response = await self.llm.generate(prompt)
        
        # Extract relevant parts
        relevant_parts = []
        current_length = 0
        
        for line in response.split("\n"):
            if not line.strip() or not line[0].isdigit():
                continue
            
            # Extract document number
            try:
                doc_num = int(line.split(".")[0])
                if 1 <= doc_num <= len(context.split("\n\n")):
                    doc_text = context.split("\n\n")[doc_num - 1]
                    if current_length + len(doc_text) <= max_length:
                        relevant_parts.append(doc_text)
                        current_length += len(doc_text)
            except ValueError:
                continue
        
        return "\n\n".join(relevant_parts)

    async def generate_custom_template(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        template_style: str,
        **kwargs
    ) -> str:
        """Generate custom template based on query and documents."""
        if not self.llm:
            return self.templates[PromptTemplateType.STANDARD.value]
        
        # Format documents
        docs_text = "\n\n".join(
            f"Document {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        )
        
        prompt = f"""
        Given the following query and documents, generate a custom prompt template.
        The template should follow the {template_style} style and effectively use the provided context.
        
        Query: {query}
        
        Documents:
        {docs_text}
        
        Generate a prompt template that:
        1. Effectively uses the provided context
        2. Follows the {template_style} style
        3. Includes placeholders for context and query
        4. Guides the model to provide a well-structured answer
        
        Template:
        """
        
        template = await self.llm.generate(prompt)
        
        # Ensure template has required placeholders
        if "{context}" not in template:
            template = template.replace("Context:", "{context}")
        if "{query}" not in template:
            template = template.replace("Question:", "{query}")
        
        return template

    async def optimize_template(
        self,
        template: str,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Optimize template based on query and documents."""
        if not self.llm:
            return template
        
        # Format documents
        docs_text = "\n\n".join(
            f"Document {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        )
        
        prompt = f"""
        Given the following template, query, and documents, optimize the template to better handle the specific case.
        Focus on improving clarity, relevance, and effectiveness.
        
        Current template:
        {template}
        
        Query: {query}
        
        Documents:
        {docs_text}
        
        Optimized template:
        """
        
        optimized = await self.llm.generate(prompt)
        
        # Ensure optimized template has required placeholders
        if "{context}" not in optimized:
            optimized = optimized.replace("Context:", "{context}")
        if "{query}" not in optimized:
            optimized = optimized.replace("Question:", "{query}")
        
        return optimized

    async def analyze_template_effectiveness(
        self,
        template: str,
        query: str,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze template effectiveness."""
        if not self.llm:
            return {
                "effectiveness": 0.0,
                "analysis": "LLM required for analysis"
            }
        
        # Format documents
        docs_text = "\n\n".join(
            f"Document {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        )
        
        prompt = f"""
        Analyze the effectiveness of the following prompt template for the given query and documents.
        Consider clarity, relevance, and potential for generating good answers.
        
        Template:
        {template}
        
        Query: {query}
        
        Documents:
        {docs_text}
        
        Analysis:
        1. Clarity:
        2. Relevance:
        3. Structure:
        4. Potential issues:
        5. Suggestions for improvement:
        
        Overall effectiveness score (0-1):
        """
        
        analysis = await self.llm.generate(prompt)
        
        # Extract effectiveness score
        try:
            score = float(
                re.search(r"Overall effectiveness score \(0-1\): ([\d.]+)", analysis)
                .group(1)
            )
        except (AttributeError, ValueError):
            score = 0.0
        
        return {
            "effectiveness": score,
            "analysis": analysis
        } 