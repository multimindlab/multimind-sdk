"""
Context optimization and advanced prompting for RAG systems.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from transformers import AutoTokenizer
from ..models.base import BaseLLM

@dataclass
class OptimizedContext:
    """Represents optimized context for generation."""
    chunks: List[Dict[str, Any]]
    total_tokens: int
    relevance_scores: List[float]
    prompt_template: str
    few_shot_examples: Optional[List[Dict[str, str]]] = None

class PromptTemplate(Enum):
    """Different prompt templates for various use cases."""
    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STRUCTURED = "structured"
    FEW_SHOT = "few_shot"
    ANALYTICAL = "analytical"

class OptimizationStrategy(Enum):
    """Strategies for context optimization."""
    RELEVANCE = "relevance"
    TOKEN_BUDGET = "token_budget"
    FEW_SHOT = "few_shot"
    HYBRID = "hybrid"

class ContextOptimizer:
    """Optimizes context based on relevance, token budget, and strategy."""

    def __init__(
        self,
        model: BaseLLM,
        max_tokens: int = 2000,
        relevance_threshold: float = 0.7,
        strategy: OptimizationStrategy = OptimizationStrategy.RELEVANCE,
        **kwargs
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.relevance_threshold = relevance_threshold
        self.strategy = strategy
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Default tokenizer
        self.kwargs = kwargs

    async def optimize_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> OptimizedContext:
        """
        Optimize context based on relevance and token budget.
        
        Args:
            query: User query
            context_chunks: List of context chunks
            max_tokens: Optional override for max tokens
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized context for generation
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Calculate relevance scores
        relevance_scores = await self._calculate_relevance_scores(query, context_chunks)
        
        # Filter and sort chunks by relevance
        filtered_chunks = [
            (chunk, score)
            for chunk, score in zip(context_chunks, relevance_scores)
            if score >= self.relevance_threshold
        ]
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Select chunks within token budget
        selected_chunks = []
        total_tokens = 0
        
        for chunk, score in filtered_chunks:
            chunk_tokens = len(self.tokenizer.encode(chunk["text"]))
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        
        return OptimizedContext(
            chunks=selected_chunks,
            total_tokens=total_tokens,
            relevance_scores=[score for _, score in filtered_chunks[:len(selected_chunks)]],
            prompt_template=kwargs.get("prompt_template", PromptTemplate.STANDARD.value)
        )

    async def _calculate_relevance_scores(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate relevance scores for chunks."""
        # Generate query embedding
        query_embedding = await self.model.embeddings([query])[0]
        
        # Calculate cosine similarity for each chunk
        scores = []
        for chunk in chunks:
            if "embedding" in chunk:
                similarity = self._cosine_similarity(
                    query_embedding,
                    chunk["embedding"]
                )
                scores.append(float(similarity))
            else:
                # If no embedding, generate one
                chunk_embedding = await self.model.embeddings([chunk["text"]])[0]
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                scores.append(float(similarity))
                chunk["embedding"] = chunk_embedding
        
        return scores

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class PromptGenerator:
    """Generates optimized prompts with various strategies."""

    def __init__(
        self,
        model: BaseLLM,
        default_template: PromptTemplate = PromptTemplate.STANDARD,
        **kwargs
    ):
        self.model = model
        self.default_template = default_template
        self.templates = self._initialize_templates()
        self.kwargs = kwargs

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize prompt templates."""
        return {
            PromptTemplate.STANDARD.value: """
            Context:
            {context}
            
            Question: {query}
            
            Answer:""",
            
            PromptTemplate.CHAIN_OF_THOUGHT.value: """
            Context:
            {context}
            
            Question: {query}
            
            Let's think about this step by step:
            1) First, let's understand what information we have in the context
            2) Then, let's analyze how this information relates to the question
            3) Finally, let's formulate a comprehensive answer
            
            Answer:""",
            
            PromptTemplate.STRUCTURED.value: """
            Context:
            {context}
            
            Question: {query}
            
            Please provide your answer in the following structure:
            1. Summary of relevant information
            2. Key points from the context
            3. Direct answer to the question
            4. Supporting evidence
            
            Answer:""",
            
            PromptTemplate.FEW_SHOT.value: """
            Here are some examples of similar questions and answers:
            
            {few_shot_examples}
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:""",
            
            PromptTemplate.ANALYTICAL.value: """
            Context:
            {context}
            
            Question: {query}
            
            Please analyze this question from multiple perspectives:
            1. What are the key facts from the context?
            2. What are the implications of these facts?
            3. Are there any limitations or uncertainties?
            4. What conclusions can we draw?
            
            Answer:"""
        }

    async def generate_prompt(
        self,
        query: str,
        context: OptimizedContext,
        template: Optional[PromptTemplate] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate optimized prompt based on context and template.
        
        Args:
            query: User query
            context: Optimized context
            template: Optional prompt template
            few_shot_examples: Optional few-shot examples
            **kwargs: Additional prompt generation parameters
            
        Returns:
            Generated prompt
        """
        template = template or self.default_template
        template_str = self.templates[template.value]
        
        # Format context
        context_text = "\n\n".join([
            f"Document {i+1} (Relevance: {score:.2f}):\n{chunk['text']}"
            for i, (chunk, score) in enumerate(zip(context.chunks, context.relevance_scores))
        ])
        
        # Format few-shot examples if provided
        few_shot_text = ""
        if few_shot_examples and template == PromptTemplate.FEW_SHOT:
            few_shot_text = "\n\n".join([
                f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['answer']}"
                for i, ex in enumerate(few_shot_examples)
            ])
        
        # Generate prompt
        prompt = template_str.format(
            context=context_text,
            query=query,
            few_shot_examples=few_shot_text
        )
        
        return prompt

    async def select_few_shot_examples(
        self,
        query: str,
        examples: List[Dict[str, str]],
        k: int = 3,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Select most relevant few-shot examples for the query.
        
        Args:
            query: User query
            examples: List of example questions and answers
            k: Number of examples to select
            **kwargs: Additional selection parameters
            
        Returns:
            Selected few-shot examples
        """
        if not examples:
            return []
        
        # Generate embeddings
        query_embedding = await self.model.embeddings([query])[0]
        example_embeddings = await self.model.embeddings([ex["question"] for ex in examples])
        
        # Calculate similarities
        similarities = [
            self._cosine_similarity(query_embedding, ex_emb)
            for ex_emb in example_embeddings
        ]
        
        # Select top k examples
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [examples[i] for i in top_k_indices]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class AdvancedRAGPrompting:
    """Combines context optimization and advanced prompting."""

    def __init__(
        self,
        model: BaseLLM,
        max_tokens: int = 2000,
        default_template: PromptTemplate = PromptTemplate.STANDARD,
        **kwargs
    ):
        self.context_optimizer = ContextOptimizer(model, max_tokens, **kwargs)
        self.prompt_generator = PromptGenerator(model, default_template, **kwargs)
        self.kwargs = kwargs

    async def prepare_generation(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        template: Optional[PromptTemplate] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Tuple[str, OptimizedContext]:
        """
        Prepare context and prompt for generation.
        
        Args:
            query: User query
            context_chunks: List of context chunks
            template: Optional prompt template
            few_shot_examples: Optional few-shot examples
            **kwargs: Additional preparation parameters
            
        Returns:
            Tuple of (generated prompt, optimized context)
        """
        # Optimize context
        optimized_context = await self.context_optimizer.optimize_context(
            query,
            context_chunks,
            **kwargs
        )
        
        # Select few-shot examples if provided
        selected_examples = None
        if few_shot_examples:
            selected_examples = await self.prompt_generator.select_few_shot_examples(
                query,
                few_shot_examples,
                **kwargs
            )
        
        # Generate prompt
        prompt = await self.prompt_generator.generate_prompt(
            query,
            optimized_context,
            template,
            selected_examples,
            **kwargs
        )
        
        return prompt, optimized_context