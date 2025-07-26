"""
Advanced prompting system for RAG with dynamic generation and optimization.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np
from datetime import datetime
import json
from ..models.base import BaseLLM

@dataclass
class PromptTemplate:
    """Template for prompt generation."""
    template: str
    variables: List[str]
    metadata: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class PromptContext:
    """Context for prompt generation."""
    query: str
    retrieved_documents: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    system_state: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class GeneratedPrompt:
    """Generated prompt with metadata."""
    prompt: str
    template: PromptTemplate
    context: PromptContext
    metadata: Dict[str, Any]
    reasoning: Optional[str] = None

class PromptType(Enum):
    """Types of prompts."""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    REASONING = "reasoning"
    REFINEMENT = "refinement"
    EVALUATION = "evaluation"

class PromptStrategy(Enum):
    """Strategies for prompt generation."""
    DIRECT = "direct"
    STEP_BY_STEP = "step_by_step"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_CONSISTENCY = "self_consistency"
    TREE_OF_THOUGHT = "tree_of_thought"

class AdvancedPrompting:
    """Advanced prompting system for RAG."""

    def __init__(
        self,
        model: BaseLLM,
        templates: Optional[Dict[str, PromptTemplate]] = None,
        **kwargs
    ):
        """
        Initialize advanced prompting system.
        
        Args:
            model: Language model
            templates: Optional prompt templates
            **kwargs: Additional parameters
        """
        self.model = model
        self.templates = templates or {}
        self.kwargs = kwargs

    async def generate_prompt(
        self,
        prompt_type: PromptType,
        context: PromptContext,
        strategy: PromptStrategy = PromptStrategy.DIRECT,
        **kwargs
    ) -> GeneratedPrompt:
        """
        Generate prompt based on type and strategy.
        
        Args:
            prompt_type: Type of prompt to generate
            context: Context for prompt generation
            strategy: Strategy to use
            **kwargs: Additional parameters
            
        Returns:
            Generated prompt
        """
        # Get template
        template = self._get_template(prompt_type)
        
        # Generate prompt based on strategy
        if strategy == PromptStrategy.DIRECT:
            prompt = await self._generate_direct_prompt(
                template,
                context,
                **kwargs
            )
        elif strategy == PromptStrategy.STEP_BY_STEP:
            prompt = await self._generate_step_by_step_prompt(
                template,
                context,
                **kwargs
            )
        elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            prompt = await self._generate_chain_of_thought_prompt(
                template,
                context,
                **kwargs
            )
        elif strategy == PromptStrategy.SELF_CONSISTENCY:
            prompt = await self._generate_self_consistency_prompt(
                template,
                context,
                **kwargs
            )
        else:  # TREE_OF_THOUGHT
            prompt = await self._generate_tree_of_thought_prompt(
                template,
                context,
                **kwargs
            )
        
        return GeneratedPrompt(
            prompt=prompt,
            template=template,
            context=context,
            metadata=kwargs,
            reasoning=await self._generate_reasoning(
                prompt,
                context,
                **kwargs
            )
        )

    async def optimize_context(
        self,
        context: PromptContext,
        **kwargs
    ) -> PromptContext:
        """
        Optimize context for prompt generation.
        
        Args:
            context: Context to optimize
            **kwargs: Additional parameters
            
        Returns:
            Optimized context
        """
        # Optimize retrieved documents
        optimized_docs = await self._optimize_documents(
            context.retrieved_documents,
            context.query,
            **kwargs
        )
        
        # Optimize conversation history
        optimized_history = await self._optimize_history(
            context.conversation_history,
            context.query,
            **kwargs
        )
        
        # Update system state
        optimized_state = await self._optimize_state(
            context.system_state,
            context.query,
            **kwargs
        )
        
        return PromptContext(
            query=context.query,
            retrieved_documents=optimized_docs,
            conversation_history=optimized_history,
            system_state=optimized_state,
            metadata=context.metadata
        )

    async def refine_prompt(
        self,
        prompt: GeneratedPrompt,
        feedback: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> GeneratedPrompt:
        """
        Refine prompt based on feedback.
        
        Args:
            prompt: Prompt to refine
            feedback: Optional feedback
            **kwargs: Additional parameters
            
        Returns:
            Refined prompt
        """
        # Generate refinement prompt
        refinement_prompt = f"""
        Refine the following prompt based on feedback.
        Consider:
        1. Clarity improvement
        2. Context utilization
        3. Constraint satisfaction
        4. Feedback incorporation
        
        Original Prompt:
        {prompt.prompt}
        
        Feedback:
        {feedback or "No feedback provided"}
        
        Context:
        {json.dumps(prompt.context.metadata, indent=2)}
        """
        
        # Get refined prompt
        refined_prompt = await self.model.generate(
            prompt=refinement_prompt,
            **kwargs
        )
        
        return GeneratedPrompt(
            prompt=refined_prompt,
            template=prompt.template,
            context=prompt.context,
            metadata={**prompt.metadata, "refined": True},
            reasoning=await self._generate_reasoning(
                refined_prompt,
                prompt.context,
                **kwargs
            )
        )

    def _get_template(
        self,
        prompt_type: PromptType
    ) -> PromptTemplate:
        """Get template for prompt type."""
        if prompt_type not in self.templates:
            # Create default template
            template = self._create_default_template(prompt_type)
            self.templates[prompt_type] = template
        
        return self.templates[prompt_type]

    def _create_default_template(
        self,
        prompt_type: PromptType
    ) -> PromptTemplate:
        """Create default template for prompt type."""
        if prompt_type == PromptType.RETRIEVAL:
            template = """
            Find relevant information from the following documents.
            Consider:
            1. Query relevance
            2. Information value
            3. Context coverage
            
            Query: {query}
            
            Documents:
            {documents}
            
            Conversation History:
            {history}
            
            System State:
            {state}
            """
        elif prompt_type == PromptType.GENERATION:
            template = """
            Generate a response based on the following context.
            Consider:
            1. Query addressing
            2. Information accuracy
            3. Response coherence
            
            Query: {query}
            
            Retrieved Information:
            {documents}
            
            Conversation History:
            {history}
            
            System State:
            {state}
            """
        elif prompt_type == PromptType.REASONING:
            template = """
            Analyze the following information and provide reasoning.
            Consider:
            1. Logical flow
            2. Evidence support
            3. Conclusion validity
            
            Query: {query}
            
            Information:
            {documents}
            
            Context:
            {history}
            
            State:
            {state}
            """
        elif prompt_type == PromptType.REFINEMENT:
            template = """
            Refine the following response.
            Consider:
            1. Clarity improvement
            2. Accuracy enhancement
            3. Coherence strengthening
            
            Original Response:
            {response}
            
            Context:
            {documents}
            
            History:
            {history}
            
            State:
            {state}
            """
        else:  # EVALUATION
            template = """
            Evaluate the following response.
            Consider:
            1. Answer quality
            2. Information accuracy
            3. Response coherence
            
            Query: {query}
            
            Response:
            {response}
            
            Context:
            {documents}
            
            History:
            {history}
            
            State:
            {state}
            """
        
        return PromptTemplate(
            template=template,
            variables=["query", "documents", "history", "state"],
            metadata={"type": prompt_type},
            constraints={
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )

    async def _generate_direct_prompt(
        self,
        template: PromptTemplate,
        context: PromptContext,
        **kwargs
    ) -> str:
        """Generate direct prompt."""
        # Format template with context
        prompt = template.template.format(
            query=context.query,
            documents=self._format_documents(context.retrieved_documents),
            history=self._format_history(context.conversation_history),
            state=json.dumps(context.system_state, indent=2)
        )
        
        return prompt

    async def _generate_step_by_step_prompt(
        self,
        template: PromptTemplate,
        context: PromptContext,
        **kwargs
    ) -> str:
        """Generate step-by-step prompt."""
        # Generate steps
        steps_prompt = f"""
        Break down the following task into steps.
        Consider:
        1. Logical progression
        2. Information needs
        3. Context utilization
        
        Task:
        {template.template}
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        steps = await self.model.generate(prompt=steps_prompt, **kwargs)
        
        # Generate step-by-step prompt
        prompt = f"""
        Follow these steps to complete the task:
        
        {steps}
        
        For each step:
        1. Consider the context
        2. Use available information
        3. Provide reasoning
        4. Generate output
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        return prompt

    async def _generate_chain_of_thought_prompt(
        self,
        template: PromptTemplate,
        context: PromptContext,
        **kwargs
    ) -> str:
        """Generate chain-of-thought prompt."""
        # Generate reasoning chain
        chain_prompt = f"""
        Generate a chain of thought for the following task.
        Consider:
        1. Logical reasoning
        2. Information processing
        3. Conclusion derivation
        
        Task:
        {template.template}
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        chain = await self.model.generate(prompt=chain_prompt, **kwargs)
        
        # Generate chain-of-thought prompt
        prompt = f"""
        Follow this chain of thought to complete the task:
        
        {chain}
        
        For each step in the chain:
        1. Explain your reasoning
        2. Use relevant information
        3. Draw conclusions
        4. Connect to next step
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        return prompt

    async def _generate_self_consistency_prompt(
        self,
        template: PromptTemplate,
        context: PromptContext,
        **kwargs
    ) -> str:
        """Generate self-consistency prompt."""
        # Generate multiple perspectives
        perspectives_prompt = f"""
        Generate multiple perspectives for the following task.
        Consider:
        1. Different approaches
        2. Various interpretations
        3. Alternative solutions
        
        Task:
        {template.template}
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        perspectives = await self.model.generate(
            prompt=perspectives_prompt,
            **kwargs
        )
        
        # Generate self-consistency prompt
        prompt = f"""
        Consider multiple perspectives and ensure consistency:
        
        {perspectives}
        
        For each perspective:
        1. Analyze independently
        2. Compare with others
        3. Identify consensus
        4. Resolve conflicts
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        return prompt

    async def _generate_tree_of_thought_prompt(
        self,
        template: PromptTemplate,
        context: PromptContext,
        **kwargs
    ) -> str:
        """Generate tree-of-thought prompt."""
        # Generate thought tree
        tree_prompt = f"""
        Generate a tree of thoughts for the following task.
        Consider:
        1. Multiple branches
        2. Decision points
        3. Outcome evaluation
        
        Task:
        {template.template}
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        tree = await self.model.generate(prompt=tree_prompt, **kwargs)
        
        # Generate tree-of-thought prompt
        prompt = f"""
        Explore this tree of thoughts to complete the task:
        
        {tree}
        
        For each branch:
        1. Evaluate options
        2. Consider consequences
        3. Choose best path
        4. Track decisions
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        return prompt

    async def _optimize_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Optimize retrieved documents."""
        # Generate optimization prompt
        optimization_prompt = f"""
        Optimize the following documents for relevance and utility.
        Consider:
        1. Query relevance
        2. Information value
        3. Redundancy removal
        4. Context coherence
        
        Query: {query}
        
        Documents:
        {self._format_documents(documents)}
        """
        
        # Get optimization instructions
        instructions = await self.model.generate(
            prompt=optimization_prompt,
            **kwargs
        )
        
        # Apply optimization
        optimized_docs = []
        for doc in documents:
            # Check relevance
            relevance_prompt = f"""
            Evaluate relevance of this document to the query.
            
            Query: {query}
            
            Document:
            {json.dumps(doc, indent=2)}
            
            Instructions:
            {instructions}
            """
            
            relevance = await self.model.generate(
                prompt=relevance_prompt,
                **kwargs
            )
            
            if "relevant" in relevance.lower():
                # Optimize content
                content_prompt = f"""
                Optimize this document's content.
                
                Document:
                {json.dumps(doc, indent=2)}
                
                Instructions:
                {instructions}
                """
                
                optimized_content = await self.model.generate(
                    prompt=content_prompt,
                    **kwargs
                )
                
                optimized_docs.append({
                    **doc,
                    "content": optimized_content,
                    "optimized": True
                })
        
        return optimized_docs

    async def _optimize_history(
        self,
        history: List[Dict[str, Any]],
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Optimize conversation history."""
        # Generate optimization prompt
        optimization_prompt = f"""
        Optimize the conversation history for relevance and utility.
        Consider:
        1. Query relevance
        2. Information value
        3. Context coherence
        4. Redundancy removal
        
        Query: {query}
        
        History:
        {self._format_history(history)}
        """
        
        # Get optimization instructions
        instructions = await self.model.generate(
            prompt=optimization_prompt,
            **kwargs
        )
        
        # Apply optimization
        optimized_history = []
        for turn in history:
            # Check relevance
            relevance_prompt = f"""
            Evaluate relevance of this conversation turn.
            
            Query: {query}
            
            Turn:
            {json.dumps(turn, indent=2)}
            
            Instructions:
            {instructions}
            """
            
            relevance = await self.model.generate(
                prompt=relevance_prompt,
                **kwargs
            )
            
            if "relevant" in relevance.lower():
                # Optimize content
                content_prompt = f"""
                Optimize this conversation turn.
                
                Turn:
                {json.dumps(turn, indent=2)}
                
                Instructions:
                {instructions}
                """
                
                optimized_content = await self.model.generate(
                    prompt=content_prompt,
                    **kwargs
                )
                
                optimized_history.append({
                    **turn,
                    "content": optimized_content,
                    "optimized": True
                })
        
        return optimized_history

    async def _optimize_state(
        self,
        state: Dict[str, Any],
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize system state."""
        # Generate optimization prompt
        optimization_prompt = f"""
        Optimize the system state for relevance and utility.
        Consider:
        1. Query relevance
        2. State coherence
        3. Information value
        4. Context alignment
        
        Query: {query}
        
        State:
        {json.dumps(state, indent=2)}
        """
        
        # Get optimization instructions
        instructions = await self.model.generate(
            prompt=optimization_prompt,
            **kwargs
        )
        
        # Apply optimization
        optimized_state = {}
        for key, value in state.items():
            # Check relevance
            relevance_prompt = f"""
            Evaluate relevance of this state value.
            
            Query: {query}
            
            Key: {key}
            Value: {json.dumps(value, indent=2)}
            
            Instructions:
            {instructions}
            """
            
            relevance = await self.model.generate(
                prompt=relevance_prompt,
                **kwargs
            )
            
            if "relevant" in relevance.lower():
                # Optimize value
                value_prompt = f"""
                Optimize this state value.
                
                Key: {key}
                Value: {json.dumps(value, indent=2)}
                
                Instructions:
                {instructions}
                """
                
                optimized_value = await self.model.generate(
                    prompt=value_prompt,
                    **kwargs
                )
                
                optimized_state[key] = {
                    "value": optimized_value,
                    "optimized": True
                }
        
        return optimized_state

    async def _generate_reasoning(
        self,
        prompt: str,
        context: PromptContext,
        **kwargs
    ) -> str:
        """Generate reasoning for prompt."""
        reasoning_prompt = f"""
        Explain the reasoning behind this prompt.
        Consider:
        1. Prompt structure
        2. Context utilization
        3. Information flow
        4. Expected outcomes
        
        Prompt:
        {prompt}
        
        Context:
        Query: {context.query}
        Documents: {self._format_documents(context.retrieved_documents)}
        History: {self._format_history(context.conversation_history)}
        State: {json.dumps(context.system_state, indent=2)}
        """
        
        return await self.model.generate(prompt=reasoning_prompt, **kwargs)

    def _format_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> str:
        """Format documents for prompt."""
        return "\n\n".join(
            f"Document {i+1}:\n{json.dumps(doc, indent=2)}"
            for i, doc in enumerate(documents)
        )

    def _format_history(
        self,
        history: List[Dict[str, Any]]
    ) -> str:
        """Format conversation history."""
        if not history:
            return ""
        
        formatted = []
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)

    async def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt for routing and optimization."""
        # Simple prompt analysis - can be enhanced with more sophisticated analysis
        analysis = {
            "task_type": "text_generation",
            "complexity": 5,
            "domain": "general",
            "language": "en",
            "context_length": len(prompt.split()),
            "has_code": "```" in prompt or "def " in prompt or "class " in prompt,
            "has_math": any(op in prompt for op in ["+", "-", "*", "/", "=", ">", "<"]),
            "has_questions": "?" in prompt,
            "sentiment": "neutral"
        }
        
        # Detect task type
        if "?" in prompt:
            analysis["task_type"] = "question_answering"
        elif "translate" in prompt.lower():
            analysis["task_type"] = "translation"
        elif "summarize" in prompt.lower():
            analysis["task_type"] = "summarization"
        elif analysis["has_code"]:
            analysis["task_type"] = "code_generation"
        elif analysis["has_math"]:
            analysis["task_type"] = "mathematical_reasoning"
        
        # Detect complexity
        word_count = len(prompt.split())
        if word_count > 100:
            analysis["complexity"] = 8
        elif word_count > 50:
            analysis["complexity"] = 6
        elif word_count > 20:
            analysis["complexity"] = 4
        else:
            analysis["complexity"] = 2
        
        # Detect domain
        domain_keywords = {
            "medical": ["health", "medical", "patient", "diagnosis", "treatment"],
            "legal": ["law", "legal", "contract", "regulation", "compliance"],
            "technical": ["code", "programming", "algorithm", "system", "technical"],
            "creative": ["story", "creative", "imagine", "write", "poem"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                analysis["domain"] = domain
                break
        
        return analysis 