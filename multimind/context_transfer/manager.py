"""
Advanced Context Transfer Manager

Core logic for extracting, summarizing, and formatting conversation context
for transfer between different LLM providers across the entire AI ecosystem.
"""

import json
import logging
import re
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextTransferManager:
    """
    Advanced manager for transferring conversation context between different LLM providers.
    """
    
    def __init__(self):
        self.supported_models = {
            "chatgpt": "ChatGPT",
            "deepseek": "DeepSeek", 
            "claude": "Claude",
            "gemini": "Gemini",
            "mistral": "Mistral",
            "llama": "Llama",
            "cohere": "Cohere",
            "anthropic_claude": "Anthropic Claude",
            "openai_gpt4": "OpenAI GPT-4",
            "gpt4": "GPT-4",
            "gpt-4": "GPT-4",
            "gpt3": "GPT-3",
            "gpt-3": "GPT-3",
            "claude-3": "Claude-3",
            "claude-2": "Claude-2",
            "claude-1": "Claude-1"
        }
        
        # Advanced configuration
        self.config = {
            "max_context_length": 32000,
            "default_summary_length": 1000,
            "include_metadata": True,
            "preserve_formatting": True,
            "smart_truncation": True,
            "context_compression": False
        }
    
    def extract_context(self, messages: List[Dict], last_n: int = 5, 
                       smart_extraction: bool = True) -> List[Dict]:
        """
        Extract the last n turns from a conversation history with smart features.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            last_n: Number of recent turns to extract (default: 5)
            smart_extraction: Use intelligent extraction based on context importance
            
        Returns:
            List of the last n message dictionaries
        """
        if not messages:
            return []
        
        if smart_extraction:
            return self._smart_extract_context(messages, last_n)
        else:
            # Basic extraction
            last_n = min(last_n, len(messages))
            extracted_messages = messages[-last_n:]
            logger.info(f"Extracted {len(extracted_messages)} messages from conversation")
            return extracted_messages
    
    def _smart_extract_context(self, messages: List[Dict], last_n: int) -> List[Dict]:
        """Smart context extraction based on importance and relevance."""
        if len(messages) <= last_n:
            return messages
        
        # Prioritize recent messages but include important context
        recent_messages = messages[-last_n:]
        
        # Look for system messages or important context in earlier messages
        important_context = []
        for msg in messages[:-last_n]:
            if msg.get("role") == "system" or self._is_important_context(msg.get("content", "")):
                important_context.append(msg)
        
        # Combine important context with recent messages
        if important_context:
            # Take up to 2 important context messages
            context_to_include = important_context[-2:]
            combined = context_to_include + recent_messages[-(last_n - len(context_to_include)):]
            logger.info(f"Smart extraction: {len(context_to_include)} important + {len(combined) - len(context_to_include)} recent messages")
            return combined
        
        return recent_messages
    
    def _is_important_context(self, content: str) -> bool:
        """Determine if a message contains important context."""
        important_keywords = [
            "system", "setup", "configuration", "requirements", "constraints",
            "important", "note", "warning", "error", "critical", "essential"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in important_keywords)
    
    def summarize_context(self, messages: List[Dict], model: str = "gpt-3.5", 
                         summary_type: str = "concise") -> str:
        """
        Advanced context summarization with multiple strategies.
        
        Args:
            messages: List of message dictionaries
            model: Model to use for summarization (placeholder for future implementation)
            summary_type: Type of summary ("concise", "detailed", "structured")
            
        Returns:
            String summary of the conversation context
        """
        if not messages:
            return "No conversation context available."
        
        if summary_type == "structured":
            return self._create_structured_summary(messages)
        elif summary_type == "detailed":
            return self._create_detailed_summary(messages)
        else:
            return self._create_concise_summary(messages)
    
    def _create_concise_summary(self, messages: List[Dict]) -> str:
        """Create a concise summary focusing on key points."""
        summary_parts = []
        
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            # Clean and truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
            
            if role == "user":
                summary_parts.append(f"User: {content}")
            elif role == "assistant":
                summary_parts.append(f"Assistant: {content}")
            elif role == "system":
                summary_parts.append(f"System: {content}")
        
        summary = "\n".join(summary_parts)
        logger.info(f"Generated concise summary with {len(summary_parts)} parts")
        return summary
    
    def _create_detailed_summary(self, messages: List[Dict]) -> str:
        """Create a detailed summary with full context."""
        summary_parts = []
        
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "user":
                summary_parts.append(f"User (Message {i+1}): {content}")
            elif role == "assistant":
                summary_parts.append(f"Assistant (Response {i+1}): {content}")
            elif role == "system":
                summary_parts.append(f"System Configuration: {content}")
        
        summary = "\n\n".join(summary_parts)
        logger.info(f"Generated detailed summary with {len(summary_parts)} parts")
        return summary
    
    def _create_structured_summary(self, messages: List[Dict]) -> str:
        """Create a structured summary with sections."""
        user_messages = []
        assistant_messages = []
        system_messages = []
        
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "user":
                user_messages.append(content)
            elif role == "assistant":
                assistant_messages.append(content)
            elif role == "system":
                system_messages.append(content)
        
        summary_parts = []
        
        if system_messages:
            summary_parts.append("System Context:")
            summary_parts.extend([f"- {msg}" for msg in system_messages])
            summary_parts.append("")
        
        summary_parts.append("Conversation Flow:")
        for i in range(max(len(user_messages), len(assistant_messages))):
            if i < len(user_messages):
                summary_parts.append(f"User: {user_messages[i]}")
            if i < len(assistant_messages):
                summary_parts.append(f"Assistant: {assistant_messages[i]}")
            summary_parts.append("")
        
        summary = "\n".join(summary_parts).strip()
        logger.info(f"Generated structured summary with {len(user_messages)} user and {len(assistant_messages)} assistant messages")
        return summary
    
    def load_conversation_from_file(self, file_path: str, 
                                   format_type: str = "auto") -> List[Dict]:
        """
        Load conversation history from various file formats.
        
        Args:
            file_path: Path to the file containing conversation history
            format_type: Format type ("auto", "json", "txt", "markdown")
            
        Returns:
            List of message dictionaries
        """
        try:
            if format_type == "auto":
                format_type = self._detect_file_format(file_path)
            
            if format_type == "json":
                return self._load_json_conversation(file_path)
            elif format_type == "txt":
                return self._load_text_conversation(file_path)
            elif format_type == "markdown":
                return self._load_markdown_conversation(file_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading conversation from {file_path}: {e}")
            raise
    
    def _detect_file_format(self, file_path: str) -> str:
        """Auto-detect file format based on extension and content."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".json":
            return "json"
        elif extension in [".md", ".markdown"]:
            return "markdown"
        else:
            return "txt"
    
    def _load_json_conversation(self, file_path: str) -> List[Dict]:
        """Load conversation from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
        elif isinstance(data, dict) and "conversation" in data:
            messages = data["conversation"]
        else:
            raise ValueError("Invalid JSON structure. Expected list of messages or dict with 'messages' key.")
        
        logger.info(f"Loaded {len(messages)} messages from JSON file {file_path}")
        return messages
    
    def _load_text_conversation(self, file_path: str) -> List[Dict]:
        """Load conversation from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple text parsing - assumes alternating User/Assistant format
        lines = content.strip().split('\n')
        messages = []
        current_role = None
        current_content = []
        
        for line in lines:
            if line.startswith('User:') or line.startswith('Assistant:') or line.startswith('System:'):
                # Save previous message
                if current_role and current_content:
                    messages.append({
                        "role": current_role.lower(),
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new message
                if line.startswith('User:'):
                    current_role = "user"
                elif line.startswith('Assistant:'):
                    current_role = "assistant"
                elif line.startswith('System:'):
                    current_role = "system"
                
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            else:
                current_content.append(line)
        
        # Add last message
        if current_role and current_content:
            messages.append({
                "role": current_role.lower(),
                "content": '\n'.join(current_content).strip()
            })
        
        logger.info(f"Loaded {len(messages)} messages from text file {file_path}")
        return messages
    
    def _load_markdown_conversation(self, file_path: str) -> List[Dict]:
        """Load conversation from markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse markdown format
        messages = []
        lines = content.split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            if line.startswith('### User:') or line.startswith('### Assistant:') or line.startswith('### System:'):
                # Save previous message
                if current_role and current_content:
                    messages.append({
                        "role": current_role.lower(),
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new message
                if line.startswith('### User:'):
                    current_role = "user"
                elif line.startswith('### Assistant:'):
                    current_role = "assistant"
                elif line.startswith('### System:'):
                    current_role = "system"
                
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            else:
                current_content.append(line)
        
        # Add last message
        if current_role and current_content:
            messages.append({
                "role": current_role.lower(),
                "content": '\n'.join(current_content).strip()
            })
        
        logger.info(f"Loaded {len(messages)} messages from markdown file {file_path}")
        return messages
    
    def save_formatted_prompt(self, content: str, output_file: str, 
                             format_type: str = "txt") -> None:
        """
        Save the formatted prompt to various file formats.
        
        Args:
            content: The formatted prompt content
            output_file: Path to the output file
            format_type: Output format ("txt", "json", "markdown")
        """
        try:
            if format_type == "json":
                self._save_json_prompt(content, output_file)
            elif format_type == "markdown":
                self._save_markdown_prompt(content, output_file)
            else:
                self._save_text_prompt(content, output_file)
            
            logger.info(f"Formatted prompt saved to {output_file} in {format_type} format")
            
        except Exception as e:
            logger.error(f"Error saving formatted prompt to {output_file}: {e}")
            raise
    
    def _save_text_prompt(self, content: str, output_file: str) -> None:
        """Save prompt as plain text."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_json_prompt(self, content: str, output_file: str) -> None:
        """Save prompt as JSON with metadata."""
        prompt_data = {
            "prompt": content,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "format": "json",
                "length": len(content)
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    
    def _save_markdown_prompt(self, content: str, output_file: str) -> None:
        """Save prompt as markdown."""
        markdown_content = f"""# Formatted Prompt

## Content

{content}

---
*Generated by MultiMind Context Transfer*
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def transfer_context(
        self,
        from_model: str,
        to_model: str,
        input_file: str,
        output_file: str,
        last_n: int = 5,
        include_summary: bool = True,
        summary_type: str = "concise",
        smart_extraction: bool = True,
        output_format: str = "txt",
        **kwargs
    ) -> str:
        """
        Advanced context transfer with comprehensive options.
        
        Args:
            from_model: Source model name
            to_model: Target model name
            input_file: Path to input file
            output_file: Path to output file
            last_n: Number of recent turns to extract
            include_summary: Whether to include conversation summary
            summary_type: Type of summary ("concise", "detailed", "structured")
            smart_extraction: Use intelligent context extraction
            output_format: Output format ("txt", "json", "markdown")
            **kwargs: Additional formatting options for the target model
            
        Returns:
            The formatted prompt content
        """
        # Load conversation
        messages = self.load_conversation_from_file(input_file)
        
        # Extract context
        extracted_messages = self.extract_context(messages, last_n, smart_extraction)
        
        # Generate summary if requested
        if include_summary:
            summary = self.summarize_context(extracted_messages, summary_type=summary_type)
        else:
            summary = self.summarize_context(extracted_messages[-1:], summary_type="concise")
        
        # Format for target model
        formatted_prompt = self._format_for_target_model(to_model, summary, from_model, **kwargs)
        
        # Save to file
        self.save_formatted_prompt(formatted_prompt, output_file, output_format)
        
        return formatted_prompt
    
    def _format_for_target_model(self, target_model: str, summary: str, source_model: str, **kwargs) -> str:
        """
        Format the summary for the target model with advanced options.
        
        Args:
            target_model: Target model name
            summary: Conversation summary
            source_model: Source model name
            **kwargs: Additional formatting options
            
        Returns:
            Formatted prompt for the target model
        """
        target_model_lower = target_model.lower().replace(" ", "_").replace("-", "_")
        
        # Get the appropriate adapter
        from .adapters import AdapterFactory
        try:
            adapter = AdapterFactory.get_adapter(target_model_lower)
            return adapter.format_context(summary, source_model, **kwargs)
        except ValueError:
            # Fallback to generic formatting
            return self._format_generic(summary, source_model, target_model, **kwargs)
    
    def _format_generic(self, summary: str, source_model: str, target_model: str, **kwargs) -> str:
        """Generic format for unknown models with advanced options."""
        include_metadata = kwargs.get('include_metadata', self.config['include_metadata'])
        
        prompt = f"""You are {target_model}, an AI assistant.

A user was previously working with {source_model} on the following conversation:

{summary}

Please continue helping the user from where they left off. Maintain the context and provide helpful responses."""
        
        if include_metadata:
            prompt += f"\n\n---\nContext transferred from {source_model} to {target_model} using MultiMind SDK"
        
        return prompt
    
    def get_supported_models(self) -> List[str]:
        """Get list of all supported models."""
        return list(self.supported_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        from .adapters import AdapterFactory
        try:
            return AdapterFactory.get_model_capabilities(model_name)
        except ValueError:
            return {
                "name": model_name,
                "supported": False,
                "note": "Model not found in adapter registry"
            }
    
    def list_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all supported models."""
        from .adapters import AdapterFactory
        return AdapterFactory.list_all_capabilities() 