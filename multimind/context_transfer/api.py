"""
Advanced Context Transfer API

Provides comprehensive API for Chrome extension integration and advanced
context transfer features across the entire LLM ecosystem.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

from .manager import ContextTransferManager
from .adapters import AdapterFactory


logger = logging.getLogger(__name__)


class ContextTransferAPI:
    """
    Advanced API for context transfer operations.
    
    Provides comprehensive functionality for Chrome extensions and other
    applications to transfer conversation context between LLM providers.
    """
    
    def __init__(self):
        self.manager = ContextTransferManager()
        self.supported_formats = ["json", "txt", "markdown"]
        self.supported_models = AdapterFactory.get_supported_models()
    
    def transfer_context_api(
        self,
        source_model: str,
        target_model: str,
        conversation_data: Union[str, List[Dict], Dict],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main API method for context transfer.
        
        Args:
            source_model: Source model name
            target_model: Target model name
            conversation_data: Conversation data (file path, list of messages, or dict)
            options: Transfer options
            
        Returns:
            Dictionary containing formatted prompt and metadata
        """
        try:
            # Set default options
            default_options = {
                "last_n": 5,
                "include_summary": True,
                "summary_type": "concise",
                "smart_extraction": True,
                "output_format": "txt",
                "include_metadata": True,
                "include_code_context": False,
                "include_reasoning": False,
                "include_safety": False,
                "include_creativity": False,
                "include_examples": False,
                "include_step_by_step": False,
                "include_multimodal": False,
                "include_web_search": False
            }
            
            if options:
                default_options.update(options)
            
            # Process conversation data
            messages = self._process_conversation_data(conversation_data)
            
            # Extract context
            extracted_messages = self.manager.extract_context(
                messages, 
                default_options["last_n"], 
                default_options["smart_extraction"]
            )
            
            # Generate summary
            if default_options["include_summary"]:
                summary = self.manager.summarize_context(
                    extracted_messages, 
                    summary_type=default_options["summary_type"]
                )
            else:
                summary = self.manager.summarize_context(
                    extracted_messages[-1:], 
                    summary_type="concise"
                )
            
            # Format for target model
            formatting_options = {
                k: v for k, v in default_options.items() 
                if k.startswith("include_") and k != "include_metadata"
            }
            
            formatted_prompt = self.manager._format_for_target_model(
                target_model, summary, source_model, **formatting_options
            )
            
            # Prepare response
            response = {
                "success": True,
                "formatted_prompt": formatted_prompt,
                "metadata": {
                    "source_model": source_model,
                    "target_model": target_model,
                    "summary_type": default_options["summary_type"],
                    "smart_extraction": default_options["smart_extraction"],
                    "messages_processed": len(messages),
                    "messages_extracted": len(extracted_messages),
                    "prompt_length": len(formatted_prompt),
                    "created_at": datetime.now().isoformat(),
                    "output_format": default_options["output_format"]
                }
            }
            
            if default_options["include_metadata"]:
                response["metadata"]["model_capabilities"] = {
                    "source": self.manager.get_model_info(source_model),
                    "target": self.manager.get_model_info(target_model)
                }
            
            logger.info(f"Context transfer completed: {source_model} -> {target_model}")
            return response
            
        except Exception as e:
            logger.error(f"Context transfer failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _process_conversation_data(self, data: Union[str, List[Dict], Dict]) -> List[Dict]:
        """
        Process conversation data from various formats.
        
        Args:
            data: Conversation data in various formats
            
        Returns:
            List of message dictionaries
        """
        if isinstance(data, str):
            # Assume it's a file path
            return self.manager.load_conversation_from_file(data)
        elif isinstance(data, list):
            # Assume it's a list of message dictionaries
            return data
        elif isinstance(data, dict):
            # Handle different dict structures
            if "messages" in data:
                return data["messages"]
            elif "conversation" in data:
                return data["conversation"]
            else:
                # Assume it's a single message
                return [data]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def get_supported_models(self) -> Dict[str, Any]:
        """
        Get comprehensive information about supported models.
        
        Returns:
            Dictionary with model information and capabilities
        """
        try:
            capabilities = AdapterFactory.list_all_capabilities()
            
            return {
                "success": True,
                "models": capabilities,
                "total_models": len(capabilities),
                "supported_formats": self.supported_formats,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "api_version": "2.0"
                }
            }
        except Exception as e:
            logger.error(f"Failed to get supported models: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed capabilities for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model capabilities
        """
        try:
            capabilities = AdapterFactory.get_model_capabilities(model_name)
            
            return {
                "success": True,
                "model": model_name,
                "capabilities": capabilities,
                "metadata": {
                    "generated_at": datetime.now().isoformat()
                }
            }
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "ValueError"
            }
        except Exception as e:
            logger.error(f"Failed to get model capabilities: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def validate_conversation_format(self, data: Union[str, List[Dict], Dict]) -> Dict[str, Any]:
        """
        Validate conversation data format.
        
        Args:
            data: Conversation data to validate
            
        Returns:
            Validation result with details
        """
        try:
            messages = self._process_conversation_data(data)
            
            # Analyze message structure
            analysis = {
                "total_messages": len(messages),
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0,
                "unknown_messages": 0,
                "average_message_length": 0,
                "has_system_context": False
            }
            
            total_length = 0
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                if role == "user":
                    analysis["user_messages"] += 1
                elif role == "assistant":
                    analysis["assistant_messages"] += 1
                elif role == "system":
                    analysis["system_messages"] += 1
                    analysis["has_system_context"] = True
                else:
                    analysis["unknown_messages"] += 1
                
                total_length += len(content)
            
            if analysis["total_messages"] > 0:
                analysis["average_message_length"] = total_length / analysis["total_messages"]
            
            return {
                "success": True,
                "valid": True,
                "analysis": analysis,
                "recommendations": self._generate_recommendations(analysis)
            }
            
        except Exception as e:
            return {
                "success": False,
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on conversation analysis."""
        recommendations = []
        
        if analysis["total_messages"] == 0:
            recommendations.append("No messages found in conversation data")
        
        if analysis["user_messages"] == 0:
            recommendations.append("No user messages found - ensure conversation has user input")
        
        if analysis["assistant_messages"] == 0:
            recommendations.append("No assistant messages found - ensure conversation has AI responses")
        
        if analysis["unknown_messages"] > 0:
            recommendations.append(f"Found {analysis['unknown_messages']} messages with unknown roles")
        
        if analysis["average_message_length"] > 1000:
            recommendations.append("Long messages detected - consider using smart extraction")
        
        if analysis["total_messages"] > 20:
            recommendations.append("Large conversation detected - consider using smart extraction and detailed summary")
        
        if not analysis["has_system_context"]:
            recommendations.append("No system context found - consider adding system messages for better context")
        
        return recommendations
    
    def batch_transfer(
        self,
        transfers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform multiple context transfers in batch.
        
        Args:
            transfers: List of transfer configurations
            
        Returns:
            Dictionary with results for all transfers
        """
        results = []
        successful = 0
        failed = 0
        
        for i, transfer_config in enumerate(transfers):
            try:
                result = self.transfer_context_api(
                    source_model=transfer_config["source_model"],
                    target_model=transfer_config["target_model"],
                    conversation_data=transfer_config["conversation_data"],
                    options=transfer_config.get("options", {})
                )
                
                result["transfer_index"] = i
                results.append(result)
                
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "transfer_index": i
                })
                failed += 1
        
        return {
            "success": failed == 0,
            "total_transfers": len(transfers),
            "successful_transfers": successful,
            "failed_transfers": failed,
            "results": results,
            "metadata": {
                "completed_at": datetime.now().isoformat()
            }
        }
    
    def create_chrome_extension_config(self) -> Dict[str, Any]:
        """
        Generate configuration for Chrome extension integration.
        
        Returns:
            Configuration suitable for Chrome extension
        """
        return {
            "api_version": "2.0",
            "supported_models": self.supported_models,
            "supported_formats": self.supported_formats,
            "default_options": {
                "last_n": 5,
                "include_summary": True,
                "summary_type": "concise",
                "smart_extraction": True,
                "output_format": "txt"
            },
            "chrome_extension": {
                "manifest_version": 3,
                "permissions": ["activeTab", "storage"],
                "content_scripts": ["content.js"],
                "background_scripts": ["background.js"],
                "popup": "popup.html"
            },
            "endpoints": {
                "transfer": "/api/transfer",
                "models": "/api/models",
                "validate": "/api/validate",
                "batch": "/api/batch"
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sdk_version": "2.0.0"
            }
        }


# Convenience functions for easy integration
def quick_transfer(
    source_model: str,
    target_model: str,
    conversation_data: Union[str, List[Dict], Dict],
    **kwargs
) -> str:
    """
    Quick context transfer function for simple use cases.
    
    Args:
        source_model: Source model name
        target_model: Target model name
        conversation_data: Conversation data
        **kwargs: Additional options
        
    Returns:
        Formatted prompt string
    """
    api = ContextTransferAPI()
    result = api.transfer_context_api(source_model, target_model, conversation_data, kwargs)
    
    if result["success"]:
        return result["formatted_prompt"]
    else:
        raise ValueError(f"Transfer failed: {result.get('error', 'Unknown error')}")


def get_all_models() -> Dict[str, Any]:
    """
    Get all supported models and their capabilities.
    
    Returns:
        Dictionary with model information
    """
    api = ContextTransferAPI()
    return api.get_supported_models()


def validate_conversation(data: Union[str, List[Dict], Dict]) -> Dict[str, Any]:
    """
    Validate conversation data format.
    
    Args:
        data: Conversation data to validate
        
    Returns:
        Validation result
    """
    api = ContextTransferAPI()
    return api.validate_conversation_format(data) 