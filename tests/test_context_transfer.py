"""
Unit tests for context transfer functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pytest

from multimind.context_transfer import ContextTransferManager, AdapterFactory


class TestContextTransferManager(unittest.TestCase):
    """Test cases for ContextTransferManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ContextTransferManager()
        self.sample_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
            {"role": "user", "content": "What's the weather like?"}
        ]
    
    def test_extract_context_basic(self):
        """Test basic context extraction."""
        extracted = self.manager.extract_context(self.sample_messages, last_n=3)
        self.assertEqual(len(extracted), 3)
        self.assertEqual(extracted[-1]["content"], "What's the weather like?")
    
    def test_extract_context_empty(self):
        """Test context extraction with empty messages."""
        extracted = self.manager.extract_context([], last_n=5)
        self.assertEqual(extracted, [])
    
    def test_extract_context_last_n_exceeds_messages(self):
        """Test when last_n exceeds number of messages."""
        extracted = self.manager.extract_context(self.sample_messages, last_n=10)
        self.assertEqual(len(extracted), 5)
    
    def test_summarize_context(self):
        """Test context summarization."""
        summary = self.manager.summarize_context(self.sample_messages)
        self.assertIn("User: Hello", summary)
        self.assertIn("Assistant: Hi there!", summary)
        self.assertIn("User: What's the weather like?", summary)
    
    def test_summarize_context_empty(self):
        """Test summarization with empty messages."""
        summary = self.manager.summarize_context([])
        self.assertEqual(summary, "No conversation context available.")
    
    def test_load_conversation_from_file(self):
        """Test loading conversation from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_messages, f)
            temp_file = f.name
        
        try:
            loaded_messages = self.manager.load_conversation_from_file(temp_file)
            self.assertEqual(loaded_messages, self.sample_messages)
        finally:
            Path(temp_file).unlink()
    
    def test_load_conversation_with_messages_key(self):
        """Test loading conversation with 'messages' key in JSON."""
        data_with_messages = {"messages": self.sample_messages}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_with_messages, f)
            temp_file = f.name
        
        try:
            loaded_messages = self.manager.load_conversation_from_file(temp_file)
            self.assertEqual(loaded_messages, self.sample_messages)
        finally:
            Path(temp_file).unlink()
    
    def test_load_conversation_file_not_found(self):
        """Test loading conversation from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.manager.load_conversation_from_file("nonexistent.json")
    
    def test_save_formatted_prompt(self):
        """Test saving formatted prompt to file."""
        content = "Test prompt content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            self.manager.save_formatted_prompt(content, temp_file)
            
            with open(temp_file, 'r') as f:
                saved_content = f.read()
            
            self.assertEqual(saved_content, content)
        finally:
            Path(temp_file).unlink()
    
    @pytest.mark.skip(reason="_format_for_claude does not exist in ContextTransferManager")
    def test_format_for_claude(self):
        """Test formatting for Claude model."""
        summary = "User: Hello\nAssistant: Hi there!"
        formatted = self.manager._format_for_claude(summary, "deepseek")
        
        self.assertIn("You are Claude", formatted)
        self.assertIn("deepseek", formatted)
        self.assertIn(summary, formatted)
    
    @pytest.mark.skip(reason="_format_for_deepseek does not exist in ContextTransferManager")
    def test_format_for_deepseek(self):
        """Test formatting for DeepSeek model."""
        summary = "User: Hello\nAssistant: Hi there!"
        formatted = self.manager._format_for_deepseek(summary, "chatgpt")
        
        self.assertIn("You are DeepSeek", formatted)
        self.assertIn("chatgpt", formatted)
        self.assertIn(summary, formatted)
    
    def test_format_generic(self):
        """Test generic formatting for unknown model."""
        summary = "User: Hello\nAssistant: Hi there!"
        formatted = self.manager._format_generic(summary, "chatgpt", "UnknownModel")
        
        self.assertIn("You are UnknownModel", formatted)
        self.assertIn("chatgpt", formatted)
        self.assertIn(summary, formatted)


class TestAdapterFactory(unittest.TestCase):
    """Test cases for AdapterFactory."""
    
    def test_get_adapter_deepseek(self):
        """Test getting DeepSeek adapter."""
        adapter = AdapterFactory.get_adapter("deepseek")
        self.assertEqual(adapter.model_name, "DeepSeek")
    
    def test_get_adapter_claude(self):
        """Test getting Claude adapter."""
        adapter = AdapterFactory.get_adapter("claude")
        self.assertEqual(adapter.model_name, "Claude")
    
    def test_get_adapter_case_insensitive(self):
        """Test that adapter names are case-insensitive."""
        adapter1 = AdapterFactory.get_adapter("DEEPSEEK")
        adapter2 = AdapterFactory.get_adapter("deepseek")
        self.assertEqual(adapter1.model_name, adapter2.model_name)
    
    def test_get_adapter_unsupported(self):
        """Test getting unsupported adapter."""
        with self.assertRaises(ValueError):
            AdapterFactory.get_adapter("unsupported_model")
    
    def test_get_supported_models(self):
        """Test getting list of supported models."""
        models = AdapterFactory.get_supported_models()
        self.assertIn("deepseek", models)
        self.assertIn("claude", models)
        self.assertIn("chatgpt", models)


class TestModelAdapters(unittest.TestCase):
    """Test cases for model adapters."""
    
    def test_deepseek_adapter(self):
        """Test DeepSeek adapter formatting."""
        adapter = AdapterFactory.get_adapter("deepseek")
        summary = "User: Hello\nAssistant: Hi there!"
        formatted = adapter.format_context(summary, "chatgpt")
        
        self.assertIn("You are DeepSeek", formatted)
        self.assertIn("chatgpt", formatted)
        self.assertIn(summary, formatted)
    
    def test_claude_adapter(self):
        """Test Claude adapter formatting."""
        adapter = AdapterFactory.get_adapter("claude")
        summary = "User: Hello\nAssistant: Hi there!"
        formatted = adapter.format_context(summary, "deepseek")
        
        self.assertIn("You are Claude", formatted)
        self.assertIn("deepseek", formatted)
        self.assertIn(summary, formatted)
    
    def test_gemini_adapter(self):
        """Test Gemini adapter formatting."""
        adapter = AdapterFactory.get_adapter("gemini")
        summary = "User: Hello\nAssistant: Hi there!"
        formatted = adapter.format_context(summary, "chatgpt")
        
        self.assertIn("You are Gemini", formatted)
        self.assertIn("chatgpt", formatted)
        self.assertIn(summary, formatted)


class TestIntegration(unittest.TestCase):
    """Integration tests for context transfer."""
    
    def test_full_transfer_workflow(self):
        """Test complete context transfer workflow."""
        manager = ContextTransferManager()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_f:
            json.dump([
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ], input_f)
            input_file = input_f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_f:
            output_file = output_f.name
        
        try:
            # Perform transfer
            result = manager.transfer_context(
                from_model="chatgpt",
                to_model="deepseek",
                input_file=input_file,
                output_file=output_file,
                last_n=3
            )
            
            # Verify output file was created
            self.assertTrue(Path(output_file).exists())
            
            # Verify content
            with open(output_file, 'r') as f:
                content = f.read()
            
            self.assertIn("You are DeepSeek", content)
            self.assertIn("chatgpt", content)
            self.assertIn("Hello", content)
            self.assertIn("Hi there!", content)
            
        finally:
            # Cleanup
            Path(input_file).unlink()
            Path(output_file).unlink()


if __name__ == "__main__":
    unittest.main() 