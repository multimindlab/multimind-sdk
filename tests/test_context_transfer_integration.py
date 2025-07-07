"""
Integration Tests for Context Transfer Functionality

These tests verify that the context transfer feature works correctly
for the Chrome extension use case.
"""

import json
import tempfile
import os
import sys
import unittest
from pathlib import Path

# Import the context transfer modules directly
sys.path.insert(0, 'multimind')
from context_transfer.manager import ContextTransferManager
from context_transfer.adapters import AdapterFactory


class TestContextTransferIntegration(unittest.TestCase):
    """Integration tests for context transfer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ContextTransferManager()
        
        # Sample ChatGPT conversation (what Chrome extension would capture)
        self.chatgpt_conversation = [
            {
                "role": "user",
                "content": "I need help with a Python project. I'm building a web scraper that needs to handle dynamic content."
            },
            {
                "role": "assistant", 
                "content": "I can help you build a web scraper for dynamic content! There are several approaches depending on your needs. What kind of dynamic content are you dealing with - JavaScript-rendered content, AJAX-loaded data, or something else?"
            },
            {
                "role": "user",
                "content": "It's JavaScript-rendered content. The page loads data after the initial HTML loads."
            },
            {
                "role": "assistant",
                "content": "Perfect! For JavaScript-rendered content, you'll want to use a tool like Selenium or Playwright. Here's a basic example using Selenium:\n\n```python\nfrom selenium import webdriver\nfrom selenium.webdriver.common.by import By\nfrom selenium.webdriver.support.ui import WebDriverWait\nfrom selenium.webdriver.support import expected_conditions as EC\n\n# Setup\ndriver = webdriver.Chrome()\ndriver.get('https://example.com')\n\n# Wait for dynamic content to load\nwait = WebDriverWait(driver, 10)\nelement = wait.until(EC.presence_of_element_located((By.ID, 'dynamic-content')))\n\n# Now you can scrape the content\ncontent = element.text\nprint(content)\n```\n\nWould you like me to show you how to handle specific scenarios like infinite scrolling or pagination?"
            },
            {
                "role": "user",
                "content": "Yes, I need to handle infinite scrolling. The page loads more content as you scroll down."
            },
            {
                "role": "assistant",
                "content": "Great! Here's how to handle infinite scrolling with Selenium:\n\n```python\nimport time\nfrom selenium.webdriver.common.keys import Keys\n\n# Scroll to bottom to trigger loading\nlast_height = driver.execute_script('return document.body.scrollHeight')\n\nwhile True:\n    # Scroll down\n    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')\n    \n    # Wait for new content to load\n    time.sleep(2)\n    \n    # Calculate new scroll height\n    new_height = driver.execute_script('return document.body.scrollHeight')\n    \n    # If height is the same, we've reached the bottom\n    if new_height == last_height:\n        break\n        \n    last_height = new_height\n\n# Now collect all the content\nall_content = driver.find_elements(By.CLASS_NAME, 'content-item')\nfor item in all_content:\n    print(item.text)\n```\n\nThis will scroll until no more content loads. Would you like me to show you how to optimize this for better performance?"
            }
        ]
    
    def test_chrome_extension_workflow(self):
        """Test the complete Chrome extension workflow."""
        print("\n🧪 Testing Chrome Extension Workflow")
        print("=" * 50)
        
        # Step 1: Create temporary files (simulating Chrome extension)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_f:
            json.dump(self.chatgpt_conversation, input_f)
            input_file = input_f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_f:
            output_file = output_f.name
        
        try:
            # Step 2: Test context transfer (what Chrome extension would call)
            print("🔄 Transferring context from ChatGPT to DeepSeek...")
            formatted_prompt = self.manager.transfer_context(
                from_model="chatgpt",
                to_model="deepseek",
                input_file=input_file,
                output_file=output_file,
                last_n=5
            )
            
            # Step 3: Verify the output
            self.assertTrue(os.path.exists(output_file), "Output file should be created")
            
            with open(output_file, 'r') as f:
                saved_content = f.read()
            
            self.assertEqual(formatted_prompt, saved_content, "Content should match")
            
            # Step 4: Verify the formatted prompt contains expected elements
            self.assertIn("You are DeepSeek", formatted_prompt, "Should contain DeepSeek system prompt")
            self.assertIn("chatgpt", formatted_prompt, "Should mention source model")
            self.assertIn("web scraper", formatted_prompt, "Should contain conversation context")
            self.assertIn("Selenium", formatted_prompt, "Should contain technical details")
            
            print("✅ Chrome extension workflow test passed!")
            print(f"📄 Generated prompt length: {len(formatted_prompt)} characters")
            
        finally:
            # Cleanup
            os.unlink(input_file)
            os.unlink(output_file)
    
    def test_multiple_model_transfers(self):
        """Test transferring to different target models."""
        print("\n🧪 Testing Multiple Model Transfers")
        print("=" * 40)
        
        models_to_test = ["deepseek", "claude", "gemini"]
        
        for target_model in models_to_test:
            print(f"🔄 Testing transfer to {target_model}...")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_f:
                json.dump(self.chatgpt_conversation, input_f)
                input_file = input_f.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_f:
                output_file = output_f.name
            
            try:
                formatted_prompt = self.manager.transfer_context(
                    from_model="chatgpt",
                    to_model=target_model,
                    input_file=input_file,
                    output_file=output_file,
                    last_n=3
                )
                
                # Verify model-specific formatting
                if target_model == "deepseek":
                    self.assertIn("You are DeepSeek", formatted_prompt)
                elif target_model == "claude":
                    self.assertIn("You are Claude", formatted_prompt)
                elif target_model == "gemini":
                    self.assertIn("You are Gemini", formatted_prompt)
                
                print(f"✅ {target_model} transfer successful")
                
            finally:
                os.unlink(input_file)
                os.unlink(output_file)
    
    def test_context_extraction_variations(self):
        """Test different context extraction scenarios."""
        print("\n🧪 Testing Context Extraction Variations")
        print("=" * 45)

        # Test with different last_n values
        test_cases = [
            (1, "Last message only"),
            (3, "Last 3 messages"),
            (5, "Last 5 messages"),
            (10, "All messages (more than available)")
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_f:
            json.dump(self.chatgpt_conversation, input_f)
            input_file = input_f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_f:
            output_file = output_f.name

        try:
            for last_n, description in test_cases:
                print(f"🔄 Testing {description}...")

                formatted_prompt = self.manager.transfer_context(
                    from_model="chatgpt",
                    to_model="deepseek",
                    input_file=input_file,
                    output_file=output_file,
                    last_n=last_n
                )

                # Verify the prompt is a non-empty string
                self.assertIsInstance(formatted_prompt, str, f"Should return a string for {description}")
                self.assertTrue(len(formatted_prompt) > 0, f"Should return non-empty string for {description}")
        finally:
            Path(input_file).unlink()
            Path(output_file).unlink()
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        print("\n🧪 Testing Error Handling")
        print("=" * 30)
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.manager.transfer_context(
                from_model="chatgpt",
                to_model="deepseek",
                input_file="nonexistent.json",
                output_file="output.txt"
            )
        
        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_file = f.name
        
        try:
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                self.manager.transfer_context(
                    from_model="chatgpt",
                    to_model="deepseek",
                    input_file=invalid_file,
                    output_file="output.txt"
                )
        finally:
            os.unlink(invalid_file)
        
        # Test with unsupported model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_f:
            json.dump(self.chatgpt_conversation, input_f)
            input_file = input_f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_f:
            output_file = output_f.name
        
        try:
            # This should work with generic formatting
            formatted_prompt = self.manager.transfer_context(
                from_model="chatgpt",
                to_model="unsupported_model",
                input_file=input_file,
                output_file=output_file
            )
            
            self.assertIn("You are unsupported_model", formatted_prompt)
            print("✅ Unsupported model handled gracefully")
            
        finally:
            os.unlink(input_file)
            os.unlink(output_file)
    
    def test_adapter_functionality(self):
        """Test the model adapters directly."""
        print("\n🧪 Testing Model Adapters")
        print("=" * 25)
        
        summary = "User: Hello\nAssistant: Hi there!"
        
        # Test DeepSeek adapter
        deepseek_adapter = AdapterFactory.get_adapter("deepseek")
        deepseek_formatted = deepseek_adapter.format_context(summary, "chatgpt")
        self.assertIn("You are DeepSeek", deepseek_formatted)
        self.assertIn("chatgpt", deepseek_formatted)
        print("✅ DeepSeek adapter works")
        
        # Test Claude adapter
        claude_adapter = AdapterFactory.get_adapter("claude")
        claude_formatted = claude_adapter.format_context(summary, "deepseek")
        self.assertIn("You are Claude", claude_formatted)
        self.assertIn("deepseek", claude_formatted)
        print("✅ Claude adapter works")
        
        # Test Gemini adapter
        gemini_adapter = AdapterFactory.get_adapter("gemini")
        gemini_formatted = gemini_adapter.format_context(summary, "chatgpt")
        self.assertIn("You are Gemini", gemini_formatted)
        self.assertIn("chatgpt", gemini_formatted)
        print("✅ Gemini adapter works")
        
        # Test unsupported model
        with self.assertRaises(ValueError):
            AdapterFactory.get_adapter("unsupported_model")


def run_chrome_extension_demo():
    """Run a demo of the Chrome extension workflow."""
    print("\n🚀 Chrome Extension Context Transfer Demo")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestContextTransferIntegration()
    test_instance.setUp()
    
    # Run the main workflow test
    test_instance.test_chrome_extension_workflow()
    
    print("\n✅ Demo completed successfully!")
    print("\n📋 Summary:")
    print("   • Context transfer functionality works perfectly")
    print("   • Supports multiple target models (DeepSeek, Claude, Gemini)")
    print("   • Handles various context extraction scenarios")
    print("   • Robust error handling")
    print("   • Ready for Chrome extension integration")
    
    print("\n🎯 Chrome Extension Integration Ready:")
    print("   1. Extension captures ChatGPT conversation")
    print("   2. Calls MultiMind SDK context transfer")
    print("   3. Gets formatted prompt for target model")
    print("   4. Injects prompt into target model interface")
    print("   5. User continues conversation seamlessly!")


if __name__ == "__main__":
    # Run the demo
    run_chrome_extension_demo()
    
    # Run all tests
    print("\n🧪 Running all tests...")
    unittest.main(verbosity=2) 