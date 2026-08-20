"""
Advanced Chrome Extension Example for Context Transfer

Demonstrates comprehensive context transfer capabilities for building
a world-class Chrome extension that supports the entire LLM ecosystem.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from multimind.context_transfer import ContextTransferAPI, get_all_models, quick_transfer
from multimind.context_transfer.adapters import AdapterFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromeExtensionDemo:
    """
    Advanced demo class for Chrome extension context transfer functionality.
    """

    def __init__(self):
        self.api = ContextTransferAPI()
        self.supported_models = self.api.get_supported_models()

    def demo_basic_transfer(self):
        """Demonstrate basic context transfer."""
        print("🚀 Basic Context Transfer Demo")
        print("=" * 50)

        # Sample conversation data
        conversation = [
            {"role": "user", "content": "I need help with Python programming"},
            {"role": "assistant", "content": "I'd be happy to help with Python! What specific topic are you working on?"},
            {"role": "user", "content": "I'm trying to understand decorators"},
            {"role": "assistant", "content": "Decorators are a powerful Python feature. They allow you to modify or enhance functions..."}
        ]

        try:
            # Transfer from ChatGPT to DeepSeek
            result = self.api.transfer_context_api(
                source_model="chatgpt",
                target_model="deepseek",
                conversation_data=conversation
            )

            if result["success"]:
                print("✅ Transfer successful!")
                print(f"📝 Formatted prompt length: {len(result['formatted_prompt'])} characters")
                print(f"📊 Messages processed: {result['metadata']['messages_processed']}")
                print(f"🎯 Summary type: {result['metadata']['summary_type']}")

                # Show preview
                preview = result['formatted_prompt'][:200] + "..." if len(result['formatted_prompt']) > 200 else result['formatted_prompt']
                print(f"\n📋 Preview:\n{preview}")
            else:
                print(f"❌ Transfer failed: {result['error']}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def demo_advanced_transfer(self):
        """Demonstrate advanced context transfer with custom options."""
        print("\n🎯 Advanced Context Transfer Demo")
        print("=" * 50)

        # More complex conversation
        conversation = [
            {"role": "system", "content": "You are a helpful coding assistant specializing in Python and JavaScript."},
            {"role": "user", "content": "I'm building a web application with React and need to implement authentication"},
            {"role": "assistant", "content": "Great! For React authentication, I recommend using JWT tokens. Here's a basic setup..."},
            {"role": "user", "content": "Can you show me how to handle token refresh?"},
            {"role": "assistant", "content": "Token refresh is crucial for security. Here's how to implement it..."},
            {"role": "user", "content": "What about error handling for expired tokens?"},
            {"role": "assistant", "content": "Error handling for expired tokens involves intercepting 401 responses..."}
        ]

        # Advanced options
        options = {
            "last_n": 8,
            "summary_type": "detailed",
            "smart_extraction": True,
            "include_code_context": True,
            "include_reasoning": True,
            "include_step_by_step": True,
            "output_format": "json"
        }

        try:
            # Transfer to multiple models
            target_models = ["claude", "gemini", "mistral"]

            for target_model in target_models:
                print(f"\n🔄 Transferring to {target_model.upper()}...")

                result = self.api.transfer_context_api(
                    source_model="chatgpt",
                    target_model=target_model,
                    conversation_data=conversation,
                    options=options
                )

                if result["success"]:
                    print(f"✅ {target_model.upper()} transfer successful!")
                    print(f"   📏 Prompt length: {result['metadata']['prompt_length']:,} chars")
                    print(f"   🧠 Smart extraction: {'✅' if result['metadata']['smart_extraction'] else '❌'}")

                    # Show model capabilities
                    if "model_capabilities" in result["metadata"]:
                        target_caps = result["metadata"]["model_capabilities"]["target"]
                        print(f"   💾 Context limit: {target_caps.get('max_context_length', 'Unknown'):,} tokens")
                        print(f"   🔧 Code support: {'✅' if target_caps.get('supports_code') else '❌'}")
                else:
                    print(f"❌ {target_model.upper()} transfer failed: {result['error']}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def demo_batch_transfer(self):
        """Demonstrate batch transfer capabilities."""
        print("\n📦 Batch Transfer Demo")
        print("=" * 50)

        # Multiple conversations
        conversations = [
            {
                "source_model": "chatgpt",
                "target_model": "deepseek",
                "conversation_data": [
                    {"role": "user", "content": "Explain machine learning basics"},
                    {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn..."}
                ],
                "options": {"summary_type": "concise"}
            },
            {
                "source_model": "claude",
                "target_model": "gemini",
                "conversation_data": [
                    {"role": "user", "content": "Help me with data analysis"},
                    {"role": "assistant", "content": "Data analysis involves collecting, cleaning, and interpreting data..."}
                ],
                "options": {"summary_type": "detailed", "include_code_context": True}
            },
            {
                "source_model": "gemini",
                "target_model": "mistral",
                "conversation_data": [
                    {"role": "user", "content": "What are the best practices for API design?"},
                    {"role": "assistant", "content": "API design best practices include RESTful principles, proper error handling..."}
                ],
                "options": {"include_reasoning": True, "include_examples": True}
            }
        ]

        try:
            result = self.api.batch_transfer(conversations)

            print("📊 Batch Transfer Results:")
            print(f"   Total transfers: {result['total_transfers']}")
            print(f"   Successful: {result['successful_transfers']}")
            print(f"   Failed: {result['failed_transfers']}")
            print(f"   Success rate: {(result['successful_transfers'] / result['total_transfers']) * 100:.1f}%")

            for i, transfer_result in enumerate(result['results']):
                if transfer_result['success']:
                    print(f"   ✅ Transfer {i+1}: {transfer_result['metadata']['source_model']} → {transfer_result['metadata']['target_model']}")
                else:
                    print(f"   ❌ Transfer {i+1}: {transfer_result['error']}")

        except Exception as e:
            print(f"❌ Batch transfer error: {e}")

    def demo_model_capabilities(self):
        """Demonstrate model capabilities and information."""
        print("\n🤖 Model Capabilities Demo")
        print("=" * 50)

        try:
            # Get all model capabilities
            models_info = self.api.get_supported_models()

            if models_info["success"]:
                print(f"📋 Total supported models: {models_info['total_models']}")
                print(f"📄 Supported formats: {', '.join(models_info['supported_formats'])}")

                # Show top models by context length
                models = models_info["models"]
                sorted_models = sorted(
                    models.items(),
                    key=lambda x: x[1].get('max_context_length', 0),
                    reverse=True
                )

                print("\n🏆 Top 5 Models by Context Length:")
                for i, (model_name, caps) in enumerate(sorted_models[:5], 1):
                    context_length = caps.get('max_context_length', 'Unknown')
                    print(f"   {i}. {model_name.upper()}: {context_length:,} tokens")

                # Show models with special capabilities
                print("\n🌟 Models with Special Capabilities:")
                for model_name, caps in models.items():
                    special_features = []
                    if caps.get('supports_images'):
                        special_features.append("Images")
                    if caps.get('supports_tools'):
                        special_features.append("Tools")
                    if caps.get('supports_code'):
                        special_features.append("Code")

                    if special_features:
                        print(f"   📌 {model_name.upper()}: {', '.join(special_features)}")

            else:
                print(f"❌ Failed to get model info: {models_info['error']}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def demo_validation(self):
        """Demonstrate conversation validation."""
        print("\n🔍 Conversation Validation Demo")
        print("=" * 50)

        # Test different conversation formats
        test_cases = [
            {
                "name": "Valid conversation",
                "data": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            },
            {
                "name": "Empty conversation",
                "data": []
            },
            {
                "name": "Conversation with system message",
                "data": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Help me"},
                    {"role": "assistant", "content": "I'm here to help!"}
                ]
            },
            {
                "name": "Conversation with unknown role",
                "data": [
                    {"role": "user", "content": "Hello"},
                    {"role": "unknown", "content": "This is unknown"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            }
        ]

        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")

            try:
                result = self.api.validate_conversation_format(test_case['data'])

                if result['success'] and result['valid']:
                    analysis = result['analysis']
                    print("   ✅ Valid conversation")
                    print(f"   📊 Total messages: {analysis['total_messages']}")
                    print(f"   👤 User messages: {analysis['user_messages']}")
                    print(f"   🤖 Assistant messages: {analysis['assistant_messages']}")
                    print(f"   ⚙️ System messages: {analysis['system_messages']}")
                    print(f"   📏 Avg length: {analysis['average_message_length']:.0f} chars")

                    if result['recommendations']:
                        print("   💡 Recommendations:")
                        for rec in result['recommendations']:
                            print(f"      • {rec}")
                else:
                    print(f"   ❌ Invalid: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

    def demo_chrome_extension_config(self):
        """Demonstrate Chrome extension configuration generation."""
        print("\n🌐 Chrome Extension Configuration Demo")
        print("=" * 50)

        try:
            config = self.api.create_chrome_extension_config()

            print("📋 Generated Configuration:")
            print(f"   API Version: {config['api_version']}")
            print(f"   Supported Models: {len(config['supported_models'])}")
            print(f"   Supported Formats: {', '.join(config['supported_formats'])}")

            print("\n🔧 Default Options:")
            for key, value in config['default_options'].items():
                print(f"   {key}: {value}")

            print("\n🌐 Endpoints:")
            for endpoint, path in config['endpoints'].items():
                print(f"   {endpoint}: {path}")

            print("\n📦 Chrome Extension Info:")
            chrome_info = config['chrome_extension']
            print(f"   Manifest Version: {chrome_info['manifest_version']}")
            print(f"   Permissions: {', '.join(chrome_info['permissions'])}")
            print(f"   Scripts: {', '.join(chrome_info['content_scripts'] + chrome_info['background_scripts'])}")

            # Save config to file
            config_file = Path("chrome_extension_config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"\n💾 Configuration saved to: {config_file}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def demo_quick_transfer(self):
        """Demonstrate quick transfer function."""
        print("\n⚡ Quick Transfer Demo")
        print("=" * 50)

        conversation = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]

        try:
            # Quick transfer using convenience function
            formatted_prompt = quick_transfer(
                source_model="chatgpt",
                target_model="claude",
                conversation_data=conversation,
                include_safety=True
            )

            print("✅ Quick transfer successful!")
            print(f"📝 Prompt length: {len(formatted_prompt)} characters")

            # Show preview
            preview = formatted_prompt[:150] + "..." if len(formatted_prompt) > 150 else formatted_prompt
            print(f"\n📋 Preview:\n{preview}")

        except Exception as e:
            print(f"❌ Quick transfer failed: {e}")

    def run_all_demos(self):
        """Run all demonstration functions."""
        print("🎉 Advanced Context Transfer Demo Suite")
        print("=" * 60)
        print("This demo showcases the comprehensive context transfer capabilities")
        print("for building a world-class Chrome extension.\n")

        demos = [
            self.demo_basic_transfer,
            self.demo_advanced_transfer,
            self.demo_batch_transfer,
            self.demo_model_capabilities,
            self.demo_validation,
            self.demo_chrome_extension_config,
            self.demo_quick_transfer
        ]

        for demo in demos:
            try:
                demo()
                print("\n" + "─" * 60 + "\n")
            except Exception as e:
                print(f"❌ Demo failed: {e}")
                print("\n" + "─" * 60 + "\n")


def main():
    """Main function to run the demo suite."""
    demo = ChromeExtensionDemo()
    demo.run_all_demos()

    print("🎊 Demo suite completed!")
    print("\n💡 Next Steps:")
    print("   1. Use the generated chrome_extension_config.json for your extension")
    print("   2. Integrate the ContextTransferAPI into your backend")
    print("   3. Use the CLI for testing and development")
    print("   4. Explore the advanced features for your specific use case")


if __name__ == "__main__":
    main()
