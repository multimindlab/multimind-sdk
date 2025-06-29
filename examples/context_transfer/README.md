# Advanced Context Transfer for Chrome Extensions

## 🌟 Overview

This module provides **world-class context transfer capabilities** for building advanced Chrome extensions that seamlessly transfer conversation context between different LLM providers across the entire AI ecosystem.

## 🚀 Key Features

### 🎯 **Comprehensive LLM Ecosystem Support**
- **15+ Supported Models**: ChatGPT, Claude, DeepSeek, Gemini, Mistral, Llama, Cohere, and more
- **Model-Specific Adapters**: Optimized formatting for each model's capabilities
- **Advanced Capabilities**: Code support, image handling, tool usage, reasoning, and more

### 🧠 **Intelligent Context Processing**
- **Smart Extraction**: Automatically identifies and preserves important context
- **Multiple Summary Types**: Concise, detailed, and structured summaries
- **Context Compression**: Optimizes for different model token limits
- **Format Detection**: Auto-detects JSON, text, and markdown formats

### 🔧 **Advanced Formatting Options**
- **Code Context**: Maintains programming language and style consistency
- **Reasoning Instructions**: Includes step-by-step explanation guidance
- **Safety & Ethics**: Incorporates safety considerations for sensitive topics
- **Creativity & Examples**: Adds creative problem-solving instructions
- **Multimodal Support**: Handles text, code, and image content
- **Web Search**: Includes real-time information capabilities

### 📦 **Batch Processing & API**
- **Batch Transfers**: Process multiple conversations simultaneously
- **RESTful API**: Full API for Chrome extension integration
- **Validation**: Comprehensive conversation format validation
- **Error Handling**: Robust error handling and recovery

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Basic Usage

#### 1. **CLI Interface**
```bash
# List all supported models
multimind context-transfer --list_models

# Basic transfer
multimind context-transfer \
  --from_model chatgpt \
  --to_model deepseek \
  --input_file conversation.json \
  --output_file deepseek_prompt.txt

# Advanced transfer with smart features
multimind context-transfer \
  --from_model claude \
  --to_model gemini \
  --input_file chat.json \
  --output_file gemini_prompt.txt \
  --last_n 10 \
  --summary_type detailed \
  --smart_extraction \
  --include_code_context \
  --include_reasoning \
  --output_format json
```

#### 2. **Python API**
```python
from multimind.context_transfer import ContextTransferAPI

# Initialize API
api = ContextTransferAPI()

# Basic transfer
result = api.transfer_context_api(
    source_model="chatgpt",
    target_model="deepseek",
    conversation_data=conversation_messages
)

# Advanced transfer with options
result = api.transfer_context_api(
    source_model="claude",
    target_model="gemini",
    conversation_data=conversation_messages,
    options={
        "last_n": 8,
        "summary_type": "detailed",
        "smart_extraction": True,
        "include_code_context": True,
        "include_reasoning": True,
        "include_multimodal": True
    }
)
```

#### 3. **Quick Transfer Function**
```python
from multimind.context_transfer import quick_transfer

# Simple one-liner transfer
formatted_prompt = quick_transfer(
    source_model="chatgpt",
    target_model="claude",
    conversation_data=conversation_messages,
    include_safety=True
)
```

## 🤖 Supported Models

### **Major LLM Providers**
| Model | Context Length | Code Support | Image Support | Tools Support |
|-------|---------------|--------------|---------------|---------------|
| **ChatGPT** | 128K tokens | ✅ | ✅ | ✅ |
| **Claude** | 200K tokens | ✅ | ❌ | ✅ |
| **DeepSeek** | 32K tokens | ✅ | ❌ | ✅ |
| **Gemini** | 1M tokens | ✅ | ✅ | ✅ |
| **Mistral** | 32K tokens | ✅ | ❌ | ✅ |
| **Llama** | 4K tokens | ✅ | ❌ | ❌ |
| **Cohere** | 2K tokens | ✅ | ❌ | ❌ |

### **Model Aliases**
- `gpt4`, `gpt-4` → OpenAI GPT-4
- `gpt3`, `gpt-3` → OpenAI GPT-3
- `claude-3`, `claude-2`, `claude-1` → Anthropic Claude

## 🎯 Advanced Features

### **Smart Context Extraction**
```python
# Automatically identifies important context
options = {
    "smart_extraction": True,  # Preserves system messages and important context
    "last_n": 5               # Number of recent turns to extract
}
```

### **Multiple Summary Types**
```python
# Concise summary (default)
options = {"summary_type": "concise"}

# Detailed summary with full context
options = {"summary_type": "detailed"}

# Structured summary with sections
options = {"summary_type": "structured"}
```

### **Advanced Formatting Options**
```python
options = {
    "include_code_context": True,    # Maintains programming style
    "include_reasoning": True,       # Adds reasoning instructions
    "include_safety": True,          # Includes safety considerations
    "include_creativity": True,      # Adds creative problem-solving
    "include_examples": True,        # Requests concrete examples
    "include_step_by_step": True,    # Adds step-by-step guidance
    "include_multimodal": True,      # Handles text, code, images
    "include_web_search": True       # Includes web search capabilities
}
```

### **Batch Processing**
```python
# Process multiple transfers simultaneously
transfers = [
    {
        "source_model": "chatgpt",
        "target_model": "deepseek",
        "conversation_data": conversation1,
        "options": {"summary_type": "concise"}
    },
    {
        "source_model": "claude",
        "target_model": "gemini",
        "conversation_data": conversation2,
        "options": {"summary_type": "detailed", "include_code_context": True}
    }
]

result = api.batch_transfer(transfers)
```

## 🌐 Chrome Extension Integration

### **API Endpoints**
```javascript
// Get supported models
const models = await fetch('/api/models').then(r => r.json());

// Transfer context
const result = await fetch('/api/transfer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        source_model: 'chatgpt',
        target_model: 'deepseek',
        conversation_data: messages,
        options: {
            last_n: 5,
            summary_type: 'concise',
            smart_extraction: true
        }
    })
}).then(r => r.json());

// Validate conversation
const validation = await fetch('/api/validate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({conversation_data: messages})
}).then(r => r.json());
```

### **Generated Configuration**
```python
# Generate Chrome extension configuration
config = api.create_chrome_extension_config()
# Saves to chrome_extension_config.json
```

## 📊 Validation & Analysis

### **Conversation Validation**
```python
# Validate conversation format
result = api.validate_conversation_format(conversation_data)

if result['success'] and result['valid']:
    analysis = result['analysis']
    print(f"Total messages: {analysis['total_messages']}")
    print(f"User messages: {analysis['user_messages']}")
    print(f"Assistant messages: {analysis['assistant_messages']}")
    print(f"System messages: {analysis['system_messages']}")
    print(f"Average length: {analysis['average_message_length']:.0f} chars")
    
    # Get recommendations
    for rec in result['recommendations']:
        print(f"💡 {rec}")
```

### **Model Capabilities**
```python
# Get all model capabilities
models_info = api.get_supported_models()

# Get specific model capabilities
capabilities = api.get_model_capabilities("deepseek")
print(f"Context length: {capabilities['max_context_length']:,} tokens")
print(f"Code support: {capabilities['supports_code']}")
print(f"Image support: {capabilities['supports_images']}")
print(f"Tools support: {capabilities['supports_tools']}")
```

## 🧪 Testing & Examples

### **Run Demo Suite**
```bash
# Run comprehensive demo
python examples/context_transfer/chrome_extension_example.py
```

### **Test Different Formats**
```python
# JSON format
conversation_json = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

# Text format
conversation_text = """
User: Hello
Assistant: Hi there!
"""

# Markdown format
conversation_md = """
### User: Hello
### Assistant: Hi there!
"""
```

## 🔧 Configuration

### **Advanced Options**
```python
config = {
    "max_context_length": 32000,
    "default_summary_length": 1000,
    "include_metadata": True,
    "preserve_formatting": True,
    "smart_truncation": True,
    "context_compression": False
}
```

### **Model-Specific Settings**
```python
# Each model adapter has specific capabilities
adapter = AdapterFactory.get_adapter("deepseek")
capabilities = adapter.get_model_metadata()
print(capabilities)
```

## 🚀 Performance & Optimization

### **Smart Features**
- **Intelligent Extraction**: Automatically preserves important context
- **Context Compression**: Optimizes for different token limits
- **Batch Processing**: Efficient handling of multiple transfers
- **Caching**: Reuses model capabilities and configurations

### **Error Handling**
- **Graceful Degradation**: Falls back to generic formatting for unknown models
- **Validation**: Comprehensive input validation and error reporting
- **Recovery**: Automatic retry and fallback mechanisms

## 📈 Use Cases

### **Chrome Extension Scenarios**
1. **Multi-Platform Chat**: Transfer between ChatGPT, Claude, and Gemini
2. **Code Assistance**: Maintain context across different coding assistants
3. **Research Continuity**: Continue research across different AI platforms
4. **Learning Progression**: Track learning progress across multiple AI tutors

### **Enterprise Applications**
1. **Customer Support**: Transfer customer conversations between AI agents
2. **Development Teams**: Share context between different AI coding assistants
3. **Content Creation**: Maintain creative direction across AI writing tools
4. **Data Analysis**: Continue analysis across different AI platforms

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time Transfer**: Live context synchronization
- **Custom Adapters**: User-defined model adapters
- **Context Compression**: Advanced compression algorithms
- **Multi-language Support**: Internationalization
- **Plugin System**: Extensible architecture

### **Integration Roadmap**
- **VS Code Extension**: IDE integration
- **Slack/Discord Bots**: Chat platform integration
- **API Gateway**: Cloud-based service
- **Mobile Apps**: iOS/Android support

## 🤝 Contributing

### **Adding New Models**
```python
class CustomModelAdapter(ModelAdapter):
    def __init__(self):
        super().__init__("CustomModel")
        self.max_context_length = 16000
        self.supports_code = True
    
    def get_system_prompt(self) -> str:
        return "You are CustomModel, an AI assistant..."
    
    def format_context(self, summary: str, source_model: str, **kwargs) -> str:
        # Custom formatting logic
        return formatted_prompt
```

### **Testing**
```bash
# Run tests
python -m pytest tests/test_context_transfer.py -v

# Run specific test
python -m pytest tests/test_context_transfer.py::test_advanced_transfer -v
```

## 📄 License

This module is part of the MultiMind SDK and follows the same licensing terms.

---

**🎉 Ready to build the next generation of AI-powered Chrome extensions!** 