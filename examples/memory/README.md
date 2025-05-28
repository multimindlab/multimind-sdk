# MultiMind Memory Examples

This directory contains examples demonstrating various memory implementations in the MultiMind SDK.

## Examples Overview

### Basic Examples

1. **Chatbot with Memory** (`chatbot_with_memory.py`)
   - Demonstrates context management using `HybridMemory`
   - Shows how to maintain conversation history
   - Includes memory statistics and suggestions

2. **Active Learning System** (`active_learning_system.py`)
   - Uses `ActiveLearningMemory` for learning from user feedback
   - Demonstrates feedback tracking and improvement
   - Shows learning statistics and patterns

3. **Knowledge Management** (`knowledge_management.py`)
   - Uses `KnowledgeGraphMemory` for structured knowledge
   - Demonstrates knowledge graph operations
   - Shows relationship tracking and inference

4. **Event Tracking** (`event_tracking.py`)
   - Uses `EventSourcedMemory` for tracking sequences
   - Demonstrates event pattern analysis
   - Shows causality tracking and timeline

5. **Cognitive Scratchpad** (`cognitive_scratchpad.py`)
   - Uses `CognitiveScratchpadMemory` for step-by-step reasoning
   - Demonstrates reasoning chain management
   - Shows dependency tracking and analysis

### CLI Examples

1. **Chatbot CLI** (`chatbot_with_memory_cli.py`)
   - Interactive CLI version of the chatbot
   - Features:
     - Topic tracking
     - Context management
     - Memory statistics
     - Export/import capabilities
   - Commands:
     - `/topic` - Show current topic
     - `/context` - Show current context
     - `/stats` - Show memory statistics
     - `/export` - Export memory
     - `/import` - Import memory
     - `/clear` - Clear memory
     - `/help` - Show help
     - `/exit` - Exit program

2. **Knowledge Management CLI** (`knowledge_management_cli.py`)
   - Interactive CLI for knowledge graph management
   - Features:
     - Knowledge addition and querying
     - Domain tracking
     - Inference capabilities
     - Validation and analysis
   - Commands:
     - `/add` - Add new knowledge
     - `/query` - Query knowledge
     - `/domain` - Show current domain
     - `/infer` - Run inference
     - `/validate` - Validate knowledge
     - `/export` - Export knowledge
     - `/import` - Import knowledge
     - `/clear` - Clear knowledge
     - `/help` - Show help
     - `/exit` - Exit program

3. **Event Tracking CLI** (`event_tracking_cli.py`)
   - Interactive CLI for event tracking
   - Features:
     - Event addition and querying
     - Session management
     - Pattern analysis
     - Causality tracking
   - Commands:
     - `/add` - Add new event
     - `/query` - Query events
     - `/session` - Show current session
     - `/pattern` - Analyze patterns
     - `/cause` - Analyze causality
     - `/timeline` - Show timeline
     - `/export` - Export events
     - `/import` - Import events
     - `/clear` - Clear events
     - `/help` - Show help
     - `/exit` - Exit program

4. **Cognitive Scratchpad CLI** (`cognitive_scratchpad_cli.py`)
   - Interactive CLI for reasoning management
   - Features:
     - Chain management
     - Step tracking
     - Dependency analysis
     - Reasoning validation
   - Commands:
     - `/start` - Start new chain
     - `/step` - Add reasoning step
     - `/chain` - Show current chain
     - `/analyze` - Analyze reasoning
     - `/deps` - Show dependencies
     - `/export` - Export reasoning
     - `/import` - Import reasoning
     - `/clear` - Clear reasoning
     - `/help` - Show help
     - `/exit` - Exit program

### Domain-Specific Examples

1. **Healthcare Assistant** (`healthcare_assistant.py`)
   - Combines `KnowledgeGraphMemory` and `CognitiveScratchpadMemory`
   - Features:
     - Medical knowledge management
     - Diagnosis reasoning
     - Symptom tracking
     - Treatment recommendations
   - Use cases:
     - Medical knowledge base
     - Diagnosis assistance
     - Treatment planning
     - Medical education

2. **Financial Advisor** (`financial_advisor.py`)
   - Combines `EventSourcedMemory` and `KnowledgeGraphMemory`
   - Features:
     - Transaction tracking
     - Financial knowledge management
     - Pattern analysis
     - Investment insights
   - Use cases:
     - Portfolio management
     - Transaction analysis
     - Investment advice
     - Financial education

3. **Educational Tutor** (`educational_tutor.py`)
   - Combines `ActiveLearningMemory` and `KnowledgeGraphMemory`
   - Features:
     - Student progress tracking
     - Educational content management
     - Learning pattern analysis
     - Personalized recommendations
   - Use cases:
     - Student assessment
     - Content organization
     - Progress monitoring
     - Adaptive learning

4. **Legal Assistant** (`legal_assistant.py`)
   - Combines `KnowledgeGraphMemory` and `CognitiveScratchpadMemory`
   - Features:
     - Legal knowledge management
     - Case analysis
     - Precedent tracking
     - Legal reasoning
   - Use cases:
     - Legal research
     - Case analysis
     - Document review
     - Legal education

### Domain-Specific CLI Examples

1. **Scientific Research CLI** (`scientific_research_cli.py`)
   - Combines `KnowledgeGraphMemory` and `EventSourcedMemory`
   - Features:
     - Research knowledge management
     - Experiment tracking
     - Pattern analysis
     - Data export/import
   - Commands:
     - `/add_knowledge` - Add research knowledge
     - `/add_experiment` - Add experiment event
     - `/query` - Query research knowledge
     - `/analyze` - Analyze experiment data
     - `/stats` - Show statistics
     - `/export` - Export data
     - `/import` - Import data
     - `/clear` - Clear data
     - `/help` - Show help
     - `/exit` - Exit program

2. **Customer Service CLI** (`customer_service_cli.py`)
   - Combines `EventSourcedMemory` and `KnowledgeGraphMemory`
   - Features:
     - Customer interaction tracking
     - Product/service knowledge
     - Pattern analysis
     - Data export/import
   - Commands:
     - `/add_knowledge` - Add product/service knowledge
     - `/add_interaction` - Add customer interaction
     - `/query` - Query product/service knowledge
     - `/analyze` - Analyze customer interactions
     - `/stats` - Show statistics
     - `/export` - Export data
     - `/import` - Import data
     - `/clear` - Clear data
     - `/help` - Show help
     - `/exit` - Exit program

3. **Project Management CLI** (`project_management_cli.py`)
   - Combines `EventSourcedMemory` and `KnowledgeGraphMemory`
   - Features:
     - Task tracking
     - Project knowledge management
     - Dependency analysis
     - Data export/import
   - Commands:
     - `/add_knowledge` - Add project knowledge
     - `/add_task` - Add task event
     - `/query` - Query project knowledge
     - `/analyze` - Analyze project data
     - `/stats` - Show statistics
     - `/export` - Export data
     - `/import` - Import data
     - `/clear` - Clear data
     - `/help` - Show help
     - `/exit` - Exit program

4. **Content Creation CLI** (`content_creation_cli.py`)
   - Combines `KnowledgeGraphMemory` and `CognitiveScratchpadMemory`
   - Features:
     - Content knowledge management
     - Content planning
     - Dependency tracking
     - Data export/import
   - Commands:
     - `/add_knowledge` - Add content knowledge
     - `/start_plan` - Start new content plan
     - `/add_step` - Add planning step
     - `/query` - Query content knowledge
     - `/analyze` - Analyze content plan
     - `/stats` - Show statistics
     - `/export` - Export data
     - `/import` - Import data
     - `/clear` - Clear data
     - `/help` - Show help
     - `/exit` - Exit program

5. **Software Development CLI** (`software_development_cli.py`)
   - Combines `KnowledgeGraphMemory` and `EventSourcedMemory`
   - Features:
     - Code knowledge management
     - Development tracking
     - Pattern analysis
     - Data export/import
   - Commands:
     - `/add_knowledge` - Add code knowledge
     - `/add_event` - Add development event
     - `/query` - Query code knowledge
     - `/analyze` - Analyze development data
     - `/stats` - Show statistics
     - `/export` - Export data
     - `/import` - Import data
     - `/clear` - Clear data
     - `/help` - Show help
     - `/exit` - Exit program

6. **Data Analysis CLI** (`data_analysis_cli.py`)
   - Combines `KnowledgeGraphMemory` and `CognitiveScratchpadMemory`
   - Features:
     - Analysis knowledge management
     - Analysis step tracking
     - Dependency tracking
     - Data export/import
   - Commands:
     - `/add_knowledge` - Add analysis knowledge
     - `/start_analysis` - Start new analysis
     - `/add_step` - Add analysis step
     - `/query` - Query analysis knowledge
     - `/analyze` - Analyze analysis chain
     - `/stats` - Show statistics
     - `/export` - Export data
     - `/import` - Import data
     - `/clear` - Clear data
     - `/help` - Show help
     - `/exit` - Exit program

## Running the Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run basic examples:
```bash
python examples/memory/chatbot_with_memory.py
python examples/memory/active_learning_system.py
python examples/memory/knowledge_management.py
python examples/memory/event_tracking.py
python examples/memory/cognitive_scratchpad.py
```

3. Run CLI examples:
```bash
python examples/memory/chatbot_with_memory_cli.py
python examples/memory/knowledge_management_cli.py
python examples/memory/event_tracking_cli.py
python examples/memory/cognitive_scratchpad_cli.py
```

4. Run domain-specific examples:
```bash
python examples/memory/healthcare_assistant.py
python examples/memory/financial_advisor.py
python examples/memory/educational_tutor.py
python examples/memory/legal_assistant.py
```

5. Run domain-specific CLI examples:
```bash
python examples/memory/scientific_research_cli.py
python examples/memory/customer_service_cli.py
python examples/memory/project_management_cli.py
python examples/memory/content_creation_cli.py
python examples/memory/software_development_cli.py
python examples/memory/data_analysis_cli.py
```

## Memory Types and Use Cases

### Basic Memory Types
- `ConversationBufferMemory`: Simple conversation history
- `TokenBufferMemory`: Token-based memory management
- `TimeWeightedMemory`: Time-based memory weighting

### Advanced Memory Types
- `DNCMemory`: Complex reasoning and decision making
- `HybridMemory`: Combination of multiple memory types
- `KnowledgeGraphMemory`: Structured knowledge management
- `VectorStoreMemory`: Semantic search and retrieval

### Specialized Memory Types
- `FederatedMemory`: Multi-user collaboration
- `ActiveLearningMemory`: Learning from feedback
- `CognitiveScratchpadMemory`: Step-by-step reasoning
- `EventSourcedMemory`: Event sequence tracking

## Best Practices

1. **Memory Selection**
   - Choose memory type based on use case
   - Consider memory limitations and requirements
   - Use hybrid approaches for complex scenarios

2. **Configuration**
   - Set appropriate size limits
   - Configure analysis intervals
   - Enable required features

3. **Integration**
   - Use appropriate system prompts
   - Handle memory errors gracefully
   - Monitor memory usage

4. **Maintenance**
   - Regular validation and cleanup
   - Backup important data
   - Monitor performance metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 