# MultiMind Playground

A Streamlit-based web interface for testing and exploring MultiMind's memory capabilities. This playground provides interactive interfaces for various domain-specific examples.

## Features

- Interactive web interface for all domain-specific examples
- Easy navigation between different examples
- Real-time memory management and analysis
- Persistent storage for each example
- Comprehensive querying and analysis capabilities
- Interactive visualizations:
  - Knowledge graph visualization
  - Event timeline visualization
  - Analysis steps visualization
- Data management:
  - Export data in multiple formats (JSON, CSV, Excel)
  - Import data from various sources
  - Real-time data validation
- Memory statistics and monitoring
- Chat history tracking
- Customizable UI with modern design

## Examples

1. **Scientific Research Assistant**
   - Research knowledge management
   - Experiment tracking
   - Pattern analysis
   - Interactive experiment timeline
   - Research knowledge graph

2. **Customer Service Assistant**
   - Customer interaction tracking
   - Product/service knowledge
   - Pattern analysis
   - Customer journey visualization
   - Service knowledge graph

3. **Project Management Assistant**
   - Task tracking
   - Project knowledge management
   - Dependency analysis
   - Project timeline visualization
   - Task dependency graph

4. **Content Creation Assistant**
   - Content knowledge management
   - Content planning
   - Dependency tracking
   - Content structure visualization
   - Planning step timeline

5. **Software Development Assistant**
   - Code knowledge management
   - Development tracking
   - Pattern analysis
   - Code structure visualization
   - Development timeline

6. **Data Analysis Assistant**
   - Analysis knowledge management
   - Analysis step tracking
   - Dependency tracking
   - Analysis workflow visualization
   - Step dependency graph

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Select an example from the sidebar
2. Use the tabs to:
   - Add knowledge
   - Add events/steps
   - Query and analyze data
3. Use the sidebar to:
   - Choose visualization type
   - Export/import data
   - View statistics
   - Track chat history
4. Each example maintains its own memory and storage
5. Switch between examples to explore different capabilities

## Memory Types

Each example combines different memory types:

- `KnowledgeGraphMemory`: For structured knowledge management
  - Node and edge management
  - Relationship inference
  - Graph visualization
- `EventSourcedMemory`: For event tracking and pattern analysis
  - Event timeline
  - Pattern detection
  - Causality analysis
- `CognitiveScratchpadMemory`: For step-by-step reasoning and planning
  - Step tracking
  - Dependency management
  - Progress visualization

## Visualization Features

- **Knowledge Graph Visualization**
  - Interactive node exploration
  - Relationship highlighting
  - Zoom and pan capabilities
  - Node clustering

- **Event Timeline Visualization**
  - Chronological event display
  - Event type filtering
  - Timeline zooming
  - Event details on hover

- **Analysis Steps Visualization**
  - Step progress tracking
  - Dependency highlighting
  - Status indicators
  - Step details on hover

## Data Management

- **Export Options**
  - JSON format for full data export
  - CSV format for tabular data
  - Excel format for spreadsheet integration

- **Import Options**
  - File upload support
  - Data validation
  - Format conversion
  - Error handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 