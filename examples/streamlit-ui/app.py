"""
MultiMind Playground
A Streamlit-based web interface for testing and exploring MultiMind's memory capabilities.
"""

import streamlit as st
import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import altair as alt
import numpy as np

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from multimind import MultiMind, Agent, AgentMemory, CalculatorTool
from multimind.memory import (
    KnowledgeGraphMemory,
    EventSourcedMemory,
    CognitiveScratchpadMemory,
    HybridMemory
)
from multimind.models import (
    OllamaLLM,
    OpenAIModel,
    ClaudeModel,
    MistralModel
)
from multimind.rag import RAG, Document
from multimind.embeddings.embeddings import get_embedder
from multimind import MCPParser, MCPExecutor

# Initialize session state
if 'current_example' not in st.session_state:
    st.session_state.current_example = None
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'mm' not in st.session_state:
    st.session_state.mm = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'visualization_type' not in st.session_state:
    st.session_state.visualization_type = "Knowledge Graph"
if 'export_format' not in st.session_state:
    st.session_state.export_format = "JSON"
if 'filter_options' not in st.session_state:
    st.session_state.filter_options = {}
if 'sort_options' not in st.session_state:
    st.session_state.sort_options = {}

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .visualization-container {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .data-management-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_memory(memory_type, **kwargs):
    """Initialize memory based on selected type."""
    llm = OllamaLLM(model="mistral")
    
    if memory_type == "KnowledgeGraphMemory":
        return KnowledgeGraphMemory(
            llm=llm,
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            enable_inference=True,
            enable_validation=True,
            validation_interval=3600,
            **kwargs
        )
    elif memory_type == "EventSourcedMemory":
        return EventSourcedMemory(
            llm=llm,
            max_events=10000,
            max_snapshots=1000,
            snapshot_interval=3600,
            enable_causality=True,
            enable_patterns=True,
            pattern_interval=3600,
            **kwargs
        )
    elif memory_type == "CognitiveScratchpadMemory":
        return CognitiveScratchpadMemory(
            llm=llm,
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            enable_dependencies=True,
            enable_validation=True,
            validation_interval=3600,
            **kwargs
        )
    elif memory_type == "HybridMemory":
        return HybridMemory(
            llm=llm,
            memory_types=kwargs.get('memory_types', []),
            routing_strategy="hybrid",
            enable_learning=True,
            enable_analysis=True,
            **kwargs
        )
    return None

def initialize_multimind(memory, system_prompt):
    """Initialize MultiMind with given memory and system prompt."""
    llm = OllamaLLM(model="mistral")
    return MultiMind(
        llm=llm,
        memory=memory,
        system_prompt=system_prompt
    )

def visualize_knowledge_graph(memory):
    """Create an interactive visualization of the knowledge graph."""
    nodes = []
    edges = []
    
    # Get graph data from memory
    graph_data = memory.memory_types[0].get_graph_data()
    
    # Create nodes
    for node in graph_data.get('nodes', []):
        nodes.append({
            'id': node['id'],
            'label': node['label'],
            'type': node.get('type', 'concept')
        })
    
    # Create edges
    for edge in graph_data.get('edges', []):
        edges.append({
            'source': edge['source'],
            'target': edge['target'],
            'label': edge.get('label', '')
        })
    
    # Create network graph
    fig = go.Figure(data=[
        go.Scatter(
            x=[node['id'] for node in nodes],
            y=[node['label'] for node in nodes],
            mode='markers+text',
            text=[node['label'] for node in nodes],
            textposition="top center",
            marker=dict(size=20)
        ),
        go.Scatter(
            x=[edge['source'] for edge in edges],
            y=[edge['target'] for edge in edges],
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=[edge['label'] for edge in edges]
        )
    ])
    
    fig.update_layout(
        title="Knowledge Graph Visualization",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def visualize_event_timeline(memory):
    """Create an interactive visualization of the event timeline."""
    events = memory.memory_types[1].get_event_timeline()
    
    # Create timeline data
    timeline_data = []
    for event in events:
        timeline_data.append({
            'timestamp': event['timestamp'],
            'type': event['type'],
            'description': event['description']
        })
    
    # Create timeline visualization
    fig = px.timeline(
        pd.DataFrame(timeline_data),
        x_start='timestamp',
        y='type',
        color='type',
        hover_data=['description']
    )
    
    fig.update_layout(
        title="Event Timeline",
        xaxis_title="Time",
        yaxis_title="Event Type"
    )
    
    return fig

def visualize_analysis_steps(memory, chain_id):
    """Create an interactive visualization of analysis steps."""
    steps = memory.memory_types[1].get_chain_steps(chain_id)
    
    # Create step data
    step_data = []
    for i, step in enumerate(steps):
        step_data.append({
            'step_number': i + 1,
            'description': step['description'],
            'status': step.get('status', 'completed')
        })
    
    # Create step visualization
    fig = px.bar(
        pd.DataFrame(step_data),
        x='step_number',
        y='status',
        color='status',
        hover_data=['description']
    )
    
    fig.update_layout(
        title="Analysis Steps",
        xaxis_title="Step Number",
        yaxis_title="Status"
    )
    
    return fig

def create_network_graph(memory):
    """Create an interactive network graph visualization."""
    graph_data = memory.memory_types[0].get_graph_data()
    
    # Create nodes
    nodes = [
        Node(id=node['id'],
             label=node['label'],
             size=25,
             color="#1f77b4")
        for node in graph_data.get('nodes', [])
    ]
    
    # Create edges
    edges = [
        Edge(source=edge['source'],
             target=edge['target'],
             label=edge.get('label', ''),
             type="STRAIGHT")
        for edge in graph_data.get('edges', [])
    ]
    
    # Configure the graph
    config = Config(
        width=800,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False
    )
    
    return agraph(nodes=nodes, edges=edges, config=config)

def create_sunburst_chart(memory):
    """Create a sunburst chart for hierarchical data visualization."""
    data = memory.get_hierarchical_data()
    
    fig = px.sunburst(
        data,
        path=['level1', 'level2', 'level3'],
        values='value',
        color='value',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title="Hierarchical Data Visualization",
        margin=dict(t=30, l=25, r=25, b=25)
    )
    
    return fig

def create_heatmap(memory):
    """Create a heatmap for pattern analysis."""
    data = memory.get_pattern_data()
    
    fig = px.imshow(
        data,
        color_continuous_scale='Viridis',
        aspect='auto'
    )
    
    fig.update_layout(
        title="Pattern Analysis Heatmap",
        xaxis_title="Features",
        yaxis_title="Patterns"
    )
    
    return fig

def create_3d_scatter(memory):
    """Create a 3D scatter plot for multi-dimensional data."""
    data = memory.get_3d_data()
    
    fig = px.scatter_3d(
        data,
        x='x',
        y='y',
        z='z',
        color='cluster',
        size='size',
        hover_data=['label']
    )
    
    fig.update_layout(
        title="3D Data Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    
    return fig

def create_parallel_coordinates(memory):
    """Create a parallel coordinates plot for multi-dimensional data."""
    data = memory.get_multi_dimensional_data()
    
    fig = px.parallel_coordinates(
        data,
        color='category',
        dimensions=['dim1', 'dim2', 'dim3', 'dim4'],
        color_continuous_scale=px.colors.diverging.Tealrose
    )
    
    fig.update_layout(
        title="Multi-dimensional Data Analysis"
    )
    
    return fig

def export_data_advanced(memory, format_type, filters=None, sort_by=None):
    """Enhanced export functionality with filtering and sorting."""
    data = memory.get_all_data()
    
    # Apply filters
    if filters:
        for key, value in filters.items():
            data = [item for item in data if item.get(key) == value]
    
    # Apply sorting
    if sort_by:
        data = sorted(data, key=lambda x: x.get(sort_by, ''))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Export based on format
    if format_type == "JSON":
        return json.dumps(data, indent=2)
    elif format_type == "CSV":
        return df.to_csv(index=False)
    elif format_type == "Excel":
        return df.to_excel(index=False)
    elif format_type == "Parquet":
        return df.to_parquet(index=False)
    elif format_type == "HTML":
        return df.to_html(index=False)
    return None

def import_data_advanced(memory, data, format_type, validation_rules=None):
    """Enhanced import functionality with validation."""
    try:
        if format_type == "JSON":
            data_dict = json.loads(data)
        elif format_type == "CSV":
            df = pd.read_csv(data)
            data_dict = df.to_dict('records')
        elif format_type == "Excel":
            df = pd.read_excel(data)
            data_dict = df.to_dict('records')
        elif format_type == "Parquet":
            df = pd.read_parquet(data)
            data_dict = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Apply validation rules
        if validation_rules:
            for rule in validation_rules:
                if not rule.validate(data_dict):
                    raise ValueError(f"Validation failed: {rule.message}")
        
        memory.load_data(data_dict)
        return True
    except Exception as e:
        st.error(f"Error importing data: {str(e)}")
        return False

def show_advanced_statistics(memory):
    """Display advanced memory statistics with multiple visualizations."""
    stats = memory.get_statistics()
    
    # Create multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Memory usage gauge
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats['memory_usage'],
            title={'text': "Memory Usage"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig1)
        
        # Node distribution
        fig2 = px.pie(
            values=stats['node_distribution'].values(),
            names=stats['node_distribution'].keys(),
            title="Node Distribution"
        )
        st.plotly_chart(fig2)
    
    with col2:
        # Edge distribution
        fig3 = px.bar(
            x=list(stats['edge_distribution'].keys()),
            y=list(stats['edge_distribution'].values()),
            title="Edge Distribution"
        )
        st.plotly_chart(fig3)
        
        # Memory growth
        fig4 = px.line(
            x=stats['memory_growth']['timestamps'],
            y=stats['memory_growth']['values'],
            title="Memory Growth Over Time"
        )
        st.plotly_chart(fig4)

def show_chat_history():
    """Display chat history with the assistant."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def add_to_chat_history(role, content):
    """Add a message to the chat history."""
    st.session_state.chat_history.append({
        "role": role,
        "content": content
    })

def scientific_research_interface():
    """Scientific Research Assistant interface."""
    st.header("Scientific Research Assistant")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="scientific_research_knowledge.json"
        )
        event_memory = EventSourcedMemory(
            llm=OllamaLLM(model="mistral"),
            max_events=10000,
            max_snapshots=1000,
            storage_path="scientific_research_events.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, event_memory],
            storage_path="scientific_research.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a scientific research assistant that helps with research knowledge management and experiment tracking."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Add Experiment", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Research Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Add Experiment")
        event_type = st.text_input("Event Type")
        description = st.text_area("Description")
        metadata = st.text_area("Metadata (JSON)")
        
        if st.button("Add Experiment"):
            try:
                metadata_dict = eval(metadata)
                st.session_state.memory.memory_types[1].add_event(
                    event_type=event_type,
                    description=description,
                    metadata=metadata_dict
                )
                st.success("Experiment added successfully!")
            except Exception as e:
                st.error(f"Error adding experiment: {str(e)}")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[1].get_event_patterns()
            st.write("Event Patterns:", patterns)

def customer_service_interface():
    """Customer Service Assistant interface."""
    st.header("Customer Service Assistant")
    
    if st.session_state.memory is None:
        event_memory = EventSourcedMemory(
            llm=OllamaLLM(model="mistral"),
            max_events=10000,
            max_snapshots=1000,
            storage_path="customer_service_events.json"
        )
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="customer_service_knowledge.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[event_memory, knowledge_memory],
            storage_path="customer_service.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a customer service assistant that helps with customer interaction tracking and product knowledge management."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Add Interaction", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Product Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[1].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Add Customer Interaction")
        event_type = st.text_input("Event Type")
        description = st.text_area("Description")
        metadata = st.text_area("Metadata (JSON)")
        
        if st.button("Add Interaction"):
            try:
                metadata_dict = eval(metadata)
                st.session_state.memory.memory_types[0].add_event(
                    event_type=event_type,
                    description=description,
                    metadata=metadata_dict
                )
                st.success("Interaction added successfully!")
            except Exception as e:
                st.error(f"Error adding interaction: {str(e)}")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[1].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[0].get_event_patterns()
            st.write("Interaction Patterns:", patterns)

def project_management_interface():
    """Project Management Assistant interface."""
    st.header("Project Management Assistant")
    
    if st.session_state.memory is None:
        event_memory = EventSourcedMemory(
            llm=OllamaLLM(model="mistral"),
            max_events=10000,
            max_snapshots=1000,
            storage_path="project_management_events.json"
        )
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="project_management_knowledge.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[event_memory, knowledge_memory],
            storage_path="project_management.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a project management assistant that helps with task tracking and project knowledge management."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Add Task", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Project Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[1].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Add Task")
        event_type = st.text_input("Event Type")
        description = st.text_area("Description")
        metadata = st.text_area("Metadata (JSON)")
        
        if st.button("Add Task"):
            try:
                metadata_dict = eval(metadata)
                st.session_state.memory.memory_types[0].add_event(
                    event_type=event_type,
                    description=description,
                    metadata=metadata_dict
                )
                st.success("Task added successfully!")
            except Exception as e:
                st.error(f"Error adding task: {str(e)}")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[1].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[0].get_event_patterns()
            st.write("Task Patterns:", patterns)

def content_creation_interface():
    """Content Creation Assistant interface."""
    st.header("Content Creation Assistant")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="content_creation_knowledge.json"
        )
        scratchpad_memory = CognitiveScratchpadMemory(
            llm=OllamaLLM(model="mistral"),
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            storage_path="content_creation_steps.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, scratchpad_memory],
            storage_path="content_creation.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a content creation assistant that helps with content knowledge management and planning."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Content Planning", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Content Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Content Planning")
        description = st.text_area("Plan Description")
        
        if st.button("Start Plan"):
            chain_id = st.session_state.memory.memory_types[1].start_chain(description)
            st.success(f"Plan started with ID: {chain_id}")
        
        chain_id = st.text_input("Chain ID")
        step_description = st.text_area("Step Description")
        
        if st.button("Add Step"):
            st.session_state.memory.memory_types[1].add_step(
                chain_id=chain_id,
                step_description=step_description
            )
            st.success("Step added successfully!")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            if chain_id:
                steps = st.session_state.memory.memory_types[1].get_chain_steps(chain_id)
                st.write("Plan Steps:", steps)

def software_development_interface():
    """Software Development Assistant interface."""
    st.header("Software Development Assistant")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="software_development_knowledge.json"
        )
        event_memory = EventSourcedMemory(
            llm=OllamaLLM(model="mistral"),
            max_events=10000,
            max_snapshots=1000,
            storage_path="software_development_events.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, event_memory],
            storage_path="software_development.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a software development assistant that helps with code knowledge management and development tracking."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Add Event", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Code Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Add Development Event")
        event_type = st.text_input("Event Type")
        description = st.text_area("Description")
        metadata = st.text_area("Metadata (JSON)")
        
        if st.button("Add Event"):
            try:
                metadata_dict = eval(metadata)
                st.session_state.memory.memory_types[1].add_event(
                    event_type=event_type,
                    description=description,
                    metadata=metadata_dict
                )
                st.success("Event added successfully!")
            except Exception as e:
                st.error(f"Error adding event: {str(e)}")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[1].get_event_patterns()
            st.write("Development Patterns:", patterns)

def data_analysis_interface():
    """Data Analysis Assistant interface."""
    st.header("Data Analysis Assistant")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="data_analysis_knowledge.json"
        )
        scratchpad_memory = CognitiveScratchpadMemory(
            llm=OllamaLLM(model="mistral"),
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            storage_path="data_analysis_steps.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, scratchpad_memory],
            storage_path="data_analysis.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a data analysis assistant that helps with analysis knowledge management and step tracking."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Analysis Steps", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Analysis Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Analysis Steps")
        description = st.text_area("Analysis Description")
        
        if st.button("Start Analysis"):
            chain_id = st.session_state.memory.memory_types[1].start_chain(description)
            st.success(f"Analysis started with ID: {chain_id}")
        
        chain_id = st.text_input("Chain ID")
        step_description = st.text_area("Step Description")
        
        if st.button("Add Step"):
            st.session_state.memory.memory_types[1].add_step(
                chain_id=chain_id,
                step_description=step_description
            )
            st.success("Step added successfully!")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            if chain_id:
                steps = st.session_state.memory.memory_types[1].get_chain_steps(chain_id)
                st.write("Analysis Steps:", steps)

def financial_analysis_interface():
    """Financial Analysis Assistant interface."""
    st.header("Financial Analysis Assistant")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="financial_analysis_knowledge.json"
        )
        scratchpad_memory = CognitiveScratchpadMemory(
            llm=OllamaLLM(model="mistral"),
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            storage_path="financial_analysis_steps.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, scratchpad_memory],
            storage_path="financial_analysis.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a financial analysis assistant that helps with financial knowledge management and analysis."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Analysis Steps", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Financial Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Financial Analysis Steps")
        description = st.text_area("Analysis Description")
        
        if st.button("Start Analysis"):
            chain_id = st.session_state.memory.memory_types[1].start_chain(description)
            st.success(f"Analysis started with ID: {chain_id}")
        
        chain_id = st.text_input("Chain ID")
        step_description = st.text_area("Step Description")
        
        if st.button("Add Step"):
            st.session_state.memory.memory_types[1].add_step(
                chain_id=chain_id,
                step_description=step_description
            )
            st.success("Step added successfully!")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            if chain_id:
                steps = st.session_state.memory.memory_types[1].get_chain_steps(chain_id)
                st.write("Analysis Steps:", steps)

def healthcare_assistant_interface():
    """Healthcare Assistant interface."""
    st.header("Healthcare Assistant")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="healthcare_knowledge.json"
        )
        scratchpad_memory = CognitiveScratchpadMemory(
            llm=OllamaLLM(model="mistral"),
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            storage_path="healthcare_steps.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, scratchpad_memory],
            storage_path="healthcare.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a healthcare assistant that helps with healthcare knowledge management and patient care."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Patient Care", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Healthcare Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Patient Care")
        event_type = st.text_input("Event Type")
        description = st.text_area("Description")
        metadata = st.text_area("Metadata (JSON)")
        
        if st.button("Add Patient Interaction"):
            try:
                metadata_dict = eval(metadata)
                st.session_state.memory.memory_types[0].add_event(
                    event_type=event_type,
                    description=description,
                    metadata=metadata_dict
                )
                st.success("Patient interaction added successfully!")
            except Exception as e:
                st.error(f"Error adding patient interaction: {str(e)}")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[1].get_event_patterns()
            st.write("Healthcare Patterns:", patterns)

def legal_assistant_interface():
    """Legal Assistant interface."""
    st.header("Legal Assistant")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="legal_knowledge.json"
        )
        event_memory = EventSourcedMemory(
            llm=OllamaLLM(model="mistral"),
            max_events=10000,
            max_snapshots=1000,
            storage_path="legal_events.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, event_memory],
            storage_path="legal.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a legal assistant that helps with legal knowledge management and case tracking."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Case Tracking", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Legal Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Case Tracking")
        event_type = st.text_input("Event Type")
        description = st.text_area("Description")
        metadata = st.text_area("Metadata (JSON)")
        
        if st.button("Add Case Event"):
            try:
                metadata_dict = eval(metadata)
                st.session_state.memory.memory_types[1].add_event(
                    event_type=event_type,
                    description=description,
                    metadata=metadata_dict
                )
                st.success("Case event added successfully!")
            except Exception as e:
                st.error(f"Error adding case event: {str(e)}")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[1].get_event_patterns()
            st.write("Case Patterns:", patterns)

def educational_tutor_interface():
    """Educational Tutor interface."""
    st.header("Educational Tutor")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="education_knowledge.json"
        )
        scratchpad_memory = CognitiveScratchpadMemory(
            llm=OllamaLLM(model="mistral"),
            max_steps=1000,
            max_chains=100,
            chain_depth=10,
            storage_path="education_steps.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, scratchpad_memory],
            storage_path="education.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are an educational tutor that helps with knowledge management and learning progress tracking."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Learning Progress", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Educational Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Learning Progress")
        description = st.text_area("Learning Activity Description")
        
        if st.button("Start Learning Session"):
            chain_id = st.session_state.memory.memory_types[1].start_chain(description)
            st.success(f"Learning session started with ID: {chain_id}")
        
        chain_id = st.text_input("Session ID")
        step_description = st.text_area("Progress Step")
        
        if st.button("Add Progress Step"):
            st.session_state.memory.memory_types[1].add_step(
                chain_id=chain_id,
                step_description=step_description
            )
            st.success("Progress step added successfully!")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            if chain_id:
                steps = st.session_state.memory.memory_types[1].get_chain_steps(chain_id)
                st.write("Learning Progress:", steps)

def active_learning_interface():
    """Active Learning System interface."""
    st.header("Active Learning System")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="active_learning_knowledge.json"
        )
        event_memory = EventSourcedMemory(
            llm=OllamaLLM(model="mistral"),
            max_events=10000,
            max_snapshots=1000,
            storage_path="active_learning_events.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, event_memory],
            storage_path="active_learning.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are an active learning system that helps with knowledge acquisition and learning optimization."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Learning Events", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Learning Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Learning Events")
        event_type = st.text_input("Event Type")
        description = st.text_area("Description")
        metadata = st.text_area("Metadata (JSON)")
        
        if st.button("Add Learning Event"):
            try:
                metadata_dict = eval(metadata)
                st.session_state.memory.memory_types[1].add_event(
                    event_type=event_type,
                    description=description,
                    metadata=metadata_dict
                )
                st.success("Learning event added successfully!")
            except Exception as e:
                st.error(f"Error adding learning event: {str(e)}")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[1].get_event_patterns()
            st.write("Learning Patterns:", patterns)

def chatbot_with_memory_interface():
    """Chatbot with Memory interface."""
    st.header("Chatbot with Memory")
    
    if st.session_state.memory is None:
        knowledge_memory = KnowledgeGraphMemory(
            llm=OllamaLLM(model="mistral"),
            max_nodes=10000,
            max_edges=50000,
            similarity_threshold=0.7,
            storage_path="chatbot_knowledge.json"
        )
        event_memory = EventSourcedMemory(
            llm=OllamaLLM(model="mistral"),
            max_events=10000,
            max_snapshots=1000,
            storage_path="chatbot_events.json"
        )
        st.session_state.memory = HybridMemory(
            llm=OllamaLLM(model="mistral"),
            memory_types=[knowledge_memory, event_memory],
            storage_path="chatbot.json"
        )
        st.session_state.mm = initialize_multimind(
            st.session_state.memory,
            "You are a chatbot with memory that helps with conversation tracking and knowledge management."
        )
    
    tab1, tab2, tab3 = st.tabs(["Add Knowledge", "Chat History", "Query & Analyze"])
    
    with tab1:
        st.subheader("Add Chat Knowledge")
        subject = st.text_input("Subject")
        predicate = st.text_input("Predicate")
        object_ = st.text_input("Object")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8)
        
        if st.button("Add Knowledge"):
            st.session_state.memory.memory_types[0].add_knowledge(
                subject=subject,
                predicate=predicate,
                object=object_,
                confidence=confidence
            )
            st.success("Knowledge added successfully!")
    
    with tab2:
        st.subheader("Chat History")
        message = st.text_area("Message")
        
        if st.button("Send Message"):
            st.session_state.memory.memory_types[1].add_event(
                event_type="message",
                description=message,
                metadata={"timestamp": datetime.now().isoformat()}
            )
            st.success("Message added to history!")
    
    with tab3:
        st.subheader("Query & Analyze")
        query = st.text_input("Query")
        
        if st.button("Query"):
            response = st.session_state.mm.chat(query)
            st.write("Response:", response)
            
            related = st.session_state.memory.memory_types[0].get_related_concepts(query)
            st.write("Related Concepts:", related)
            
            patterns = st.session_state.memory.memory_types[1].get_event_patterns()
            st.write("Conversation Patterns:", patterns)

def ollama_chat_interface():
    """Ollama Chat interface."""
    st.header("Ollama Chat")
    
    # Initialize session state for Ollama chat
    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = "mistral"
    if 'ollama_history' not in st.session_state:
        st.session_state.ollama_history = []
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["mistral", "llama2", "codellama", "vicuna"],
        index=0
    )
    
    if model_name != st.session_state.ollama_model:
        st.session_state.ollama_model = model_name
        st.session_state.ollama_history = []
    
    # Chat interface
    st.subheader("Chat")
    message = st.text_area("Message")
    
    if st.button("Send"):
        try:
            llm = OllamaLLM(model=model_name)
            response = llm.chat(message)
            
            # Add to history
            st.session_state.ollama_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.ollama_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            st.success("Message sent successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Chat history
    st.subheader("Chat History")
    for msg in st.session_state.ollama_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            st.caption(msg["timestamp"])

def basic_agents_interface():
    """Basic Agents interface."""
    st.header("Basic Agents")
    
    # Initialize session state for agents
    if 'agent_history' not in st.session_state:
        st.session_state.agent_history = []
    
    # Model selection
    st.sidebar.subheader("Agent Settings")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["OpenAI", "Claude", "Mistral"],
        index=0
    )
    
    # Create model instance
    if model_type == "OpenAI":
        model = OpenAIModel(model="gpt-3.5-turbo", temperature=0.7)
    elif model_type == "Claude":
        model = ClaudeModel(model="claude-3-sonnet-20240229", temperature=0.7)
    else:
        model = MistralModel(model="mistral-medium", temperature=0.7)
    
    # Create agent
    memory = AgentMemory(max_history=50)
    calculator = CalculatorTool()
    agent = Agent(
        model=model,
        memory=memory,
        tools=[calculator],
        system_prompt="You are a helpful AI assistant that can perform calculations."
    )
    
    # Task input
    st.subheader("Task")
    task = st.text_area("Enter your task")
    
    if st.button("Run Task"):
        try:
            response = asyncio.run(agent.run(task))
            
            # Add to history
            st.session_state.agent_history.append({
                "task": task,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            st.success("Task completed successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Task history
    st.subheader("Task History")
    for entry in st.session_state.agent_history:
        st.write(f"Task: {entry['task']}")
        st.write(f"Response: {entry['response']}")
        st.caption(entry['timestamp'])
        st.divider()

def rag_interface():
    """RAG System interface."""
    st.header("RAG System")
    
    # Initialize session state for RAG
    if 'rag_documents' not in st.session_state:
        st.session_state.rag_documents = []
    
    # Model and embedder selection
    st.sidebar.subheader("RAG Settings")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["OpenAI", "Claude", "Mistral"],
        index=0
    )
    
    # Create model instance
    if model_type == "OpenAI":
        model = OpenAIModel(model="gpt-3.5-turbo", temperature=0.7)
        embedder = get_embedder("openai", model="text-embedding-ada-002")
    elif model_type == "Claude":
        model = ClaudeModel(model="claude-3-sonnet-20240229", temperature=0.7)
        embedder = get_embedder("openai", model="text-embedding-ada-002")
    else:
        model = MistralModel(model="mistral-medium", temperature=0.7)
        embedder = get_embedder("openai", model="text-embedding-ada-002")
    
    # Create RAG instance
    rag = RAG(
        embedder=embedder,
        vector_store="faiss",
        model=model,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=3
    )
    
    # Document input
    st.subheader("Add Documents")
    document = st.text_area("Enter document text")
    metadata = st.text_area("Metadata (JSON)")
    
    if st.button("Add Document"):
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
            asyncio.run(rag.add_documents([document], metadata=metadata_dict))
            st.session_state.rag_documents.append({
                "text": document,
                "metadata": metadata_dict,
                "timestamp": datetime.now().isoformat()
            })
            st.success("Document added successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Query interface
    st.subheader("Query")
    query = st.text_input("Enter your query")
    
    if st.button("Search"):
        try:
            results = asyncio.run(rag.query(query))
            
            st.write("Search Results:")
            for i, (doc, score) in enumerate(results, 1):
                st.write(f"Document {i} (Score: {score:.3f}):")
                st.write(f"Text: {doc.text}")
                st.write(f"Metadata: {doc.metadata}")
                st.divider()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Document history
    st.subheader("Document History")
    for doc in st.session_state.rag_documents:
        st.write(f"Text: {doc['text']}")
        st.write(f"Metadata: {doc['metadata']}")
        st.caption(doc['timestamp'])
        st.divider()

def mcp_workflow_interface():
    """MCP Workflow interface."""
    st.header("MCP Workflow")
    
    # Initialize session state for MCP
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []
    
    # Model selection
    st.sidebar.subheader("Workflow Settings")
    models = st.sidebar.multiselect(
        "Select Models",
        ["gpt-3.5", "claude-3", "mistral"],
        default=["gpt-3.5", "claude-3"]
    )
    
    # Create models
    model_instances = {}
    if "gpt-3.5" in models:
        model_instances["gpt-3.5"] = OpenAIModel(model="gpt-3.5-turbo", temperature=0.7)
    if "claude-3" in models:
        model_instances["claude-3"] = ClaudeModel(model="claude-3-sonnet-20240229", temperature=0.7)
    if "mistral" in models:
        model_instances["mistral"] = MistralModel(model="mistral-medium", temperature=0.7)
    
    # Create MCP executor
    executor = MCPExecutor()
    for name, model in model_instances.items():
        executor.register_model(name, model)
    
    # Workflow definition
    st.subheader("Workflow Definition")
    workflow_json = st.text_area("Workflow JSON", height=300)
    
    # Input parameters
    st.subheader("Input Parameters")
    params_json = st.text_area("Parameters JSON", height=100)
    
    if st.button("Execute Workflow"):
        try:
            workflow = json.loads(workflow_json)
            params = json.loads(params_json) if params_json else {}
            
            results = asyncio.run(executor.execute(workflow, params))
            
            # Add to history
            st.session_state.workflow_history.append({
                "workflow": workflow,
                "params": params,
                "results": results,
                "timestamp": datetime.now().isoformat()
            })
            
            st.success("Workflow executed successfully!")
            
            # Display results
            st.subheader("Results")
            for step_id, result in results.items():
                st.write(f"{step_id.upper()}:")
                st.write(result)
                st.divider()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Workflow history
    st.subheader("Workflow History")
    for entry in st.session_state.workflow_history:
        st.write("Workflow:")
        st.json(entry["workflow"])
        st.write("Parameters:")
        st.json(entry["params"])
        st.write("Results:")
        st.json(entry["results"])
        st.caption(entry["timestamp"])
        st.divider()

def main():
    st.title("MultiMind Playground")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Example selection
    example = st.sidebar.selectbox(
        "Select Example",
        [
            "Scientific Research",
            "Customer Service",
            "Project Management",
            "Content Creation",
            "Software Development",
            "Data Analysis",
            "Financial Analysis",
            "Healthcare Assistant",
            "Legal Assistant",
            "Educational Tutor",
            "Active Learning",
            "Chatbot with Memory",
            "Ollama Chat",
            "Basic Agents",
            "RAG System",
            "MCP Workflow"
        ]
    )
    
    # Reset memory when changing examples
    if st.session_state.current_example != example:
        st.session_state.memory = None
        st.session_state.mm = None
        st.session_state.current_example = example
        st.session_state.chat_history = []
    
    # Settings
    st.sidebar.title("Settings")
    st.session_state.visualization_type = st.sidebar.selectbox(
        "Visualization Type",
        [
            "Knowledge Graph",
            "Event Timeline",
            "Analysis Steps",
            "Network Graph",
            "Sunburst Chart",
            "Heatmap",
            "3D Scatter",
            "Parallel Coordinates"
        ]
    )
    
    st.session_state.export_format = st.sidebar.selectbox(
        "Export Format",
        ["JSON", "CSV", "Excel", "Parquet", "HTML"]
    )
    
    # Display selected example interface
    if example == "Scientific Research":
        scientific_research_interface()
    elif example == "Customer Service":
        customer_service_interface()
    elif example == "Project Management":
        project_management_interface()
    elif example == "Content Creation":
        content_creation_interface()
    elif example == "Software Development":
        software_development_interface()
    elif example == "Data Analysis":
        data_analysis_interface()
    elif example == "Financial Analysis":
        financial_analysis_interface()
    elif example == "Healthcare Assistant":
        healthcare_assistant_interface()
    elif example == "Legal Assistant":
        legal_assistant_interface()
    elif example == "Educational Tutor":
        educational_tutor_interface()
    elif example == "Active Learning":
        active_learning_interface()
    elif example == "Chatbot with Memory":
        chatbot_with_memory_interface()
    elif example == "Ollama Chat":
        ollama_chat_interface()
    elif example == "Basic Agents":
        basic_agents_interface()
    elif example == "RAG System":
        rag_interface()
    elif example == "MCP Workflow":
        mcp_workflow_interface()
    
    # Common features
    st.sidebar.title("Common Features")
    
    # Data Management
    st.sidebar.subheader("Data Management")
    with st.sidebar.expander("Export Options"):
        # Filter options
        st.write("Filters")
        filter_key = st.text_input("Filter Key")
        filter_value = st.text_input("Filter Value")
        if filter_key and filter_value:
            st.session_state.filter_options[filter_key] = filter_value
        
        # Sort options
        st.write("Sort Options")
        sort_by = st.text_input("Sort By")
        if sort_by:
            st.session_state.sort_options = sort_by
        
        if st.button("Export Data"):
            data = export_data_advanced(
                st.session_state.memory,
                st.session_state.export_format,
                st.session_state.filter_options,
                st.session_state.sort_options
            )
            st.sidebar.download_button(
                "Download Data",
                data,
                file_name=f"{example.lower().replace(' ', '_')}_data.{st.session_state.export_format.lower()}",
                mime=f"application/{st.session_state.export_format.lower()}"
            )
    
    with st.sidebar.expander("Import Options"):
        uploaded_file = st.file_uploader(
            "Import Data",
            type=[st.session_state.export_format.lower()]
        )
        if uploaded_file is not None:
            if st.button("Import"):
                if import_data_advanced(
                    st.session_state.memory,
                    uploaded_file,
                    st.session_state.export_format
                ):
                    st.sidebar.success("Data imported successfully!")
    
    # Visualization
    st.sidebar.subheader("Visualization")
    if st.session_state.memory is not None:
        with st.container():
            st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
            if st.session_state.visualization_type == "Knowledge Graph":
                st.plotly_chart(visualize_knowledge_graph(st.session_state.memory))
            elif st.session_state.visualization_type == "Event Timeline":
                st.plotly_chart(visualize_event_timeline(st.session_state.memory))
            elif st.session_state.visualization_type == "Analysis Steps":
                chain_id = st.sidebar.text_input("Chain ID")
                if chain_id:
                    st.plotly_chart(visualize_analysis_steps(st.session_state.memory, chain_id))
            elif st.session_state.visualization_type == "Network Graph":
                create_network_graph(st.session_state.memory)
            elif st.session_state.visualization_type == "Sunburst Chart":
                st.plotly_chart(create_sunburst_chart(st.session_state.memory))
            elif st.session_state.visualization_type == "Heatmap":
                st.plotly_chart(create_heatmap(st.session_state.memory))
            elif st.session_state.visualization_type == "3D Scatter":
                st.plotly_chart(create_3d_scatter(st.session_state.memory))
            elif st.session_state.visualization_type == "Parallel Coordinates":
                st.plotly_chart(create_parallel_coordinates(st.session_state.memory))
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics
    st.sidebar.subheader("Statistics")
    if st.session_state.memory is not None:
        with st.container():
            st.markdown('<div class="data-management-container">', unsafe_allow_html=True)
            show_advanced_statistics(st.session_state.memory)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat history
    st.sidebar.subheader("Chat History")
    show_chat_history()

if __name__ == "__main__":
    main() 