# MultiMind SDK Core Module

This document provides detailed documentation for the core components of the MultiMind SDK.

## Core Components

### Router
- Intelligent request routing and provider management
- Multiple routing strategies (cost-based, latency-based, quality-based)
- Provider fallback mechanisms
- Request metrics collection

### Provider
- AI service provider adapters and interfaces
- Unified provider interface
- Provider-specific configurations
- Error handling and retries

### Configuration
- Centralized configuration management
- Environment variable handling
- Provider configuration
- System-wide settings

### Monitoring
- Performance tracking
- Metrics collection
- Cost monitoring
- Health checks

### Base Components
- Base classes and interfaces
- Common functionality
- Standardized interfaces
- Extension points

### Local Runner
- Local execution capabilities
- Offline mode support
- Local model management
- Resource optimization

## Key Features

- Unified interface for multiple AI providers
- Intelligent request routing
- Performance monitoring
- Configuration management
- Local execution support

## Usage Examples

### Router Configuration
```python
from multimind.core.router import Router, RoutingStrategy

router = Router()
router.configure_task(
    task_type="text_generation",
    config={
        "preferred_providers": ["openai", "anthropic"],
        "fallback_providers": ["cohere"],
        "routing_strategy": RoutingStrategy.CASCADE
    }
)
```

### Provider Management
```python
from multimind.core.provider import ProviderAdapter

# Register a new provider
router.register_provider("custom_provider", ProviderAdapter())
```

### Configuration
```python
from multimind.core.config import Config

config = Config()
settings = config.load()
```

## Related Documentation

- [Architecture Overview](../docs/architecture.md)
- [API Reference](../docs/api_reference/)
- [Configuration Guide](../docs/configuration.md)
- [Development Guide](../docs/development.md)

## Model Client Architecture and Routing

The MultiMind SDK provides an extensible model client system:
- **ModelClient**: Base class for all model clients. Subclass to implement custom models.
- **Prebuilt Clients**: Includes LSTMModelClient, RNNModelClient, GRUModelClient, MoEModelClient, DynamicMoEModelClient, MultiModalClient, and more.
- **Routing**: Use MoE and DynamicMoE for expert selection, or FederatedRouter for local/cloud routing.
- **Extensibility**: Add new model types or routing strategies by subclassing ModelClient or implementing custom routers.

This architecture enables advanced workflows, dynamic model selection, and multimodal support. 