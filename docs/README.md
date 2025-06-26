# MultiMind Documentation

Welcome to the MultiMind documentation! This documentation will help you understand and use the MultiMind SDK effectively.

## Table of Contents

1. [Getting Started](getting_started.md)
2. [Core Concepts](core_concepts.md)
3. [Pipeline System](pipeline.md)
4. [Ensemble System](ensemble.md)
5. [CLI Interface](cli.md)
6. [API Reference](api_reference/README.md)
   - [REST API](api_reference/rest_api.md)
   - [Python API](api_reference/python_api.md)
   - [WebSocket API](api_reference/websocket_api.md)
   - [Authentication](api_reference/authentication.md)
   - [Rate Limiting](api_reference/rate_limiting.md)
   - [Error Codes](api_reference/error_codes.md)
   - [Data Types](api_reference/data_types.md)
   - [Webhooks](api_reference/webhooks.md)
   - [SDKs](api_reference/sdks.md)
7. [Examples](examples.md)
8. [Configuration](configuration.md)
9. [Troubleshooting](troubleshooting.md)

## Quick Links

- [Installation Guide](getting_started.md#installation)
- [Basic Usage](getting_started.md#basic-usage)
- [Pipeline Examples](pipeline.md#examples)
- [Ensemble Examples](ensemble.md#examples)
- [CLI Commands](cli.md#available-commands)
- [REST API Endpoints](api_reference/rest_api.md#available-endpoints)

## Overview

MultiMind is a powerful SDK that provides a unified interface for working with multiple AI models and providers. It offers:

- A flexible pipeline system for complex AI workflows
- An ensemble system for combining results from multiple providers
- **Extensible model client system for custom, multimodal, and mixture-of-experts (MoE) models**
- **Advanced routing logic (MoE, dynamic, federated/local-cloud) for optimal model selection**
- Command-line interface for quick access to functionality
- REST API for integration with other systems
- Support for various AI tasks and providers

## Key Features

### Pipeline System
- Define and execute complex AI workflows
- Support for multiple task types
- Flexible configuration and customization
- Error handling and retry mechanisms

### Ensemble System
- Combine results from multiple AI providers
- Various ensemble methods (weighted voting, confidence cascade, etc.)
- Configurable provider weights and confidence thresholds
- Detailed confidence scoring and explanations

### Model Client System & Routing
- **Extensible ModelClient base class** for custom and classic models (LSTM, RNN, GRU, etc.)
- **Mixture-of-Experts (MoE) and DynamicMoE** for expert selection and runtime optimization
- **MultiModalClient** for unified text, image, audio, video, and code model access
- **FederatedRouter** for local/cloud and custom routing strategies

### CLI Interface
- Easy-to-use command-line tools
- Support for all major features
- JSON output format
- Comprehensive error handling

### API Interface
- RESTful API endpoints
- Swagger/OpenAPI documentation
- Support for all features
- Easy integration with other systems

## Getting Help

- Check the [Troubleshooting](troubleshooting.md) guide for common issues
- Review the [Examples](examples.md) for usage patterns
- Visit our [GitHub repository](https://github.com/your-repo/multimind) for updates
- Join our [Discord community](https://discord.gg/your-server) for support

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.

## License

MultiMind is licensed under the [MIT License](LICENSE). 