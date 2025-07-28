# Docker Setup for MultiMind SDK

This guide provides detailed instructions for running MultiMind SDK using Docker and Docker Compose.

## Prerequisites

- Docker Engine (version 20.10.0 or higher)
- Docker Compose (version 2.0.0 or higher)
- Git

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/multimind-dev/multimind-sdk.git
cd multimind-sdk
```

2. Create a `.env` file:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. Start the services:
```bash
docker-compose up -d
```

## Services Overview

The MultiMind SDK Docker setup includes the following services:

### 1. MultiMind Service
- Main application service
- Port: 8000
- Features:
  - REST API
  - WebSocket API
  - CLI interface
  - Model management
  - Ensemble system
  - Pipeline system

### 2. Redis
- Caching and session management
- Port: 6379
- Features:
  - In-memory data store
  - Session management
  - Rate limiting
  - Task queue

### 3. Chroma
- Vector database for RAG
- Port: 8001
- Features:
  - Vector storage
  - Similarity search
  - Metadata management

### 4. Ollama
- Local model service
- Port: 11434
- Features:
  - Local model hosting
  - Model management
  - Inference API

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
HF_TOKEN=your_huggingface_token_here

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Chroma Configuration
CHROMA_HOST=chroma
CHROMA_PORT=8000

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Model Configuration
DEFAULT_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
VISION_MODEL=gpt-4-vision-preview

# Ollama Configuration
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_MODEL=mistral
```

### Volumes

The setup uses the following volumes:

- `redis_data`: Redis persistent storage
- `chroma_data`: Chroma vector database storage
- `ollama_data`: Ollama model storage
- `multimind_logs`: Application logs
- `multimind_data`: Application data
- `multimind_cache`: Application cache

## Usage

### Starting Services

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d multimind redis

# View logs
docker-compose logs -f
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Accessing Services

- MultiMind API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Chroma API: http://localhost:8001
- Redis: localhost:6379
- Ollama API: http://localhost:11434

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# View health check logs
docker-compose events
```

## Development

### Hot Reloading

The setup includes hot reloading for development:

```bash
# Mount source code
docker-compose up -d

# View logs in real-time
docker-compose logs -f multimind
```

### Debugging

Enable debug mode in `.env`:

```env
DEBUG=true
LOG_LEVEL=DEBUG
```

### Building Custom Images

```bash
# Build specific service
docker-compose build multimind

# Build with no cache
docker-compose build --no-cache multimind
```

## Security

### Network Isolation

Services are isolated in a dedicated network:
```yaml
networks:
  default:
    name: multimind_network
```

### Volume Permissions

Proper permissions are set in the Dockerfile:
```dockerfile
RUN chmod -R 755 /app
```

### Environment Variables

Sensitive data is managed through environment variables:
```yaml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - CLAUDE_API_KEY=${CLAUDE_API_KEY}
  - HF_TOKEN=${HF_TOKEN}
```

## Monitoring

### Logs

Access logs for each service:
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs multimind

# Follow logs
docker-compose logs -f
```

### Health Checks

Monitor service health:
```bash
# Check status
docker-compose ps

# View health check results
docker inspect --format='{{json .State.Health}}' multimind-sdk_multimind_1
```

## Troubleshooting

### Common Issues

1. **Service Not Starting**
   ```bash
   # Check logs
   docker-compose logs multimind
   
   # Check health status
   docker-compose ps
   ```

2. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :8000
   
   # Change port in docker-compose.yml
   ports:
     - "8001:8000"
   ```

3. **Volume Issues**
   ```bash
   # Check volume permissions
   docker-compose exec multimind ls -la /app
   
   # Recreate volumes
   docker-compose down -v
   docker-compose up -d
   ```

### Debugging Commands

```bash
# Enter service container
docker-compose exec multimind bash

# View service logs
docker-compose logs -f multimind

# Check service status
docker-compose ps

# Restart service
docker-compose restart multimind
```

## Best Practices

1. **Resource Management**
   - Monitor container resource usage
   - Set appropriate resource limits
   - Use volume mounts for persistent data

2. **Security**
   - Keep API keys secure
   - Use environment variables
   - Regular security updates
   - Network isolation

3. **Maintenance**
   - Regular image updates
   - Clean up unused volumes
   - Monitor logs
   - Backup important data

4. **Development**
   - Use hot reloading
   - Enable debug mode when needed
   - Follow logging best practices
   - Use health checks

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MultiMind SDK Documentation](../README.md)
- [API Reference](../api_reference/README.md)