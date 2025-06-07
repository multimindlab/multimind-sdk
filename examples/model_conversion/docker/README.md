# Docker Examples for Model Conversion

This directory contains Docker-related files for running model conversion in a containerized environment.

## Files

- `Dockerfile`: Defines the container image for model conversion
- `docker-compose.yml`: Orchestrates the model conversion and Ollama services

## Prerequisites

- Docker
- Docker Compose
- NVIDIA GPU with CUDA support (optional)

## Usage

1. Build and start the services:
```bash
docker-compose up --build
```

2. The services will be available with the following configurations:
   - Model converter service:
     - Mounted volumes:
       - `/app/models`: For input models
       - `/app/output`: For converted models
     - GPU support enabled
   - Ollama service:
     - Port: 11434
     - Volume: ollama_data for model storage

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Controls GPU visibility (default: 0)
- `PYTHONPATH`: Set to /app for proper module resolution

## Volumes

- `models`: For input models
- `output`: For converted models
- `ollama_data`: For Ollama model storage

## GPU Support

The services are configured to use NVIDIA GPUs if available. Make sure you have:
1. NVIDIA drivers installed
2. NVIDIA Container Toolkit installed
3. Docker configured to use NVIDIA runtime 