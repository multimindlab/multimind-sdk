FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY requirements-base.txt .
COPY requirements-compliance.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model conversion code
COPY multimind/model_conversion /app/multimind/model_conversion

# Copy example scripts
COPY examples/model_conversion /app/examples/model_conversion

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "/app/examples/model_conversion/model_conversion_example.py"] 