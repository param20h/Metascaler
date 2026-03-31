# Use Python 3.11 slim base
FROM python:3.11-slim

# Metadata
LABEL maintainer="metaXscaler"
LABEL description="SQL Query Optimizer — OpenEnv Environment"

# Set working directory
WORKDIR /app

# Install dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces default port
EXPOSE 7860

# Start the FastAPI server
# Using server.app:app for proper module path resolution
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
