FROM python:3.11-slim

# Install nvidia-smi (nvidia-utils) if available
# Note: For GPU access, you need nvidia-docker runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY gpu_exporter.py .

# Expose port
EXPOSE 9100

# Run application
CMD ["python", "gpu_exporter.py"]

