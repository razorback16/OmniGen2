# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies including Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies following README instructions
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# Install other required packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Install flash-attn for optimal performance (version specified for CUDA 12.4 compatibility)
RUN pip3 install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Copy the entire project
COPY . .

# Create directories for models, outputs, static files, and HuggingFace cache
RUN mkdir -p /app/pretrained_models /app/outputs /app/static/svgs /root/.cache/huggingface

# Set BASE_URL environment variable for vectorize endpoint
ENV BASE_URL="https://svgmaker.subhagato.com"

# Set environment variables for model management
ENV MODEL_IDLE_TIMEOUT_SECONDS=7200

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "app_server.py", "--host", "0.0.0.0", "--port", "8000"]
