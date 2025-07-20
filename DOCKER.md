# Docker Setup for OmniGen2

This guide explains how to run OmniGen2 using Docker with GPU support.

## Prerequisites

1. **NVIDIA GPU** with at least 17GB VRAM (RTX 3090 or equivalent)
2. **Docker** installed on your system
3. **NVIDIA Container Toolkit** installed for GPU support

### Installing NVIDIA Container Toolkit

For Ubuntu/Debian:
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

For other systems, see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Quick Start

### Build and Run with Docker

```bash
# Build the image
docker build -t omnigen2-server .

# Run the container
docker run -d \
  --name omnigen2-server \
  --gpus all \
  -p 9432:8000 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  -v $(pwd)/outputs:/app/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  omnigen2-server
```

The server will be available at `http://localhost:9432`

## Configuration Options

### Environment Variables

You can customize the server behavior using environment variables:

```bash
# Example with custom settings
docker run -d \
  --name omnigen2-server \
  --gpus all \
  -p 9432:8000 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  -v $(pwd)/outputs:/app/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  omnigen2-server \
  python app_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model_path "OmniGen2/OmniGen2" \
  --dtype "bf16" \
  --enable_model_cpu_offload
```

### Command Line Arguments

The app_server.py supports various command line arguments:

- `--model_path`: Path to model checkpoint (default: "OmniGen2/OmniGen2")
- `--transformer_path`: Path to transformer checkpoint
- `--transformer_lora_path`: Path to transformer LoRA checkpoint
- `--scheduler`: Scheduler to use ("euler" or "dpmsolver", default: "euler")
- `--dtype`: Data type for model weights ("fp32", "fp16", "bf16", default: "bf16")
- `--enable_model_cpu_offload`: Enable model CPU offload (reduces VRAM by ~50%)
- `--enable_sequential_cpu_offload`: Enable sequential CPU offload (uses <3GB VRAM but slower)
- `--enable_group_offload`: Enable group offload
- `--host`: Host to bind the server to (default: "0.0.0.0")
- `--port`: Port to bind the server to (default: 8000)

## Memory Optimization

### For GPUs with Limited VRAM

If you have less than 17GB VRAM, use CPU offloading:

```bash
# Run with model CPU offload (reduces VRAM by ~50%)
docker run -d \
  --name omnigen2-server \
  --gpus all \
  -p 9432:8000 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  -v $(pwd)/outputs:/app/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  omnigen2-server \
  python app_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model_path "OmniGen2/OmniGen2" \
  --dtype "bf16" \
  --enable_model_cpu_offload
```

For very limited VRAM (< 8GB):
```bash
# Run with sequential CPU offload (uses <3GB VRAM but slower)
docker run -d \
  --name omnigen2-server \
  --gpus all \
  -p 9432:8000 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  -v $(pwd)/outputs:/app/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  omnigen2-server \
  python app_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model_path "OmniGen2/OmniGen2" \
  --dtype "bf16" \
  --enable_sequential_cpu_offload
```

## Development Mode

For development with live code reloading:

```bash
# Run in development mode with source code mounted
docker run -d \
  --name omnigen2-dev \
  --gpus all \
  -p 9433:8000 \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  -v $(pwd)/outputs:/app/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/omnigen2:/app/omnigen2 \
  -v $(pwd)/app_server.py:/app/app_server.py \
  omnigen2-server \
  python app_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model_path "OmniGen2/OmniGen2" \
  --dtype "bf16" \
  --enable_model_cpu_offload
```

Development server will be available at `http://localhost:9433`

## API Usage Examples

Once the server is running, you can test it:

### Health Check
```bash
curl http://localhost:9432/health
```

### Text-to-Image Generation
```bash
curl -X POST "http://localhost:9432/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A vector illustration of a Panda eating a cake.",
        "n": 1,
        "size": "1024x1024",
        "quality": "high"
    }' | jq -r '.data[0].b64_json' | base64 --decode > output.png
```

### Image Editing
```bash
curl -X POST "http://localhost:9432/v1/images/edits" \
    -F "prompt=Change the background to a classroom" \
    -F "image=@your_image.png" \
    -F "n=1" \
    -F "size=auto" \
    -F "quality=high" | jq -r '.data[0].b64_json' | base64 --decode > edited.png
```

## Troubleshooting

### Common Issues

1. **GPU not detected**:
   ```bash
   # Check if NVIDIA runtime is available
   docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi
   ```

2. **Out of memory errors**:
   - Enable CPU offloading with `--enable_model_cpu_offload`
   - Use sequential CPU offload with `--enable_sequential_cpu_offload`
   - Reduce batch size or image resolution

3. **Model not found**:
   - Ensure models are downloaded to `./pretrained_models/`
   - Check the `--model_path` argument

4. **Permission issues**:
   ```bash
   # Fix permissions for mounted volumes
   sudo chown -R $USER:$USER ./pretrained_models ./outputs
   ```

### Logs and Debugging

```bash
# Check container status
docker ps -a

# View logs to see why container exited
docker logs omnigen2-server

# View logs in real-time (if container is running)
docker logs -f omnigen2-server

# Enter the container for debugging (if running)
docker exec -it omnigen2-server bash

# Check GPU usage inside container (if running)
docker exec -it omnigen2-server nvidia-smi

# Run container interactively for debugging
docker run -it --rm --gpus all \
  -v $(pwd)/pretrained_models:/app/pretrained_models \
  -v $(pwd)/outputs:/app/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  omnigen2-server bash
```

### Common Exit Issues

1. **Missing models**: The container may exit if the model path doesn't exist
   ```bash
   # Create the model directory first
   mkdir -p pretrained_models
   
   # Or run with a different model path
   docker run -d \
     --name omnigen2-server \
     --gpus all \
     -p 9432:8000 \
     -v $(pwd)/pretrained_models:/app/pretrained_models \
     -v $(pwd)/outputs:/app/outputs \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     omnigen2-server \
     python app_server.py \
     --host 0.0.0.0 \
     --port 8000 \
     --model_path "OmniGen2/OmniGen2"
   ```

2. **Python path issues**: Try running with explicit Python path
   ```bash
   docker run -d \
     --name omnigen2-server \
     --gpus all \
     -p 9432:8000 \
     -v $(pwd)/pretrained_models:/app/pretrained_models \
     -v $(pwd)/outputs:/app/outputs \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     omnigen2-server \
     python3 app_server.py --host 0.0.0.0 --port 8000
   ```

3. **Permission issues**: Fix volume permissions
   ```bash
   sudo chown -R $USER:$USER ./pretrained_models ./outputs
   chmod -R 755 ./pretrained_models ./outputs
   ```

## Performance Tips

1. **Use flash-attn**: Already included in the Docker image for optimal performance
2. **Adjust cfg_range_end**: Reduce this parameter to speed up inference with minimal quality impact
3. **Use appropriate dtype**: `bf16` offers the best balance of speed and quality
4. **Enable model CPU offload**: Reduces VRAM usage by ~50% with minimal speed impact

## Stopping the Service

```bash
# Stop and remove the container
docker stop omnigen2-server
docker rm omnigen2-server

# Remove the image (optional)
docker rmi omnigen2-server
```

## Building for Different Architectures

The Dockerfile supports multi-architecture builds:

```bash
# Build for specific platform
docker buildx build --platform linux/amd64 -t omnigen2-server .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t omnigen2-server .
```

## Security Considerations

- The server runs on `0.0.0.0:8000` internally (mapped to port 9432 externally), making it accessible from any network interface
- For production use, consider:
  - Using a reverse proxy (nginx, traefik)
  - Adding authentication
  - Restricting network access
  - Using HTTPS/TLS

## Resource Requirements

- **Minimum**: 17GB GPU VRAM (with CPU offload: ~8GB)
- **Recommended**: 24GB+ GPU VRAM for optimal performance
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 50GB+ for models and temporary files
