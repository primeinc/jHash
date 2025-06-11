# jHash MVP Development Requirements - Single Source of Truth

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Target Platform**: NVIDIA Jetson AGX Orin 64GB  
**JetPack Version**: r35.4.1

## Executive Summary

This document constitutes the authoritative specification for all development dependencies, environment configurations, and system requirements for the jHash MVP. It supersedes all other dependency specifications and serves as the single reference point for environment setup, package management, and deployment configuration.

## 1. System Requirements

### 1.1 Hardware Specifications

```yaml
platform:
  device: NVIDIA Jetson AGX Orin
  memory: 64GB unified memory
  compute_capability: 8.7 (Ampere)
  storage: 250GB+ available
  architecture: aarch64/ARM64
```

### 1.2 Base Operating System

```yaml
os:
  distribution: Ubuntu 20.04 LTS (L4T)
  kernel: 5.10.104-tegra
  jetpack: 5.1.2 (r35.4.1)
```

## 2. System-Level Dependencies

### 2.1 Essential System Packages

```bash
# File: system-packages.txt
# Install with: sudo apt-get update && sudo apt-get install -y $(cat system-packages.txt)

# Core Development Tools
build-essential
cmake
ninja-build
git
wget
curl
htop
nano
vim

# Python Development
python3.8-dev
python3-pip
python3-setuptools
python3-wheel
python3-venv

# Linear Algebra & Mathematics
libopenblas-base
libopenblas-dev
libatlas-base-dev
liblapack-dev
libblas-dev
gfortran
libeigen3-dev

# Image Processing
libjpeg8-dev
libjpeg-dev
libpng-dev
libtiff5-dev
libwebp-dev
libopenjp2-7-dev
libdc1394-22-dev

# Video Processing
libavcodec-dev
libavformat-dev
libswscale-dev
libv4l-dev
libxvidcore-dev
libx264-dev
libgstreamer1.0-dev
libgstreamer-plugins-base1.0-dev

# Data Storage
libhdf5-serial-dev
hdf5-tools
libhdf5-dev
zlib1g-dev
liblmdb-dev

# Networking & Security
libssl-dev
libffi-dev
ca-certificates

# System Monitoring
nvidia-jetpack
jtop
```

### 2.2 NVIDIA-Specific Runtime

```bash
# Pre-installed with JetPack r35.4.1
# Verify with: dpkg -l | grep nvidia

nvidia-l4t-core
nvidia-l4t-cuda
nvidia-l4t-cudnn
nvidia-l4t-tensorrt
nvidia-container-runtime
```

## 3. Python Package Requirements

### 3.1 Core Python Dependencies

```ini
# File: requirements.txt
# Install with: pip install -r requirements.txt

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Database & Storage
duckdb==0.9.2
faiss-cpu==1.7.4  # ARM64 wheel from conda-forge recommended

# AI/ML Core
numpy==1.24.4
scipy==1.10.1
scikit-learn==1.3.2
pandas==2.0.3

# Image Processing
opencv-python==4.8.1.78
Pillow==10.1.0
Pillow-SIMD==9.5.0  # If available for ARM64
ImageHash==4.3.1

# Deep Learning (Jetson-specific versions)
# torch==2.1.0a0+41361538.nv23.06  # Install from NVIDIA wheel
# torchvision==0.16.0a0  # Install from NVIDIA wheel
# torchaudio==2.1.0a0  # Install from NVIDIA wheel

# Model Conversion
onnx==1.15.0
onnxruntime==1.16.3
# tensorrt==8.5.2.2  # Pre-installed with JetPack

# Transformers & Model Loading
transformers==4.36.2
huggingface-hub==0.19.4
safetensors==0.4.1
tokenizers==0.15.0

# CLI Tools
click==8.1.7
rich==13.7.0
tqdm==4.66.1

# Async & Concurrent Processing
aiofiles==23.2.1
asyncio==3.4.3
watchdog==3.0.0  # Removed from MVP, but listed for completeness

# Testing & Development
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.12.0
mypy==1.7.1

# Monitoring & Logging
prometheus-client==0.19.0
python-json-logger==2.0.7
```

### 3.2 PyTorch Installation (Jetson-Specific)

```bash
# File: install-pytorch.sh
#!/bin/bash
set -e

# PyTorch 2.1.0 for JetPack 5.1.2
wget -q https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# TorchVision
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev
git clone --branch v0.16.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.0
python3 setup.py install --user
cd .. && rm -rf torchvision
```

### 3.3 Development Dependencies

```ini
# File: requirements-dev.txt
# Install with: pip install -r requirements-dev.txt

# Code Quality
flake8==6.1.0
pylint==3.0.3
bandit==1.7.5

# Documentation
mkdocs==1.5.3
mkdocs-material==9.4.14
pydoc-markdown==4.8.2

# Performance Profiling
py-spy==0.3.14
memory-profiler==0.61.0
line-profiler==4.1.2
```

## 4. Environment Configuration

### 4.1 System Environment Variables

```bash
# File: .env.system
# Source with: source .env.system

# CUDA Configuration
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Jetson-Specific
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=12

# Python Configuration
export PYTHONPATH=/app:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Model Cache (Offline Mode)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Application Configuration
export JHASH_ENV=production
export JHASH_LOG_LEVEL=INFO
```

### 4.2 Application Configuration

```yaml
# File: config.yaml
application:
  name: jHash
  version: 1.0.0
  
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
models:
  clip:
    name: openai/clip-vit-base-patch32
    engine_path: addons/models/clip_vitb32.engine
    embedding_dim: 512
    
  dino:
    name: facebook/dinov2-vitb14
    engine_path: addons/models/dino_vitb14.engine
    embedding_dim: 768
    
faiss:
  index_type: IndexIVFFlat
  nlist: 8192
  metric: METRIC_L2
  
duckdb:
  path: db/metadata.duckdb
  max_memory: 4GB
```

## 5. Docker Configuration

### 5.1 Base Dockerfile

```dockerfile
# File: Dockerfile.jetson
FROM dusty-nv/l4t-pytorch:r35.4.1

# System dependencies
COPY system-packages.txt /tmp/
RUN apt-get update && \
    apt-get install -y $(cat /tmp/system-packages.txt) && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Application
WORKDIR /app
COPY . .

# TensorRT engines
RUN mkdir -p addons/models

ENV PYTHONPATH=/app
ENV HF_HUB_OFFLINE=1

CMD ["python3", "backend/main.py"]
```

### 5.2 Docker Compose Configuration

```yaml
# File: docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: jetson-deploy/Dockerfile.jetson
    image: jhash-backend:jetson-r35.4.1
    container_name: jhash_backend
    runtime: nvidia
    volumes:
      - ./db:/app/db
      - ${JHASH_IMAGE_DATA_PATH:-./data}:/app/data
      - ./logs:/app/logs
    ports:
      - "${JHASH_BACKEND_PORT:-8000}:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

## 6. Pre-Installation Validation

### 6.1 System Validation Script

```bash
#!/bin/bash
# File: validate-environment.sh

echo "=== jHash Environment Validation ==="

# Check Jetson Model
if [ -f /etc/nv_tegra_release ]; then
    echo "✓ NVIDIA Jetson detected"
    cat /etc/nv_tegra_release
else
    echo "✗ Not running on Jetson platform"
    exit 1
fi

# Check JetPack Version
jetpack_version=$(dpkg -l | grep nvidia-jetpack | awk '{print $3}')
echo "JetPack Version: $jetpack_version"

# Check CUDA
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA installed: $(nvcc --version | grep release)"
else
    echo "✗ CUDA not found"
fi

# Check Python
python_version=$(python3 --version)
echo "Python Version: $python_version"

# Check Docker
if command -v docker &> /dev/null; then
    echo "✓ Docker installed: $(docker --version)"
    # Check nvidia-runtime
    if docker info | grep -q nvidia; then
        echo "✓ NVIDIA runtime configured"
    else
        echo "✗ NVIDIA runtime not configured"
    fi
else
    echo "✗ Docker not installed"
fi

# Check available memory
total_mem=$(free -g | awk '/^Mem:/{print $2}')
echo "Total Memory: ${total_mem}GB"

# Check storage
available_storage=$(df -BG /home | awk 'NR==2 {print $4}')
echo "Available Storage: $available_storage"
```

## 7. Troubleshooting Guide

### 7.1 Common Issues and Resolutions

|Issue                     |Symptom                                                             |Resolution                                     |
|--------------------------|--------------------------------------------------------------------|-----------------------------------------------|
|CUDA library not found    |`libcudart.so.11.0: cannot open shared object file`                 |Run `sudo ldconfig` and verify LD_LIBRARY_PATH |
|PyTorch import fails      |`Illegal instruction (core dumped)`                                 |Ensure correct wheel for JetPack version       |
|Faiss compilation error   |`cc1plus: error: unrecognized command line option '-mavx2'`         |Use ARM64-specific flags: `-mcpu=cortex-a78`   |
|Docker GPU access denied  |`docker: Error response from daemon: could not select device driver`|Install nvidia-container-runtime               |
|Out of memory during build|`c++: fatal error: Killed signal terminated program`                |Increase swap: `sudo fallocate -l 8G /swapfile`|

## 8. Version Control & Updates

### 8.1 Dependency Update Protocol

1. Test updates in isolated environment
1. Verify ARM64 compatibility
1. Benchmark performance impact
1. Update this document with new versions
1. Tag repository with dependency version

### 8.2 Document Maintenance

- Review quarterly for security updates
- Update for new JetPack releases
- Track breaking changes in CHANGELOG.md

## 9. Compliance & Validation

This document satisfies the following compliance requirements:

- ✓ Complete dependency specification
- ✓ Reproducible environment setup
- ✓ Offline deployment capability
- ✓ Platform-specific optimizations
- ✓ Version pinning for stability

**Document Status**: APPROVED FOR IMPLEMENTATION
