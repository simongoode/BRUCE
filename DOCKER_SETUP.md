# Docker Setup Guide for BRUCE

Quick guide for getting BRUCE running with Docker.

## Prerequisites

1. **Docker & Docker Compose**
   - [Install Docker](https://docs.docker.com/get-docker/)
   - Docker Compose usually comes with Docker Desktop

2. **NVIDIA Docker Runtime** (for GPU support)
   ```bash
   # Check if you have it
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```
   
   If not installed, see [installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

3. **Google API Key**
   - Get one from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Installation (5 minutes)

```bash
# 1. Clone the repo
git clone https://github.com/simongoode/BRUCE.git
cd BRUCE

# 2. Add your API key
echo "GOOGLE_API_KEY=your_actual_key_here" > bruce_flows/.env

# 3. Build and run
docker-compose up --build
```

First build takes ~10 minutes. Results appear in `bruce_flows/results/`.

## Daily Usage

### Run the analysis
```bash
docker-compose up
```

### Edit code and re-run
Just edit files normally, then:
```bash
docker-compose run bruce crewai run
```

### Interactive shell
```bash
docker-compose run bruce bash
```

### Stop everything
```bash
docker-compose down
```

## File Locations

- **Edit code**: `bruce_flows/src/bruce_flows/`
- **View results**: `bruce_flows/results/`
- **Config files**: `bruce_flows/src/bruce_flows/crews/parameter_expert_crew/config/`

## Common Issues

### "permission denied" when running docker
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### "could not select device driver" or GPU errors
Make sure NVIDIA Docker runtime is installed (see prerequisites)

### Container exits immediately
Check your `.env` file has the API key:
```bash
cat bruce_flows/.env
```

### Changes not reflecting
The source code is mounted, so changes are immediate. If you modified dependencies:
```bash
docker-compose build
```

## Architecture

The Docker setup:
- Uses NVIDIA CUDA base image
- Installs Python 3.11.14 and all dependencies
- Mounts your source code (so you can edit freely)
- Mounts results directory (so outputs persist)
- Provides GPU access to JAX for fast computation

## Questions?

Check the main [README.md](README.md) or open an issue on GitHub.
