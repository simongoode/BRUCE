# Docker Setup Verification

Quick checks to verify your Docker setup is working correctly before running BRUCE.

## Step 1: Verify Docker Installation

```bash
docker --version
docker-compose --version
```

Expected: Both commands should show version numbers.

## Step 2: Verify NVIDIA Docker Runtime

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

Expected: Should show your GPU information (model, memory, CUDA version).

**If this fails:**
- Your NVIDIA drivers might not be installed
- NVIDIA Container Toolkit might not be installed
- See DOCKER_SETUP.md for installation instructions

## Step 3: Verify API Key Setup

```bash
cat bruce_flows/.env
```

Expected: Should show `GOOGLE_API_KEY=your_actual_key`

**If missing:**
```bash
echo "GOOGLE_API_KEY=your_key_here" > bruce_flows/.env
```

## Step 4: Build the Docker Image

```bash
docker-compose build
```

Expected: 
- Takes 5-10 minutes on first build
- Should complete without errors
- Final message: "Successfully tagged bruce:latest"

**Common issues:**
- Network errors: Check your internet connection
- Permission denied: Run `sudo usermod -aG docker $USER && newgrp docker`

## Step 5: Test Container Access

```bash
docker-compose run bruce bash
```

Expected: You should get a bash prompt inside the container.

**Inside the container, test:**

```bash
# Check Python version
python3 --version
# Should show: Python 3.11.x

# Check JAX can see GPU
python3 -c "import jax; print(jax.devices())"
# Should show: [cuda(id=0)] or similar

# Check blackjax submodule
ls -la blackjax_ns_gw/src/
# Should show Python files

# Check CrewAI
crewai --version
# Should show CrewAI version

# Exit container
exit
```

## Step 6: Run a Quick Test

```bash
docker-compose run bruce python3 -c "import jax; import crewai; print('✓ All imports successful')"
```

Expected: Should print "✓ All imports successful"

## Step 7: Full Pipeline Test

```bash
docker-compose up
```

Expected:
- Container starts
- Runs parameter estimation (takes several minutes)
- Creates files in `bruce_flows/results/`
- Shows progress messages

**Stop with:** Ctrl+C, then `docker-compose down`

## Troubleshooting

### GPU Memory Issues
If you see "out of memory" errors:
```bash
# Clear GPU memory
nvidia-smi
# Find processes using GPU, kill if necessary
```

### Import Errors
If Python packages are missing:
```bash
# Rebuild the image
docker-compose build --no-cache
```

### Container Exits Immediately
Check logs:
```bash
docker-compose logs
```

### Code Changes Not Reflected
Code is mounted as volume, so changes should be immediate. If not:
```bash
# Restart container
docker-compose restart
```

## Success Criteria

✅ All commands above complete without errors  
✅ GPU is detected by nvidia-smi  
✅ JAX can see CUDA devices  
✅ Results appear in `bruce_flows/results/`  

If all checks pass, your Docker setup is ready! 🎉

## Next Steps

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for daily usage patterns.
