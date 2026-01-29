# Docker Quick Reference for BRUCE

Common commands for daily use.

## First Time Setup

```bash
# Clone and enter directory
git clone https://github.com/simongoode/BRUCE.git
cd BRUCE

# Add your API key
echo "GOOGLE_API_KEY=your_key" > bruce_flows/.env

# Build and run
docker-compose up --build
```

---

## Daily Commands

### Run the analysis
```bash
docker-compose up
```

### Run in background (detached)
```bash
docker-compose up -d
```

### Stop everything
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs
docker-compose logs -f  # follow mode
```

---

## Development Commands

### Open a shell in the container
```bash
docker-compose run bruce bash
```

### Run a specific command
```bash
docker-compose run bruce python3 src/scripts/run_pe.py
docker-compose run bruce crewai run
```

### Run without starting dependent services
```bash
docker-compose run --rm bruce bash
```

---

## Rebuilding

### After changing dependencies
```bash
docker-compose build
docker-compose up
```

### Force rebuild (ignore cache)
```bash
docker-compose build --no-cache
```

### Rebuild and run
```bash
docker-compose up --build
```

---

## Editing Code

### Edit files normally on your machine
Your IDE/editor works as usual:
```bash
code bruce_flows/src/bruce_flows/main.py
vim bruce_flows/src/bruce_flows/crews/parameter_expert_crew/parameter_expert_crew.py
```

Changes are immediately available in the container (no rebuild needed).

### Re-run after editing
```bash
docker-compose run bruce crewai run
```

---

## Troubleshooting

### Check if container is running
```bash
docker ps
docker-compose ps
```

### View container logs
```bash
docker-compose logs bruce
```

### Restart container
```bash
docker-compose restart
```

### Remove containers and start fresh
```bash
docker-compose down
docker-compose up --build
```

### Check GPU inside container
```bash
docker-compose run bruce nvidia-smi
```

### Clear everything and start over
```bash
docker-compose down -v  # removes volumes too
docker system prune     # cleans up Docker
docker-compose up --build
```

---

## File Locations

### On your host machine
- **Source code**: `bruce_flows/src/bruce_flows/`
- **Results**: `bruce_flows/results/`
- **Config**: `bruce_flows/src/bruce_flows/crews/parameter_expert_crew/config/`
- **API key**: `bruce_flows/.env`

### Inside container
- **Working directory**: `/app/bruce_flows/`
- **Source**: `/app/bruce_flows/src/`
- **Results**: `/app/bruce_flows/results/` (mounted from host)

---

## Common Workflows

### Quick test after code change
```bash
# Edit code in your editor
# Then:
docker-compose run bruce crewai run
```

### Interactive debugging
```bash
docker-compose run bruce bash
# Inside container:
python3 src/scripts/run_pe.py
# or
ipython
```

### Check what's different from Git
```bash
git status
git diff
```

### Pull latest updates
```bash
git pull
docker-compose up --build  # rebuild if dependencies changed
```

---

## Help Commands

```bash
# Docker help
docker --help
docker-compose --help

# CrewAI help (inside container)
docker-compose run bruce crewai --help

# Python package info (inside container)
docker-compose run bruce pip list
docker-compose run bruce pip show jax
```

---

## Performance Tips

### First run is slow
- Building image: ~10 minutes
- Running PE: ~5-10 minutes (depends on GPU)

### Subsequent runs are faster
- Image is cached
- Only changed layers rebuild
- Results persist in `bruce_flows/results/`

### Improve build times
- Don't run `build --no-cache` unless necessary
- Keep `requirements.txt` stable
- Use `.dockerignore` to exclude large directories

---

## Safety Notes

- ✅ Code changes are immediate (mounted as volume)
- ✅ Results persist on your machine
- ✅ Container can be deleted/recreated safely
- ⚠️ Don't edit files inside `/app` directly in container (changes won't persist)
- ⚠️ Keep your `.env` file secure (contains API key)

---

See also:
- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Full setup guide
- [VERIFY_DOCKER.md](VERIFY_DOCKER.md) - Verify installation
- [DISTRIBUTION.md](DISTRIBUTION.md) - Share with coworkers
