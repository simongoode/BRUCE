# Distributing BRUCE to Your Team

Guide for sharing BRUCE with coworkers who want to use the Docker setup.

## Option 1: GitHub Repository (Recommended)

The simplest method - coworkers clone from GitHub.

### What you share:
Just send the repository URL:
```
https://github.com/simongoode/BRUCE
```

### What coworkers do:
```bash
git clone https://github.com/simongoode/BRUCE.git
cd BRUCE
echo "GOOGLE_API_KEY=their_key" > bruce_flows/.env
docker-compose up --build
```

**Pros:**
- ✅ Always get latest version
- ✅ Easy to pull updates with `git pull`
- ✅ Single source of truth
- ✅ Minimal setup

**Cons:**
- ⚠️ Requires Git and GitHub access
- ⚠️ First build takes 10+ minutes (downloads dependencies)

---

## Option 2: Docker Hub / Container Registry

Share a pre-built Docker image (no build time for users).

### Setup (one-time):

1. **Build and tag the image:**
```bash
cd BRUCE
docker build -t yourusername/bruce:latest .
```

2. **Push to Docker Hub:**
```bash
docker login
docker push yourusername/bruce:latest
```

### What you share:
```
docker pull yourusername/bruce:latest
```

### What coworkers do:
```bash
# Pull the pre-built image
docker pull yourusername/bruce:latest

# Clone just the code (or share it separately)
git clone https://github.com/simongoode/BRUCE.git
cd BRUCE

# Add API key
echo "GOOGLE_API_KEY=their_key" > bruce_flows/.env

# Run using the pre-built image
docker run --gpus all \
  -v $(pwd)/bruce_flows/src:/app/bruce_flows/src \
  -v $(pwd)/bruce_flows/results:/app/bruce_flows/results \
  -v $(pwd)/bruce_flows/.env:/app/bruce_flows/.env \
  yourusername/bruce:latest
```

**Pros:**
- ✅ No build time (instant start)
- ✅ Consistent environment
- ✅ Can be private (Docker Hub private repos)

**Cons:**
- ⚠️ Large download (~3-4 GB)
- ⚠️ Need to push updates manually
- ⚠️ Requires Docker Hub account

---

## Option 3: Shared Drive / USB (Offline)

For environments without internet access or GitHub.

### Create the distributable:

1. **Save the Docker image:**
```bash
cd BRUCE
docker-compose build
docker save bruce:latest | gzip > bruce-docker-image.tar.gz
```

2. **Create a distribution package:**
```bash
# Create a zip with code and image
tar -czf BRUCE-distribution.tar.gz \
  --exclude=.git \
  --exclude=.venv \
  --exclude=results \
  --exclude=__pycache__ \
  BRUCE/ bruce-docker-image.tar.gz
```

### What you share:
- `BRUCE-distribution.tar.gz` (approx. 4-5 GB)

### What coworkers do:

```bash
# Extract everything
tar -xzf BRUCE-distribution.tar.gz

# Load the Docker image
docker load < bruce-docker-image.tar.gz

# Set up the code
cd BRUCE
echo "GOOGLE_API_KEY=their_key" > bruce_flows/.env

# Run
docker-compose up
```

**Pros:**
- ✅ Works offline
- ✅ Single package contains everything
- ✅ No GitHub/Docker Hub needed

**Cons:**
- ⚠️ Large file size (4-5 GB)
- ⚠️ Manual distribution
- ⚠️ Updates require re-distributing entire package

---

## Option 4: Internal GitLab/Gitea Server

If your organization has an internal Git server.

Same as Option 1, but push to your internal server:
```bash
git remote add internal https://your-internal-git.com/you/BRUCE.git
git push internal main
```

Share the internal URL with coworkers.

---

## What Coworkers Need (All Options)

Before they can use BRUCE, they need:

1. **Docker & Docker Compose** installed
2. **NVIDIA Docker runtime** (for GPU support)
3. **NVIDIA GPU** with CUDA support
4. **Google API key** from https://aistudio.google.com/app/apikey

Share these setup guides with them:
- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Complete setup instructions
- [VERIFY_DOCKER.md](VERIFY_DOCKER.md) - Verify their setup works

---

## Recommended Approach

For most teams: **Option 1 (GitHub) + documentation**

Send coworkers:
1. Repository URL
2. Link to [DOCKER_SETUP.md](DOCKER_SETUP.md)
3. Link to get API key: https://aistudio.google.com/app/apikey

They'll be up and running in ~15 minutes (plus build time).

---

## Updating the Distribution

When you push code updates:

**Option 1 (GitHub):** 
Coworkers just run `git pull && docker-compose up --build`

**Option 2 (Docker Hub):**
```bash
docker build -t yourusername/bruce:latest .
docker push yourusername/bruce:latest
```
Then notify coworkers to `docker pull`

**Option 3 (Offline):**
Create new distribution package and share again

---

## Security Notes

⚠️ **Never commit `.env` files** - they contain API keys  
⚠️ **Don't share your personal API key** - each user should get their own  
⚠️ **Check `.gitignore`** - ensure sensitive files aren't tracked  

The current `.dockerignore` and `.gitignore` are configured to exclude:
- `.env` files
- Virtual environments
- Results directory
- Cache files

---

## Questions?

- Docker issues: See [VERIFY_DOCKER.md](VERIFY_DOCKER.md)
- Usage questions: See [DOCKER_SETUP.md](DOCKER_SETUP.md)
- General setup: See [README.md](README.md)
