# BRUCE - Bayesian Reconstruction Using Computer intelligencE
# Docker container for gravitational wave parameter estimation with GPU support

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# (includes JAX with CUDA, scientific packages)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (need source code for editable install)
COPY . .

# If submodules aren't present, clone them manually
# The blackjax_ns_gw submodule should be in bruce_flows/
RUN if [ ! -d "bruce_flows/blackjax_ns_gw/src" ]; then \
        cd bruce_flows && \
        git clone https://github.com/mrosep/blackjax_ns_gw.git blackjax_ns_gw; \
    fi

# Install the bruce_flows package in editable mode
# (must be after source code is copied)
RUN pip install --no-cache-dir -e bruce_flows/

# Set PYTHONPATH to ensure all packages are findable in subprocesses
ENV PYTHONPATH="/app/bruce_flows/src${PYTHONPATH:+:${PYTHONPATH}}"

# Set the working directory to bruce_flows for running crewai commands
WORKDIR /app/bruce_flows

# Default command - can be overridden
CMD ["crewai", "run"]
