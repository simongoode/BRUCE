# BRUCE

**B**ayesian **R**econstruction **U**sing **C**omputer intellig**E**ncE

A Python pipeline for automated Bayesian parameter estimation, specialized for gravitational wave data analysis using AI agents powered by CrewAI.

## Overview

BRUCE combines advanced gravitational wave parameter estimation with AI-powered analysis agents. The system uses CrewAI flows to orchestrate multiple rounds of parameter estimation and expert analysis, providing insights into mass and distance posteriors from gravitational wave signals.

## Quick Start with Docker 🐳

The easiest way to get started with BRUCE is using Docker. This provides a pre-configured environment with all dependencies including CUDA support.

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.x support
- Google API key ([get one here](https://aistudio.google.com/app/apikey))

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/simongoode/BRUCE.git
cd BRUCE
```

2. **Create your API key file:**
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > bruce_flows/.env
```

3. **Build and run:**
```bash
docker-compose up --build
```

That's it! The container will:
- Install all dependencies automatically
- Initialize git submodules
- Run the BRUCE analysis flow
- Save results to `bruce_flows/results/` on your host machine

### Development Workflow

The Docker setup mounts your source code, so you can edit files normally and re-run without rebuilding:

```bash
# Edit any Python file in bruce_flows/src/
# Then run again (no rebuild needed):
docker-compose run bruce crewai run

# Run a shell inside the container:
docker-compose run bruce bash

# Stop and remove containers:
docker-compose down

# Rebuild after dependency changes:
docker-compose build
```

## Project Structure

```
BRUCE/
├── bruce_flows/              # Main CrewAI project directory
│   ├── src/
│   │   ├── bruce_flows/      # CrewAI agents and flows
│   │   │   ├── crews/        # Agent crews (parameter experts)
│   │   │   ├── tools/        # Custom tools
│   │   │   └── main.py       # Main flow definition
│   │   └── scripts/
│   │       └── run_pe.py     # Parameter estimation script
│   ├── blackjax_ns_gw/       # BlackJAX nested sampling library
│   ├── results/              # Output files and reports
│   ├── pyproject.toml        # CrewAI project config
│   └── .env                  # API keys (not tracked)
├── archive/                  # Archived scripts and utilities
└── requirements.txt          # Python dependencies
```

## Manual Installation

If you prefer not to use Docker, you can install BRUCE manually.

### Prerequisites

- Python 3.10 - 3.13 (3.11.14 recommended)
- Git
- CUDA-capable GPU (recommended for parameter estimation)
- CUDA Toolkit 12.x

### Step 1: Clone the Repository

```bash
git clone https://github.com/simongoode/BRUCE.git
cd BRUCE
```

### Step 2: Create Virtual Environment

Create a virtual environment in the `bruce_flows/` directory:

```bash
cd bruce_flows
python -m venv .venv
```

Activate the virtual environment:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required packages using `uv` (or `pip`):

```bash
# Navigate back to project root
cd ..

# Install with uv (recommended)
uv pip install -r requirements.txt

# Or use standard pip
pip install -r requirements.txt
```

**Note:** This will install:
- JAX with CUDA support for GPU acceleration
- CrewAI with Google Gemini integration
- Scientific computing packages (NumPy 2.0+, SciPy, Astropy 7.0+)
- Gravitational wave analysis tools (jimgw, bilby)
- BlackJAX nested sampling library

### Step 4: Configure API Keys

Create a `.env` file in the `bruce_flows/` directory with your Google API key:

```bash
cd bruce_flows
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

**To obtain a Google API key:**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key into your `.env` file

### Step 5: Configure AI Model (Optional)

By default, BRUCE uses Google's Gemini model. If you want to use a different LLM provider (OpenAI, Anthropic, etc.), you'll need to:

1. Update the model configuration in:
   ```
   bruce_flows/src/bruce_flows/crews/parameter_expert_crew/parameter_expert_crew.py
   ```

2. Add the corresponding API key to your `.env` file

3. Install the appropriate CrewAI extras if needed

## Usage

### Running the Analysis Flow

Navigate to the `bruce_flows/` directory and run the CrewAI flow:

```bash
cd bruce_flows
crewai run
```

This will:
1. Execute the parameter estimation script (`run_pe.py`)
2. Generate a PE report at `results/bruce_pe_report.md`
3. Launch AI agents to analyze mass and distance posteriors
4. Run multiple rounds of analysis (default: 3 rounds)
5. Generate expert analysis reports in the `results/` directory

### Output Files

After execution, you'll find:

- `results/bruce_pe_report.md` - Parameter estimation summary with posteriors
- `results/mass-expert-report-round-*.txt` - Mass parameter analysis by round
- `results/distance-expert-report-round-*.txt` - Distance parameter analysis by round

## Workflow Details

The BRUCE flow consists of:

1. **Parameter Estimation**: Runs nested sampling using BlackJAX to estimate gravitational wave parameters
2. **Mass Expert Analysis**: AI agent analyzes mass-related posteriors (chirp mass, mass ratio, component masses)
3. **Distance Expert Analysis**: AI agent analyzes distance-related posteriors (luminosity distance, inclination)
4. **Iterative Refinement**: Multiple rounds of analysis build on previous insights

## Troubleshooting

### Docker Issues

#### GPU not detected
If you see errors about CUDA or GPU not being available:

1. Verify NVIDIA Docker runtime is installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

2. If the above fails, install NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Permission denied errors
If you get permission errors with Docker:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### Container exits immediately
Check your `.env` file exists and contains a valid Google API key:
```bash
cat bruce_flows/.env
```

### NumPy Version Issues

If you encounter `AttributeError: module 'numpy' has no attribute 'in1d'`, ensure you have NumPy 2.0+ and Astropy 7.0+ installed:

```bash
pip install --upgrade "numpy>=2.0.0" "astropy>=7.0.0" "scipy>=1.15.0"
```

### Module Import Errors

If you see `ModuleNotFoundError: No module named 'blackjax_ns_gw'`, ensure:
1. You're running `crewai run` from the `bruce_flows/` directory
2. The git submodule was properly cloned: `git submodule update --init --recursive`

### GPU/CUDA Issues

For CUDA-related errors, verify:
- CUDA toolkit is installed (version 12.x recommended)
- JAX CUDA installation: `pip install --upgrade "jax[cuda12_pip]>=0.4.31"`

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add your license information here]

## Citation

If you use BRUCE in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

- GitHub: [simongoode/BRUCE](https://github.com/simongoode/BRUCE)
- [Add your contact information here]

## Acknowledgments

- Built with [CrewAI](https://www.crewai.com/)
- Uses [jimgw](https://github.com/kazewong/jim) for gravitational wave analysis
- Powered by [BlackJAX](https://github.com/blackjax-devs/blackjax) nested sampling
