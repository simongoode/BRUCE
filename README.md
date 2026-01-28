# BRUCE

**B**ayesian **R**econstruction **U**sing **C**omputer intelli**E**ncE

A Python pipeline for automated Bayesian parameter estimation, specialized for gravitational wave data analysis using AI agents powered by CrewAI.

## Overview

BRUCE combines advanced gravitational wave parameter estimation with AI-powered analysis agents. The system uses CrewAI flows to orchestrate multiple rounds of parameter estimation and expert analysis, providing insights into mass and distance posteriors from gravitational wave signals.

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

## Installation

### Prerequisites

- Python 3.10 - 3.13
- Git
- CUDA-capable GPU (recommended for parameter estimation)

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
