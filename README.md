# Adaptiq Image Generation Agent Benchmark

A comprehensive benchmark for evaluating and optimizing image generation agents using reinforcement learning and adaptive optimization techniques.

## Overview

The Adaptiq Image Generation Agent Benchmark is part of the [Adaptiq framework](https://github.com/adaptiq-ai/adaptiq) - an adaptive optimization system that uses reinforcement learning to improve AI agent performance while reducing costs. This benchmark specifically focuses on evaluating image generation agents across various metrics including quality, prompt adherence, efficiency, and resource utilization.

## Quick Start

### Prerequisites

- Python 3.11+
- Credits required to run the test (~$8 minimum)
- Required dependencies (see `requirements.txt`)

### Installation

First, install `uv` if you haven't already:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pip install uv
```

Clone the repository and set up the environment:
```bash
git clone https://github.com/adaptiq-ai/adaptiq-benchmark.git
cd image_gen_agent
```

Create and activate virtual environment using `uv`:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Configuration

Set your API keys in the `.env` file:
```
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-ai-key  
REPLICATE_API_KEY=your-replicate-key
```

### Running the Tests

Run the benchmark test:
```bash
python flow_test.py
```

Generate PDF report:
```bash
python generate_report.py
```

### Additional Information

- **Prompts Configuration**: Inspect the prompts used in `./config/prompts.json`
- **Test Images**: Original images for testing are located in `./images` with all sources from Pinterest
- **Example Test Run**: See an example of a complete test run in `./test_results/20250813_100138`

---

*Part of the Adaptiq ecosystem for adaptive AI agent optimization*
