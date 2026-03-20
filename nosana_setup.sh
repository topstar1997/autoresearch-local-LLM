#!/bin/bash
set -e

echo "=============================================="
echo "Autoresearch Local LLM — Nosana GPU Setup"
echo "=============================================="

# System dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    git curl python3 python3-pip python3-venv wget \
    build-essential

# Install ollama
echo "Installing ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start ollama server in background
echo "Starting ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 10

# Pull the LLM model
echo "Pulling Qwen 3.5 9B..."
ollama pull qwen3.5:9b

# Verify ollama is working
echo "Verifying ollama..."
ollama list

# Install uv (Python package manager)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace

# Install autoresearch dependencies
echo "Installing dependencies..."
uv sync

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Total VRAM: {props.total_mem / 1024**3:.1f} GB')
else:
    print('No CUDA GPU available!')
"

echo ""
echo "Setup complete! Starting pipeline..."
echo ""

# Run the pipeline
bash run_pipeline.sh
