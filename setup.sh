#!/bin/bash
set -e

echo "=== LLM Pretrainer Setup for RunPod ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure uv is in PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Create venv and install torch first (flash-attn needs it for building)
echo "Installing torch first..."
uv venv
uv pip install torch

# Install flash-attn build dependencies (needed if building from source)
echo "Installing flash-attn build dependencies..."
uv pip install psutil numpy ninja packaging

# Now install flash-attn with torch available
echo "Installing flash-attn (this may take several minutes if building from source)..."
uv pip install flash-attn --no-build-isolation

# Install remaining dependencies
echo "Installing remaining dependencies..."
uv sync

# Disable wandb by default (set WANDB_API_KEY to enable)
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set, disabling wandb..."
    export WANDB_MODE=disabled
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run training:"
echo "  uv run train.py"
echo ""
echo "With custom parameters:"
echo "  uv run train.py \\"
echo "      --seq_len 128 \\"
echo "      --micro_batch_size 4 \\"
echo "      --gradient_accumulation_steps 8 \\"
echo "      --max_tokens 1000000"
echo ""
