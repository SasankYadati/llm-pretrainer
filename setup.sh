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

# Install dependencies
echo "Installing dependencies..."
uv sync

# flash-attn sometimes needs special handling
echo "Ensuring flash-attn is installed..."
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn already installed or installed via uv sync"

# Disable wandb by default (set WANDB_API_KEY to enable)
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set, disabling wandb..."
    export WANDB_MODE=disabled
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run training:"
echo "  uv run torchrun --nproc_per_node 1 train.py"
echo ""
echo "With custom parameters:"
echo "  uv run torchrun --nproc_per_node 1 train.py \\"
echo "      --seq_len 128 \\"
echo "      --micro_batch_size 4 \\"
echo "      --gradient_accumulation_steps 8 \\"
echo "      --max_tokens 1000000"
echo ""
