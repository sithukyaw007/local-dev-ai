#!/bin/bash
# Start the mlx-lm server (OpenAI-compatible) with Qwen3.5-35B-A3B-4bit model.
# Usage: ./start_server.sh [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8000}"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Starting mlx-lm server..."
echo "  Model: $MODEL"
echo "  Port:  $PORT"
echo "  URL:   http://localhost:$PORT/v1"
echo ""

python -m mlx_lm server \
  --model "$MODEL" \
  --port "$PORT"
