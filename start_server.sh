#!/bin/bash
# Start the mlx-lm server (OpenAI-compatible) with Qwen3.5-35B-A3B-4bit model.
# Logs to file — use: tail -f logs/mlx-server.log
# Usage: ./start_server.sh [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/mlx-server.log"
PORT="${1:-8000}"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"

mkdir -p "$LOG_DIR"

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Starting mlx-lm server (DEBUG mode)..."
echo "  Model: $MODEL"
echo "  Port:  $PORT"
echo "  URL:   http://localhost:$PORT/v1"
echo "  Logs:  $LOG_FILE"
echo ""
echo "  To stream logs:  tail -f $LOG_FILE"
echo ""

python -m mlx_lm server \
  --model "$MODEL" \
  --port "$PORT" \
  --log-level DEBUG \
  >> "$LOG_FILE" 2>&1
