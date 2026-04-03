#!/bin/bash
# Start the mlx-lm server (OpenAI-compatible) with Qwen3.5-35B-A3B-4bit model.
# Logs to file — use: tail -f logs/mlx-server.log
# Usage: ./start_server.sh [port] [log-level]
#   log-level: DEBUG (default) | INFO | WARNING | ERROR
#   DEBUG shows: request payloads, model generation text, response bodies
#   INFO  shows: prompt processing progress, KV cache stats, HTTP status

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/mlx-server.log"
PORT="${1:-8000}"
LOG_LEVEL="${2:-DEBUG}"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"

mkdir -p "$LOG_DIR"

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Starting mlx-lm server..."
echo "  Model:     $MODEL"
echo "  Port:      $PORT"
echo "  Log level: $LOG_LEVEL"
echo "  URL:       http://localhost:$PORT/v1"
echo "  Logs:      $LOG_FILE"
echo ""
echo "  Stream logs: tail -f $LOG_FILE"
echo ""

python -m mlx_lm server \
  --model "$MODEL" \
  --port "$PORT" \
  --log-level "$LOG_LEVEL" \
  >> "$LOG_FILE" 2>&1
