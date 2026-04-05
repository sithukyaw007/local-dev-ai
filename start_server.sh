#!/bin/bash
# Start the mlx-lm server (OpenAI-compatible) with a local model.
# Logs to file — use: tail -f logs/mlx-server.log
# Usage: ./start_server.sh [port] [log-level] [model]
#   port:      default 8000
#   log-level: DEBUG (default) | INFO | WARNING | ERROR
#   model:     HuggingFace model ID (default: mlx-community/Qwen3.5-35B-A3B-4bit)
#
# For Gemma 4 (requires Unsloth venv):
#   ./start_server.sh 8000 DEBUG unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/mlx-server.log"
PORT="${1:-8000}"
LOG_LEVEL="${2:-DEBUG}"
MODEL="${3:-mlx-community/Qwen3.5-35B-A3B-4bit}"

GEMMA4_VENV="$HOME/.unsloth/unsloth_gemma4_mlx"

mkdir -p "$LOG_DIR"

# Activate the appropriate virtual environment
if [[ "$MODEL" == *gemma-4* ]] && [ -d "$GEMMA4_VENV" ]; then
    echo "Detected Gemma 4 model — using Unsloth venv"
    source "$GEMMA4_VENV/bin/activate"
else
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

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
