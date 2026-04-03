# Local Mac AI — Qwen3.5-35B-A3B on MLX

Run **Qwen3.5-35B-A3B** (4-bit quantized) locally on Apple Silicon via [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) for maximum performance. Includes an **agent with web search** via tool calling.

## Model Info

| Property | Value |
|----------|-------|
| Model | [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) |
| Architecture | Mixture-of-Experts (MoE) — 35B total, 3B active per token |
| Quantization | 4-bit (~20 GB download) |
| Context Window | 262,144 tokens |
| RAM Usage | ~20–22 GB at inference |
| License | Apache 2.0 |

## Prerequisites

- macOS 13.0+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~22 GB free disk space for model weights (cached in `~/.cache/huggingface/`)
- [Tavily API key](https://tavily.com) (free tier — 1,000 searches/month) for web search

## Setup

```bash
# Clone/navigate to project
cd /path/to/local-mac-ai

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (required for web search agent)
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY
```

## Usage

### 1. Single Prompt Generation (Direct, No Server)

```bash
python run_generate.py --prompt "What is quantum computing?"
python run_generate.py --prompt "Write a Python quicksort" --max-tokens 1024 --temperature 0.3
```

### 2. Interactive Chat (Direct, No Server)

```bash
python run_chat.py
python run_chat.py --temperature 0.3 --max-tokens 2048
python run_chat.py --system "You are a senior Python developer. Be concise."
```

### 3. Agent with Web Search (Server + Tool Calling)

This mode gives the model access to real-time web search via Tavily.

**Step 1: Start the server** (in a separate terminal):
```bash
./start_server.sh          # default port 8000
./start_server.sh 9000     # custom port
```

**Step 2: Run the agent** (in another terminal):
```bash
source .venv/bin/activate
python run_agent.py

# Example questions the agent can search for:
#   "What's the current weather in Singapore?"
#   "What happened in tech news today?"
#   "Who won the latest Champions League match?"
```

The agent automatically decides when to search the web vs. answer from knowledge.

### CLI Options

**run_generate.py / run_chat.py** (direct MLX-LM):
- `-p / --prompt` — input prompt (generate only)
- `-m / --max-tokens` — max tokens to generate (default: 512 / 1024)
- `-t / --temperature` — sampling temperature (default: 0.7)
- `--top-p` — nucleus sampling threshold (default: 0.9)
- `--system` — system prompt (chat only)
- `--model` — override HuggingFace model ID

**run_agent.py** (server-based, tool calling):
- `-p / --port` — server port (default: 8000)
- `-m / --max-tokens` — max tokens per response (default: 1024)
- `-t / --temperature` — sampling temperature (default: 0.7)
- `--system` — system prompt

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ Direct Mode (run_chat.py / run_generate.py)          │
│   User → mlx-lm (in-process) → Response             │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Agent Mode (run_agent.py)                            │
│   User → OpenAI SDK → mlx-openai-server (localhost)  │
│     → Qwen3.5 decides: answer directly OR tool_call  │
│     → If tool_call: execute Tavily search             │
│     → Feed search results back to model               │
│     → Final answer with real-time data                │
└──────────────────────────────────────────────────────┘
```

## First Run

The first run will download the model (~20 GB) from HuggingFace. Subsequent runs load from cache.

```bash
# Quick test (direct mode)
python run_generate.py
```

## Performance Notes

On M4 Max (64 GB):
- Model load: ~5–10s (first load), ~2–3s (cached)
- Generation: ~134 tokens/sec
- RAM usage: ~19.6 GB (leaves ~44 GB free for other tools)

## Adding More Tools

Create a new file in `tools/` following the pattern in `tools/web_search.py`:
1. Define `TOOL_SCHEMA` (OpenAI function calling format)
2. Define `execute(**kwargs)` function
3. Register it in `run_agent.py`'s `TOOL_REGISTRY`

## Upgrading Later

- **Q8 quantization**: `--model mlx-community/Qwen3.5-35B-A3B-8bit` (better quality, ~35 GB RAM)
- **Multimodal (vision)**: `pip install mlx-vlm` and use `mlx-community/Qwen3.5-35B-A3B-4bit` with image prompts
