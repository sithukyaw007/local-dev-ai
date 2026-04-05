# Local Mac AI — Run LLMs on Apple Silicon via MLX

Run large language models locally on Apple Silicon via [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) for maximum performance. Includes an **agent with web search** via tool calling.

## Supported Models

| Model | ID | Architecture | Active Params | RAM Usage | Context |
|-------|-----|-------------|---------------|-----------|---------|
| **Qwen3.5-35B-A3B** (default) | `mlx-community/Qwen3.5-35B-A3B-4bit` | MoE — 35B total | 3B | ~20–22 GB | 262K |
| **Gemma 4 26B-A4B** | `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit` | MoE — 26B total | 3.8B | ~15–18 GB | 256K |

Both models are 4-bit quantized, Apache 2.0 licensed, and support thinking/reasoning mode.

## Prerequisites

- macOS 13.0+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~22 GB free disk space for model weights (cached in `~/.cache/huggingface/`)
- [Tavily API key](https://tavily.com) (free tier — 1,000 searches/month) for web search

## Setup

### Default Setup (Qwen3.5)

```bash
cd /path/to/local-mac-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment (required for web search agent)
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY
```

### Gemma 4 Setup (Optional)

Gemma 4 requires a separate venv with patched `mlx-lm` (upstream MLX support is pending):

```bash
# One-command install via Unsloth (creates ~/.unsloth/unsloth_gemma4_mlx/)
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/install_gemma4_mlx.sh | sh

# Verify it works
source ~/.unsloth/unsloth_gemma4_mlx/bin/activate
python -m mlx_lm chat --model unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit --max-tokens 200
```

## Usage

### 1. Single Prompt Generation (Direct, No Server)

```bash
# Default model (Qwen3.5)
python run_generate.py --prompt "What is quantum computing?"
python run_generate.py --prompt "Write a Python quicksort" --max-tokens 1024 --temperature 0.3

# Gemma 4 (activate Unsloth venv first)
source ~/.unsloth/unsloth_gemma4_mlx/bin/activate
python run_generate.py --model unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit --prompt "What is quantum computing?"
```

### 2. Interactive Chat (Direct, No Server)

```bash
# Default model (Qwen3.5)
python run_chat.py
python run_chat.py --temperature 0.3 --max-tokens 2048
python run_chat.py --system "You are a senior Python developer. Be concise."

# Gemma 4
source ~/.unsloth/unsloth_gemma4_mlx/bin/activate
python run_chat.py --model unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit
```

### 3. Agent with Web Search (Server + Tool Calling)

This mode gives the model access to real-time web search via Tavily.

**Step 1: Start the server** (in a separate terminal):
```bash
# Default model (Qwen3.5)
./start_server.sh              # default: port 8000, DEBUG logging
./start_server.sh 8000 INFO    # INFO logging (less verbose)

# Gemma 4 — pass model as 3rd argument (auto-activates Unsloth venv)
./start_server.sh 8000 DEBUG unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit
```

**Step 2: Stream the logs** (in another terminal):
```bash
tail -f logs/mlx-server.log
```

**Step 3: Run the agent** (in another terminal):
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
- `-m / --max-tokens` — max tokens to generate (default: 8192 / 16384)
- `-t / --temperature` — sampling temperature (default: 0.7)
- `--top-p` — nucleus sampling threshold (default: 0.9)
- `--system` — system prompt (chat only)
- `--model` — override HuggingFace model ID
- `--no-think` — disable thinking mode (see below)

**run_agent.py** (server-based, tool calling):
- `-p / --port` — server port (default: 8000)
- `-m / --max-tokens` — max tokens per response (default: 16384)
- `-t / --temperature` — sampling temperature (default: 0.7)
- `--system` — system prompt
- `--no-think` — disable thinking mode (see below)

## Thinking Mode

Both Qwen3.5 and Gemma 4 have a **thinking mode** (enabled by default) that generates internal reasoning before producing the visible answer. This improves quality on complex reasoning/coding/math tasks but consumes extra tokens.

| Model | Thinking Tags | Overhead |
|-------|--------------|----------|
| Qwen3.5 | `<think>...</think>` | ~700–750 tokens |
| Gemma 4 | `<\|channel>thought...<channel\|>` | Varies |

```bash
# Default: thinking enabled (better reasoning, uses more tokens)
python run_generate.py --prompt "Solve this step by step: what is 23 * 47?"

# Disable thinking: faster, more concise (good for simple questions)
python run_generate.py --no-think --prompt "What is Python?"
python run_chat.py --no-think
python run_agent.py --no-think
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ Direct Mode (run_chat.py / run_generate.py)          │
│   User → mlx-lm (in-process) → Response             │
│   Supports: --model <any MLX model ID>               │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Agent Mode (run_agent.py)                            │
│   User → OpenAI SDK → mlx-openai-server (localhost)  │
│     → Model decides: answer directly OR tool_call    │
│     → If tool_call: execute Tavily search            │
│     → Feed search results back to model              │
│     → Final answer with real-time data               │
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

## Server Logging

The server logs to `logs/mlx-server.log`. Stream with `tail -f logs/mlx-server.log`.

```bash
# DEBUG (default) — full request/response payloads, token-by-token output
./start_server.sh 8000 DEBUG

# INFO — prompt processing progress, KV cache stats, HTTP status codes only
./start_server.sh 8000 INFO
```

| Level | What's Logged |
|-------|---------------|
| **DEBUG** | Incoming request body (messages, params), each generated token, full outgoing response JSON |
| **INFO** | Prompt processing progress, KV cache size, HTTP request/response status |
| **WARNING** | Unusual conditions (e.g., missing stop tokens) |
| **ERROR** | Request failures, JSON parse errors |

## Adding More Tools

Create a new file in `tools/` following the pattern in `tools/web_search.py`:
1. Define `TOOL_SCHEMA` (OpenAI function calling format)
2. Define `execute(**kwargs)` function
3. Register it in `run_agent.py`'s `TOOL_REGISTRY`

## Upgrading Later

- **Q8 quantization**: `--model mlx-community/Qwen3.5-35B-A3B-8bit` (better quality, ~35 GB RAM)
- **Multimodal (vision)**: `pip install mlx-vlm` and use `mlx-community/Qwen3.5-35B-A3B-4bit` with image prompts
- **When mlx-lm adds native Gemma 4 support**: You can switch to the standard `.venv` for Gemma 4 too (no separate Unsloth venv needed). Check [mlx-lm releases](https://github.com/ml-explore/mlx-examples/releases) for updates.
