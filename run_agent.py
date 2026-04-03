#!/usr/bin/env python3
"""Interactive agent with tool calling via mlx-openai-server + OpenAI SDK.

Connects to a local mlx-openai-server, sends messages with tool definitions,
and executes tools (e.g., web search) when the model requests them.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import web_search

load_dotenv()

DEFAULT_PORT = 8000
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMP = 0.7
DEFAULT_SYSTEM = (
    "You are a helpful AI assistant with access to web search. "
    "When the user asks about current events, weather, news, or anything "
    "requiring up-to-date information, use the web_search tool. "
    "For general knowledge questions, answer directly."
)

# Registry of available tools: name → (schema, execute_fn)
TOOL_REGISTRY = {
    "web_search": (web_search.TOOL_SCHEMA, web_search.execute),
}


def get_tool_schemas() -> list[dict]:
    """Return OpenAI-compatible tool schemas for all registered tools."""
    return [schema for schema, _ in TOOL_REGISTRY.values()]


def execute_tool_call(name: str, arguments: dict) -> str:
    """Look up and execute a tool by name."""
    if name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{name}'"
    _, execute_fn = TOOL_REGISTRY[name]
    return execute_fn(**arguments)


def run_with_tools(client: OpenAI, model: str, messages: list[dict],
                   tools: list[dict], max_tokens: int, temperature: float) -> str:
    """Send a message and handle the tool-calling loop until a final answer."""
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        message = choice.message

        # If no tool calls, we have the final answer
        if not message.tool_calls:
            return message.content or ""

        # Append the assistant's tool-call message
        messages.append(message.model_dump())

        # Execute each tool call and append results
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"\n  🔧 Calling tool: {fn_name}({json.dumps(fn_args)})", flush=True)
            result = execute_tool_call(fn_name, fn_args)
            print(f"  ✅ Got result ({len(result)} chars)", flush=True)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


def main():
    parser = argparse.ArgumentParser(
        description="Agent chat with tool calling (web search) via local MLX server"
    )
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT,
                        help="mlx-openai-server port")
    parser.add_argument("-m", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Max tokens per response")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMP,
                        help="Sampling temperature")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="not-needed")

    # Verify server is reachable
    try:
        models = client.models.list()
        model_id = models.data[0].id if models.data else "default"
        print(f"Connected to server at {base_url}")
        print(f"Model: {model_id}")
    except Exception as e:
        print(f"Error: Cannot connect to server at {base_url}")
        print(f"Start the server first: ./start_server.sh")
        print(f"Details: {e}")
        sys.exit(1)

    tools = get_tool_schemas()
    messages = [{"role": "system", "content": args.system}]

    print(f"Tools available: {', '.join(TOOL_REGISTRY.keys())}")
    print("Type 'quit' or Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("\033[1;32mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("\033[1;34mAssistant:\033[0m ", end="", flush=True)

        try:
            answer = run_with_tools(
                client, model_id, messages, tools,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"\n{answer}\n")
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"\n{error_msg}\n")
            messages.pop()  # remove failed user message


if __name__ == "__main__":
    main()
