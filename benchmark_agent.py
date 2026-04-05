#!/usr/bin/env python3
"""Benchmark script: compare models in Agent Mode (server + tool calling).

Runs identical prompts against a running mlx-lm server and measures:
  - Response latency (time to first token, total time)
  - Token counts (prompt, completion)
  - Tool calling behavior (did it call tools? which ones?)
  - Response content for quality comparison

Usage:
    # Start server first, then:
    python benchmark_agent.py --port 8001 --label "Gemma 4"
    python benchmark_agent.py --port 8000 --label "Qwen 3.5"
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Import tools from the project
sys.path.insert(0, os.path.dirname(__file__))
from tools import web_search

TOOL_REGISTRY = {
    "web_search": (web_search.TOOL_SCHEMA, web_search.execute),
}

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to web search. "
    "When the user asks about current events, weather, news, or anything "
    "requiring up-to-date information, use the web_search tool. "
    "For general knowledge questions, answer directly. "
    "Be concise — keep answers under 100 words."
)

# Test prompts: (category, prompt, expects_tool_call)
TEST_PROMPTS = [
    ("general_knowledge", "What is the capital of Japan?", False),
    ("reasoning", "If I have 3 boxes with 7 apples each, and I eat 4 apples, how many are left? Think step by step.", False),
    ("coding", "Write a Python function to check if a string is a palindrome. Keep it short.", False),
    ("tool_calling", "What is the latest news about Apple today?", True),
    ("tool_calling", "What is the current weather in Tokyo?", True),
]


def get_tool_schemas():
    return [schema for schema, _ in TOOL_REGISTRY.values()]


def execute_tool_call(name, arguments):
    if name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{name}'"
    _, execute_fn = TOOL_REGISTRY[name]
    return execute_fn(**arguments)


def run_single_prompt(client, model_id, tools, prompt, system, max_tokens=2048, temperature=0.7):
    """Run a single prompt through the agent loop, return metrics."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    tool_calls_made = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    rounds = 0

    t_start = time.perf_counter()

    while True:
        rounds += 1
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools if tools else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        message = choice.message
        usage = response.usage

        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens

        if not message.tool_calls:
            t_end = time.perf_counter()
            return {
                "answer": message.content or "",
                "tool_calls": tool_calls_made,
                "rounds": rounds,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_time": t_end - t_start,
                "finish_reason": choice.finish_reason,
            }

        messages.append(message.model_dump())

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            tool_calls_made.append({"name": fn_name, "args": fn_args})

            result = execute_tool_call(fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if rounds > 5:
            t_end = time.perf_counter()
            return {
                "answer": "(max rounds exceeded)",
                "tool_calls": tool_calls_made,
                "rounds": rounds,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_time": t_end - t_start,
                "finish_reason": "max_rounds",
            }


def main():
    parser = argparse.ArgumentParser(description="Benchmark agent mode")
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("-l", "--label", default="Model")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="not-needed")

    try:
        models = client.models.list()
        model_id = models.data[0].id if models.data else "default"
    except Exception as e:
        print(f"❌ Cannot connect to server at {base_url}: {e}")
        sys.exit(1)

    tools = get_tool_schemas()

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: {args.label}")
    print(f"  Server: {base_url}  |  Model: {model_id}")
    print(f"  Thinking: {'OFF' if args.no_think else 'ON'}")
    print(f"{'=' * 70}\n")

    results = []

    for i, (category, prompt, expects_tool) in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] ({category}) {prompt[:60]}...")
        try:
            r = run_single_prompt(client, model_id, tools, prompt, SYSTEM_PROMPT)
            results.append({**r, "category": category, "prompt": prompt, "expects_tool": expects_tool})

            tool_str = ", ".join(tc["name"] for tc in r["tool_calls"]) if r["tool_calls"] else "none"
            tok_per_sec = r["completion_tokens"] / r["total_time"] if r["total_time"] > 0 else 0
            print(f"  ⏱  {r['total_time']:.1f}s | {r['completion_tokens']} tokens | {tok_per_sec:.1f} tok/s | tools: {tool_str}")

            # Truncate answer for display
            answer_preview = r["answer"].replace("\n", " ")[:120]
            print(f"  💬 {answer_preview}...")
            print()
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            results.append({
                "category": category, "prompt": prompt, "expects_tool": expects_tool,
                "answer": f"ERROR: {e}", "tool_calls": [], "rounds": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_time": 0,
                "finish_reason": "error",
            })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {args.label} ({model_id})")
    print(f"{'=' * 70}")

    total_time = sum(r["total_time"] for r in results)
    total_comp_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    avg_tok_per_sec = total_comp_tokens / total_time if total_time > 0 else 0

    tool_expected = [r for r in results if r["expects_tool"]]
    tool_correct = sum(1 for r in tool_expected if len(r["tool_calls"]) > 0)
    no_tool_expected = [r for r in results if not r["expects_tool"]]
    no_tool_correct = sum(1 for r in no_tool_expected if len(r["tool_calls"]) == 0)

    print(f"  Total time:          {total_time:.1f}s")
    print(f"  Avg time/prompt:     {total_time / len(results):.1f}s")
    print(f"  Total tokens:        {total_prompt_tokens} prompt + {total_comp_tokens} completion")
    print(f"  Avg tok/s:           {avg_tok_per_sec:.1f}")
    print(f"  Tool call accuracy:  {tool_correct}/{len(tool_expected)} called when expected, "
          f"{no_tool_correct}/{len(no_tool_expected)} skipped when not needed")
    print()

    # Per-prompt table
    print(f"  {'#':<3} {'Category':<18} {'Time':>6} {'Tokens':>7} {'Tok/s':>7} {'Tools':>8} {'Rounds':>7}")
    print(f"  {'-'*3} {'-'*18} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
    for i, r in enumerate(results, 1):
        tok_s = r["completion_tokens"] / r["total_time"] if r["total_time"] > 0 else 0
        tool_str = "✅" if r["tool_calls"] else "—"
        print(f"  {i:<3} {r['category']:<18} {r['total_time']:>5.1f}s {r['completion_tokens']:>7} {tok_s:>6.1f} {tool_str:>8} {r['rounds']:>7}")

    print(f"\n{'=' * 70}\n")

    # Save full results to JSON
    output_file = f"logs/benchmark_{args.label.lower().replace(' ', '_')}.json"
    os.makedirs("logs", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"label": args.label, "model": model_id, "results": results}, f, indent=2)
    print(f"  Full results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
