#!/usr/bin/env python3
"""Interactive chat with Qwen3.5-35B-A3B via MLX-LM with streaming output."""

import argparse
import sys

from mlx_lm import load, stream_generate
from mlx_lm.generate import make_sampler


MODEL_ID = "mlx-community/Qwen3.5-35B-A3B-4bit"

DEFAULT_MAX_TOKENS = 16384
DEFAULT_TEMP = 0.7
DEFAULT_SYSTEM = "You are Qwen, a helpful AI assistant. Be concise and accurate."


def build_prompt(tokenizer, messages: list[dict], enable_thinking: bool = True) -> str:
    """Apply the model's chat template to format the conversation."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def main():
    parser = argparse.ArgumentParser(description="Chat with Qwen3.5-35B-A3B (MLX)")
    parser.add_argument("-m", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens per response")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMP, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode (faster, fewer tokens, but less reasoning)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    think_status = "OFF" if args.no_think else "ON"
    print(f"Model loaded! Thinking: {think_status}. Type 'quit' or Ctrl+C to exit.\n")

    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    messages = [{"role": "system", "content": args.system}]

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
        prompt = build_prompt(tokenizer, messages, enable_thinking=not args.no_think)

        print("\033[1;34mAssistant:\033[0m ", end="", flush=True)

        response_tokens = []
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        ):
            token_text = chunk.text
            print(token_text, end="", flush=True)
            response_tokens.append(token_text)

        full_response = "".join(response_tokens)
        messages.append({"role": "assistant", "content": full_response})
        print("\n")


if __name__ == "__main__":
    main()
