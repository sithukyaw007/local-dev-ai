#!/usr/bin/env python3
"""Single-prompt text generation using Qwen3.5-35B-A3B via MLX-LM."""

import argparse
import time

from mlx_lm import load, generate
from mlx_lm.generate import make_sampler


MODEL_ID = "mlx-community/Qwen3.5-35B-A3B-4bit"

DEFAULT_PROMPT = "Explain the Mixture-of-Experts architecture in 3 sentences."
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMP = 0.7


def main():
    parser = argparse.ArgumentParser(description="Generate text with Qwen3.5-35B-A3B (MLX)")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="Input prompt")
    parser.add_argument("-m", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMP, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    print(f"Prompt: {args.prompt}\n")
    print("=" * 60)

    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=True,
    )
    gen_time = time.perf_counter() - t0

    print("=" * 60)
    print(f"\nGeneration completed in {gen_time:.1f}s")


if __name__ == "__main__":
    main()
