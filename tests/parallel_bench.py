#!/usr/bin/env python3
"""Parallelism test suite: fires concurrent chat requests and measures
throughput, per-request latency, and total wall time.

Usage:
  python tests/parallel_bench.py --base http://127.0.0.1:8112 --concurrency 4
  python tests/parallel_bench.py --base http://127.0.0.1:8115 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def chat(base: str, messages: list[dict], max_tokens: int, timeout: int) -> dict:
    body = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    resp["_elapsed"] = elapsed
    return resp


PROMPTS = [
    "Explain why the sky is blue in exactly 3 sentences.",
    "Write a Python function to check if a number is prime. Output only code.",
    "List the 5 largest countries by area with their capitals.",
    "What is the difference between TCP and UDP? Be concise.",
    "Write a haiku about programming.",
    "Explain recursion to a 10-year-old in 2 sentences.",
    "What are the SOLID principles? One sentence each.",
    "Write a bash one-liner to find all .py files modified today.",
    "Explain the CAP theorem in simple terms.",
    "What is the time complexity of merge sort and why?",
    "Write a SQL query to find duplicate emails in a users table.",
    "Explain the difference between a stack and a queue.",
    "What is a closure in JavaScript? Give a short example.",
    "Write a regex to match email addresses.",
    "Explain how HTTPS works in 3 steps.",
    "What is the difference between git rebase and git merge?",
]


def run_batch(
    base: str, concurrency: int, prompts: list[str], max_tokens: int, timeout: int
) -> list[dict]:
    """Fire all prompts with given concurrency, return results."""
    results = []

    def do_one(idx: int, prompt: str) -> dict:
        resp = chat(base, [{"role": "user", "content": prompt}], max_tokens, timeout)
        t = resp.get("timings") or {}
        return {
            "idx": idx,
            "prompt": prompt[:50],
            "gen_tokens": t.get("predicted_n", 0),
            "prompt_tokens": t.get("prompt_n", 0),
            "gen_tok_s": t.get("predicted_per_second", 0),
            "prompt_tok_s": t.get("prompt_per_second", 0),
            "elapsed_s": round(resp["_elapsed"], 3),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(do_one, i, p): i for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r["idx"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Parallelism bench")
    parser.add_argument("--base", default="http://127.0.0.1:8112")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=4, help="Repeat prompt set N times")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    print(f"Target: {args.base}  concurrency={args.concurrency}  "
          f"max_tokens={args.max_tokens}  rounds={args.rounds}")
    print()

    # Warmup
    print("warmup...")
    chat(args.base, [{"role": "user", "content": "Reply ok"}], 4, args.timeout)

    # Build full prompt list (repeat prompts to fill ~1 min)
    all_prompts = (PROMPTS * args.rounds)[: 16 * args.rounds]

    print(f"Firing {len(all_prompts)} requests with concurrency {args.concurrency}...")
    print()

    t_start = time.perf_counter()
    results = run_batch(
        args.base, args.concurrency, all_prompts, args.max_tokens, args.timeout
    )
    t_total = time.perf_counter() - t_start

    # Stats
    total_gen = sum(r["gen_tokens"] for r in results)
    total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
    latencies = [r["elapsed_s"] for r in results]
    gen_rates = [r["gen_tok_s"] for r in results if r["gen_tok_s"] > 0]
    prompt_rates = [r["prompt_tok_s"] for r in results if r["prompt_tok_s"] > 0]

    print(f"{'idx':>3} {'elapsed':>7} {'gen_tok':>7} {'gen t/s':>7} {'prmpt t/s':>9}  prompt")
    print("-" * 80)
    for r in results:
        print(
            f"{r['idx']:>3} {r['elapsed_s']:>7.2f}s {r['gen_tokens']:>7} "
            f"{r['gen_tok_s']:>7.1f} {r['prompt_tok_s']:>9.1f}  {r['prompt']}"
        )

    print()
    print("=" * 80)
    print(f"Total requests:         {len(results)}")
    print(f"Total output tokens:    {total_gen}")
    print(f"Total prompt tokens:    {total_prompt}")
    print(f"Total wall time:        {t_total:.2f}s")
    print(f"Output throughput:      {total_gen / t_total:.1f} tok/s (aggregate output)")
    print(f"Prompt throughput:      {total_prompt / t_total:.1f} tok/s (aggregate input)")
    print(f"Requests/sec:           {len(results) / t_total:.2f}")
    print(f"Avg latency:            {sum(latencies) / len(latencies):.2f}s")
    print(f"Min/Max latency:        {min(latencies):.2f}s / {max(latencies):.2f}s")
    if gen_rates:
        print(f"Avg per-req decode:     {sum(gen_rates) / len(gen_rates):.1f} tok/s")
    if prompt_rates:
        print(f"Avg per-req prefill:    {sum(prompt_rates) / len(prompt_rates):.1f} tok/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
