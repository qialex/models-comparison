#!/usr/bin/env python3
"""Latency bench for FLUX.2-klein (images/sec, not tok/s).

  python tests/flux_latency.py --base http://127.0.0.1:8116
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post_json(url: str, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


PROMPTS = [
    "A red apple on a wooden table, soft daylight, photo",
    "A cat holding a sign that says hello world",
    "Minimal product shot of a ceramic mug, studio lighting",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8116")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--warmup", action="store_true", default=True)
    args = p.parse_args()

    try:
        health = get_json(f"{args.base}/health", timeout=30)
    except Exception as e:
        print(f"health failed: {e}", file=sys.stderr)
        return 1
    print("health:", json.dumps(health, indent=2))

    if args.warmup:
        print("warmup...")
        post_json(
            f"{args.base}/generate",
            {
                "prompt": "warmup square",
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "seed": 0,
            },
            args.timeout,
        )

    rows = []
    wall0 = time.perf_counter()
    for i, prompt in enumerate(PROMPTS):
        out = post_json(
            f"{args.base}/generate",
            {
                "prompt": prompt,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "seed": i,
            },
            args.timeout,
        )
        # drop huge base64 from summary
        row = {k: v for k, v in out.items() if k != "image_base64"}
        rows.append(row)
        print(
            f"[{i}] {row['latency_s']}s  {row.get('images_per_sec')} img/s  "
            f"vram_max={row.get('vram_max_allocated_mb')}MB  {prompt[:48]!r}"
        )
    wall = time.perf_counter() - wall0

    latencies = [r["latency_s"] for r in rows]
    avg = sum(latencies) / len(latencies)
    print()
    print("=" * 60)
    print(f"images:           {len(rows)}")
    print(f"size:             {args.width}x{args.height}  steps={args.steps}")
    print(f"avg latency:      {avg:.3f}s")
    print(f"avg images/sec:   {1.0 / avg:.3f}")
    print(f"wall (3 imgs):    {wall:.2f}s")
    print(f"peak vram (last): {rows[-1].get('vram_max_allocated_mb')} MB")
    print("=" * 60)
    print("(Image models report latency / images-per-sec, not tok/s.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
