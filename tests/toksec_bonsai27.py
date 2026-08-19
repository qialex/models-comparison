#!/usr/bin/env python3
"""Tok/s suite against a llama.cpp OpenAI-compatible server.

MLX weights (prism-ml/Bonsai-27B-mlx-1bit) cannot run on Windows/CUDA.
This hits the GGUF twin (bonsai27_gpu on :8101 by default).

Cases:
  boolean_1k   ~1000-token prompt, short true/false answer
  soak_500     ~500-token prompt, then keep generating until --min-seconds
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
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tokenize(base: str, text: str, timeout: int) -> list[int]:
    out = post_json(f"{base}/tokenize", {"content": text}, timeout=timeout)
    tokens = out.get("tokens")
    if isinstance(tokens, list):
        return tokens
    raise RuntimeError(f"unexpected tokenize response: {out!r}")


def count_tokens(base: str, text: str, timeout: int) -> int:
    tokens = tokenize(base, text, timeout)
    if isinstance(tokens, list):
        return len(tokens)
    raise RuntimeError(f"unexpected tokenize response: {tokens!r}")


FILLER = (
    "The survey station logged rainfall, wind direction, soil temperature, "
    "and canopy cover for plot {n} on the northern ridge. Observers noted "
    "lichen on granite, a narrow trail used by deer, and a creek that ran "
    "clear after the overnight frost. Nothing in this paragraph is needed "
    "to answer the final question; it only lengthens the prompt."
)


def pad_user_text(base: str, prefix: str, target: int, timeout: int) -> str:
    parts: list[str] = []
    n = 0
    body = prefix
    while count_tokens(base, body, timeout) < target:
        n += 1
        parts.append(FILLER.format(n=n))
        body = "\n".join(parts) + "\n\n" + prefix
        if n > 400:
            raise RuntimeError(f"could not reach {target} tokens after {n} filler blocks")
    return body


def chat(base: str, messages: list[dict], max_tokens: int, timeout: int) -> dict:
    return post_json(
        f"{base}/v1/chat/completions",
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )


def timings_of(resp: dict) -> dict:
    choice = resp["choices"][0]["message"]
    content = (choice.get("content") or "").strip()
    t = resp.get("timings") or {}
    return {
        "answer": content.replace("\n", " ")[:160],
        "content": content,
        "prompt_tokens": int(t.get("prompt_n") or 0),
        "gen_tokens": int(t.get("predicted_n") or 0),
        "prompt_tok_s": float(t["prompt_per_second"]) if t.get("prompt_per_second") else None,
        "gen_tok_s": float(t["predicted_per_second"]) if t.get("predicted_per_second") else None,
        "prompt_ms": float(t.get("prompt_ms") or 0),
        "gen_ms": float(t.get("predicted_ms") or 0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8101", help="llama.cpp base URL (no /v1)")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--min-seconds", type=float, default=30.0, help="keep generating at least this long")
    p.add_argument("--chunk-tokens", type=int, default=128, help="tokens per soak request")
    args = p.parse_args()
    base = args.base.rstrip("/")

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"health check failed for {base}: {e}", file=sys.stderr)
        return 1

    print(f"server {base} health={health}")
    print("warmup…")
    chat(base, [{"role": "user", "content": "Reply with the single word ok."}], 8, args.timeout)

    bool_q = (
        "Using only the last sentence, answer with a single word: true or false.\n"
        "The capital of France is Paris."
    )
    gen_q = (
        "Write a clear explanation of why Paris is the capital of France. "
        "Use complete sentences. Keep going with related history until asked to stop. "
        "Do not repeat the background notes."
    )

    print("building ~1000-token boolean prompt…")
    bool_user = pad_user_text(base, bool_q, 1000, args.timeout)
    print("building ~500-token generation prompt…")
    gen_user = pad_user_text(base, gen_q, 500, args.timeout)

    print("running boolean_1k…")
    r1 = timings_of(chat(base, [{"role": "user", "content": bool_user}], 8, args.timeout))
    r1["case"] = "boolean_1k"
    r1["rounds"] = 1
    r1["wall_s"] = round((r1["prompt_ms"] + r1["gen_ms"]) / 1000.0, 3)
    if r1["prompt_tok_s"] is not None:
        r1["prompt_tok_s"] = round(r1["prompt_tok_s"], 2)
    if r1["gen_tok_s"] is not None:
        r1["gen_tok_s"] = round(r1["gen_tok_s"], 2)
    r1.pop("content", None)

    print(f"running soak_500 for at least {args.min_seconds:.0f}s…")
    messages = [{"role": "user", "content": gen_user}]
    wall0 = time.perf_counter()
    total_gen = 0
    total_gen_ms = 0.0
    total_prompt = 0
    total_prompt_ms = 0.0
    rounds = 0
    last_answer = ""
    while True:
        resp = chat(base, messages, args.chunk_tokens, args.timeout)
        t = timings_of(resp)
        rounds += 1
        total_gen += t["gen_tokens"]
        total_gen_ms += t["gen_ms"]
        total_prompt += t["prompt_tokens"]
        total_prompt_ms += t["prompt_ms"]
        last_answer = t["content"]
        elapsed = time.perf_counter() - wall0
        print(
            f"  round {rounds}: gen={t['gen_tokens']} @ "
            f"{t['gen_tok_s']:.1f} tok/s  wall={elapsed:.1f}s"
        )
        if elapsed >= args.min_seconds:
            break
        if t["gen_tokens"] <= 0:
            print("  model produced 0 tokens; stopping soak", file=sys.stderr)
            break
        messages.append({"role": "assistant", "content": t["content"]})
        messages.append(
            {
                "role": "user",
                "content": "Continue from the last sentence. Do not recap.",
            }
        )

    wall_s = time.perf_counter() - wall0
    r2 = {
        "case": "soak_500",
        "answer": last_answer.replace("\n", " ")[:160],
        "prompt_tokens": total_prompt,
        "gen_tokens": total_gen,
        "prompt_tok_s": round(total_prompt / (total_prompt_ms / 1000.0), 2) if total_prompt_ms else None,
        "gen_tok_s": round(total_gen / (total_gen_ms / 1000.0), 2) if total_gen_ms else None,
        "prompt_ms": round(total_prompt_ms, 3),
        "gen_ms": round(total_gen_ms, 3),
        "rounds": rounds,
        "wall_s": round(wall_s, 3),
    }

    rows = [r1, r2]
    printable = [{k: v for k, v in r.items() if k != "content"} for r in rows]
    print(json.dumps(printable, indent=2))
    print()
    print(
        f"{'case':<12} {'rounds':>6} {'wall_s':>8} {'prompt_n':>8} "
        f"{'gen_n':>6} {'prompt tok/s':>13} {'gen tok/s':>10}"
    )
    for r in rows:
        print(
            f"{r['case']:<12} {r['rounds']:>6} {r['wall_s']:>8} {r['prompt_tokens']:>8} "
            f"{r['gen_tokens']:>6} {r['prompt_tok_s']:>13} {r['gen_tok_s']:>10}"
        )
        print(f"  answer: {r['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
