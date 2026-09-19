#!/usr/bin/env python3
"""Bench Kokoro CPU vs GPU: ~10 min audio, short vs long utterance split."""
from __future__ import annotations

import json
import statistics
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).with_name("bench_10m_short_long.json")

# ~1000 chars ≈ 1 min audio (HF rule of thumb) → ~10k chars ≈ 10 min
PARA = (
    "Kokoro is a lightweight open text-to-speech model with eighty-two million parameters. "
    "It produces natural speech quickly on modest hardware, and ships many preset voices. "
    "This paragraph is filler used only to reach a stable target duration for benchmarking. "
)

# Long: few multi-sentence blocks (~2 min each × 5 ≈ 10 min)
LONG_BLOCKS = 5
CHARS_PER_LONG = 2000  # ~2 min each

# Short: many brief lines (~6–8 s each)
SHORT_TARGET_CHARS = 10000
SHORT_LINE = "The red bicycle crossed the foggy bridge before sunrise."  # ~55 chars


def make_long_text() -> str:
    unit = PARA * ((CHARS_PER_LONG // len(PARA)) + 1)
    blocks = [unit[:CHARS_PER_LONG].rstrip() + "." for _ in range(LONG_BLOCKS)]
    return "\n\n".join(blocks)


def make_short_texts() -> list[str]:
    lines: list[str] = []
    total = 0
    i = 0
    while total < SHORT_TARGET_CHARS:
        i += 1
        line = f"{SHORT_LINE} Item number {i}."
        lines.append(line)
        total += len(line)
    return lines


def http_json(url: str, payload: dict | None = None, timeout: float = 3600.0) -> tuple[dict, float]:
    t0 = time.perf_counter()
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode()), (time.perf_counter() - t0) * 1000


def wait_health(base: str, timeout_s: float = 600.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            h, _ = http_json(f"{base}/health", timeout=30)
            if h.get("status") == "ok":
                return h
            last = h
        except Exception as e:
            last = {"error": str(e)}
        time.sleep(3)
    raise SystemExit(f"health timeout {base}: {last}")


def run_mode(base: str, label: str, texts: list[str], split_pattern: str | None) -> dict:
    print(f"\n--- {label}: {len(texts)} request(s) ---")
    # warmup
    http_json(
        f"{base}/tts",
        {"text": "Warmup sentence.", "voice": "af_heart", "speed": 1.0, "return_audio": False},
        timeout=120,
    )
    runs = []
    wall0 = time.perf_counter()
    for i, text in enumerate(texts, 1):
        body, rtt_ms = http_json(
            f"{base}/tts",
            {
                "text": text,
                "voice": "af_heart",
                "speed": 1.0,
                "split_pattern": split_pattern,
                "return_audio": False,
            },
        )
        runs.append(
            {
                "i": i,
                "chars": len(text),
                "rtt_ms": round(rtt_ms, 1),
                "generate_s": body.get("generate_s"),
                "audio_s": body.get("audio_s"),
                "rtf": body.get("rtf"),
                "segments": body.get("segments"),
            }
        )
        print(
            f"  [{i}/{len(texts)}] audio={body.get('audio_s')}s "
            f"gen={body.get('generate_s')}s rtf={body.get('rtf')} "
            f"rtt={rtt_ms:.0f}ms segs={body.get('segments')}"
        )
    wall_s = time.perf_counter() - wall0
    audio_total = sum(r["audio_s"] or 0 for r in runs)
    gen_total = sum(r["generate_s"] or 0 for r in runs)
    return {
        "label": label,
        "n_requests": len(texts),
        "chars_total": sum(len(t) for t in texts),
        "audio_s_total": round(audio_total, 2),
        "audio_min_total": round(audio_total / 60, 2),
        "generate_s_total": round(gen_total, 2),
        "wall_s_total": round(wall_s, 2),
        "rtf_overall_gen": round(gen_total / audio_total, 4) if audio_total else None,
        "rtf_overall_wall": round(wall_s / audio_total, 4) if audio_total else None,
        "rtf_median": statistics.median([r["rtf"] for r in runs if r.get("rtf")]),
        "runs": runs,
    }


def bench_endpoint(name: str, base: str) -> dict:
    print(f"\n========== {name} {base} ==========")
    health = wait_health(base)
    print("health", health)
    short_texts = make_short_texts()
    long_text = make_long_text()
    short = run_mode(base, "short", short_texts, split_pattern=None)
    long = run_mode(base, "long", [long_text], split_pattern=r"\n+")
    return {"name": name, "base": base, "health": health, "short": short, "long": long}


def main():
    endpoints = [
        ("cpu", "http://127.0.0.1:8130"),
        ("gpu", "http://127.0.0.1:8131"),
    ]
    results = [bench_endpoint(n, b) for n, b in endpoints]
    out = {
        "method": {
            "target": "~10 min audio per mode (short & long), both CPU and GPU",
            "char_rule_of_thumb": "~1000 chars ≈ 1 min",
            "short": "many ~1-line utterances, sequential /tts, no split_pattern",
            "long": "5 ~2k-char paragraphs joined with blank lines, split_pattern=\\n+",
            "metrics": "generate_s = server synthesis; wall includes HTTP; rtf = time/audio",
            "voice": "af_heart",
            "speed": 1.0,
            "return_audio": False,
        },
        "results": results,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n========== SUMMARY ==========")
    for r in results:
        for mode in ("short", "long"):
            m = r[mode]
            print(
                f"{r['name']:3} {mode:5} | audio={m['audio_min_total']:.2f} min | "
                f"gen={m['generate_s_total']:.1f}s | wall={m['wall_s_total']:.1f}s | "
                f"rtf_gen={m['rtf_overall_gen']} rtf_wall={m['rtf_overall_wall']}"
            )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
