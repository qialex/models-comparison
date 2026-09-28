#!/usr/bin/env python3
"""Smoke test: health + a few /tts calls across text lengths, report RTF.

Same TEXTS/shape as kokoro-82m/smoke_test.py and
qwen3-tts-*-customvoice/smoke_test.py, for direct cross-model RTF comparison.
Uses the bundled default reference voice (no ref_audio override).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

TEXTS = {
    "short": "The quick brown fox jumps over the lazy dog.",
    "medium": (
        "Qwen3-TTS is a text-to-speech system covering ten major languages. "
        "It supports natural language instruction based voice control and "
        "streaming generation with very low latency."
    ),
    "long": (
        "Qwen3-TTS is a text-to-speech system covering ten major languages, "
        "including Chinese, English, Japanese, Korean, German, French, Russian, "
        "Portuguese, Spanish, and Italian. It supports natural language "
        "instruction based voice control, adaptive tone and speaking rate, and "
        "streaming generation with end-to-end synthesis latency as low as "
        "ninety-seven milliseconds. This paragraph exists purely to give the "
        "benchmark a longer utterance to synthesize so we can compare "
        "real-time factor across text lengths."
    ),
}


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0) -> tuple[dict, float]:
    t0 = time.perf_counter()
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8198")
    p.add_argument("--save-wav", type=Path, default=None, help="Save the 'long' wav here for a listen check")
    args = p.parse_args()

    base = args.base.rstrip("/")
    print(f"waiting for {base}/health ...", flush=True)
    health = wait_health(base)
    print("health:", health, flush=True)

    # warmup (first call pays for CUDA kernel compilation / cache warmup)
    print("\nwarmup...", flush=True)
    body, rtt_ms = http_json(f"{base}/tts", {"text": "Warmup.", "return_audio": False})
    print(f"  warmup rtt={rtt_ms:.0f}ms gen={body.get('generate_s')}s", flush=True)

    results = []
    for label, text in TEXTS.items():
        want_audio = args.save_wav is not None and label == "long"
        body, rtt_ms = http_json(
            f"{base}/tts",
            {"text": text, "return_audio": want_audio},
        )
        row = {
            "label": label,
            "chars": len(text),
            "rtt_ms": round(rtt_ms, 1),
            "generate_s": body.get("generate_s"),
            "audio_s": body.get("audio_s"),
            "rtf": body.get("rtf"),
            "sample_rate": body.get("sample_rate"),
            "vram_allocated_mb": body.get("vram_allocated_mb"),
        }
        results.append(row)
        print(
            f"  [{label:6}] chars={row['chars']:4} audio={row['audio_s']}s "
            f"gen={row['generate_s']}s rtf={row['rtf']} rtt={rtt_ms:.0f}ms "
            f"vram={row['vram_allocated_mb']}MB",
            flush=True,
        )
        if want_audio and "wav_base64" in body:
            import base64

            args.save_wav.write_bytes(base64.b64decode(body["wav_base64"]))
            print(f"  wrote {args.save_wav}", flush=True)

    print("\n========== SUMMARY ==========")
    print(json.dumps({"health": health, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
