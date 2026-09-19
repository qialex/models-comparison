#!/usr/bin/env python3
"""Stress Kokoro GPU: peak VRAM under long + multi-voice + concurrent load."""
from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path(__file__).with_name("vram_stress.json")
BASE = "http://127.0.0.1:8131"

VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

# ~3–4 min audio (~3500 chars) — long single request
LONG = (
    "Kokoro stress test paragraph for peak memory. "
    "The quick brown fox jumps over the lazy dog beside the river. "
) * 40
LONG = LONG[:3500]

MIX = "af_bella,af_jessica,af_nicole,am_michael"

lock = threading.Lock()
samples: list[dict] = []


def nvidia() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used, total, util = [x.strip() for x in out.split(",")]
        return {
            "smi_used_mb": float(used),
            "smi_total_mb": float(total),
            "smi_util": float(util),
        }
    except Exception as e:
        return {"smi_error": str(e)}


def health() -> dict:
    with urllib.request.urlopen(f"{BASE}/health", timeout=30) as r:
        return json.loads(r.read().decode())


def tts(text: str, voice: str, return_audio: bool = False) -> dict:
    payload = json.dumps(
        {
            "text": text,
            "voice": voice,
            "speed": 1.0,
            "return_audio": return_audio,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/tts",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def snap(label: str) -> dict:
    h = health()
    n = nvidia()
    row = {
        "t": round(time.time(), 3),
        "label": label,
        "torch_alloc_mb": h.get("vram_allocated_mb"),
        "torch_reserved_mb": h.get("vram_reserved_mb"),
        **n,
    }
    with lock:
        samples.append(row)
    print(
        f"  [{label}] torch_alloc={row.get('torch_alloc_mb')} "
        f"torch_rsv={row.get('torch_reserved_mb')} "
        f"smi={row.get('smi_used_mb')} MiB",
        flush=True,
    )
    return row


def poller(stop: threading.Event, interval: float = 0.5):
    i = 0
    while not stop.wait(interval):
        i += 1
        try:
            snap(f"poll_{i}")
        except Exception:
            pass


def main():
    print("baseline")
    base = snap("baseline")
    stop = threading.Event()
    th = threading.Thread(target=poller, args=(stop,), daemon=True)
    th.start()

    phases = []

    # 1) warm all voices (cache packs on device)
    print("\n=== phase: load all voices (short) ===")
    t0 = time.time()
    for v in VOICES:
        tts("Short warm line.", v, return_audio=False)
    phases.append({"phase": "load_all_voices", "s": round(time.time() - t0, 2)})
    snap("after_all_voices")

    # 2) long single request + audio payload (wav in RAM on server briefly)
    print("\n=== phase: long single + wav ===")
    t0 = time.time()
    long_body = tts(LONG, "af_heart", return_audio=True)
    phases.append(
        {
            "phase": "long_wav",
            "s": round(time.time() - t0, 2),
            "audio_s": long_body.get("audio_s"),
            "generate_s": long_body.get("generate_s"),
            "chars": len(LONG),
        }
    )
    snap("after_long_wav")

    # 3) long with voice mix
    print("\n=== phase: long voice-mix ===")
    t0 = time.time()
    mix_body = tts(LONG, MIX, return_audio=False)
    phases.append(
        {
            "phase": "long_mix",
            "s": round(time.time() - t0, 2),
            "audio_s": mix_body.get("audio_s"),
        }
    )
    snap("after_mix")

    # 4) concurrent long requests (peak activation)
    print("\n=== phase: concurrent x4 long ===")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(tts, LONG, VOICES[i % len(VOICES)], False)
            for i in range(4)
        ]
        for f in as_completed(futs):
            f.result()
    phases.append({"phase": "concurrent_x4_long", "s": round(time.time() - t0, 2)})
    snap("after_concurrent")

    # 5) concurrent x8 shorter
    print("\n=== phase: concurrent x8 medium ===")
    med = LONG[:1500]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(tts, med, VOICES[i % len(VOICES)], False) for i in range(8)]
        errs = 0
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                errs += 1
                print("  concurrent err", e)
    phases.append(
        {"phase": "concurrent_x8_medium", "s": round(time.time() - t0, 2), "errors": errs}
    )
    snap("after_concurrent8")

    stop.set()
    time.sleep(0.6)
    final = snap("final")

    alloc = [s["torch_alloc_mb"] for s in samples if s.get("torch_alloc_mb") is not None]
    rsv = [s["torch_reserved_mb"] for s in samples if s.get("torch_reserved_mb") is not None]
    smi = [s["smi_used_mb"] for s in samples if s.get("smi_used_mb") is not None]

    summary = {
        "goal": "fit on cheap 4GB VRAM Vast box",
        "baseline": base,
        "final": final,
        "peak_torch_alloc_mb": max(alloc) if alloc else None,
        "peak_torch_reserved_mb": max(rsv) if rsv else None,
        "peak_smi_used_mb": max(smi) if smi else None,
        "delta_smi_vs_baseline_mb": (max(smi) - base["smi_used_mb"]) if smi and base.get("smi_used_mb") else None,
        "fit_4gb": {
            "torch_alloc_under_3500": (max(alloc) or 0) < 3500,
            "torch_reserved_under_3500": (max(rsv) or 0) < 3500,
            "note": "Use torch reserved / isolated container for true footprint; host SMI includes other processes",
        },
        "phases": phases,
        "samples_n": len(samples),
        "samples": samples,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n========== PEAK ==========")
    print(f"torch_alloc peak:    {summary['peak_torch_alloc_mb']} MB")
    print(f"torch_reserved peak: {summary['peak_torch_reserved_mb']} MB")
    print(f"nvidia-smi peak:     {summary['peak_smi_used_mb']} MiB")
    print(f"smi delta vs base:   {summary['delta_smi_vs_baseline_mb']} MiB")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
