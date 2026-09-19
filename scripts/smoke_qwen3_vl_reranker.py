#!/usr/bin/env python3
"""Smoke latency + mature/SFW detection checks for Qwen3-VL-Reranker-2B (:8128)."""
from __future__ import annotations

import base64
import json
import statistics
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8128"
ROOT = Path(__file__).resolve().parents[1]
DEMO_IMG = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def post(path: str, body: dict, timeout: int = 180):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def as_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def bench(name: str, body: dict, n: int = 3, path: str = "/score"):
    times = []
    last = None
    for i in range(n):
        out, wall = post(path, body)
        times.append(wall)
        last = out
        print(
            f"  [{i+1}/{n}] wall={wall*1000:.0f}ms "
            f"api={out.get('latency_s')}s "
            f"vram_peak={out.get('vram_max_allocated_mb')}"
        )
    med = statistics.median(times)
    row = {
        "case": name,
        "n": n,
        "wall_med_ms": round(med * 1000, 1),
        "wall_min_ms": round(min(times) * 1000, 1),
        "wall_max_ms": round(max(times) * 1000, 1),
        "api_latency_s": last.get("latency_s"),
        "scores": last.get("scores") or last.get("rankings"),
    }
    print(f"  -> median wall {med*1000:.0f}ms\n")
    return last, row


def main() -> int:
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        h = json.load(r)
    print("HEALTH", json.dumps({k: h[k] for k in h if k != "default_prompt"}, indent=2), "\n")

    sfw_local = None
    mature_local = None
    for p in [
        ROOT / "cache/flux2-klein-9b-kv-int8/_debug/vast-50320491-smoke.png",
        ROOT / "cache/flux2-klein-9b-kv-int8/_debug/vast-50320562-smoke.png",
    ]:
        if p.exists():
            sfw_local = p
            break
    for p in [
        ROOT / "results/image-bench/flux2-klein-9b-kv-int8-vast/04_bath_scene/image.png",
        ROOT / "results/image-bench/flux2-klein-9b-kv-int8-vast-50320562/04_bath_scene/image.png",
        ROOT / "results/image-bench/flux2-klein-9b-kv-int8-vast/01_studio_portrait/image.png",
    ]:
        if p.exists():
            mature_local = p
            break

    rows = []

    print("=== TEXT small (1q x 2 short docs) ===")
    _, row = bench(
        "text_small_2docs",
        {
            "query": "A woman playing with her dog on a beach at sunset.",
            "documents": [
                "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
                "A cat sleeping on a windowsill.",
            ],
            "sigmoid": True,
        },
        n=5,
    )
    rows.append(row)

    print("=== TEXT large (1q x 32 longer docs) ===")
    long_rel = (
        "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
        "as the dog offers its paw in a heartwarming display of companionship and trust. Soft golden light, "
        "wet sand, gentle waves in the background."
    )
    long_irrel = (
        "Technical documentation for a distributed key-value store describing Raft consensus, WAL replay, "
        "snapshot compaction, and multi-tenant quota enforcement under high write throughput."
    )
    docs = [long_rel if i % 4 == 0 else f"{long_irrel} Variant {i}." for i in range(32)]
    _, row = bench(
        "text_large_32docs",
        {
            "query": "Find scenes of a woman playing with a dog on the beach at sunset.",
            "documents": docs,
            "sigmoid": True,
        },
        n=3,
    )
    rows.append(row)

    print("=== TEXT mature-vs-SFW (query = NSFW nude photography) ===")
    last, row = bench(
        "text_mature_detect",
        {
            "query": "explicit adult nude photography, full nudity, sexualized body, NSFW",
            "documents": [
                "Photorealistic adult nude woman in a photography studio, full nudity, editorial nude.",
                "A red bicycle parked on a rainy city street at night, cinematic lighting.",
                "Family picnic in a sunny park with children flying a kite, wholesome.",
                "Tasteful art-nude figure study of an adult model reclining, candles, wet skin.",
            ],
            "prompt": "Determine how relevant the document is to adult NSFW / explicit nude content.",
            "sigmoid": True,
        },
        n=3,
    )
    rows.append(row)
    labels = ["nude_studio", "bicycle_sfw", "family_sfw", "art_nude"]
    for lab, s in zip(labels, last["scores"]):
        print(f"    {lab}: {s:.4f}")
    print()

    print("=== IMAGE remote URL (SFW beach demo) ===")
    _, row = bench(
        "image_url_sfw",
        {
            "query": "A woman playing with her dog on a beach at sunset.",
            "documents": [DEMO_IMG, "A cat sleeping on a windowsill."],
            "sigmoid": True,
        },
        n=3,
    )
    rows.append(row)

    if sfw_local and mature_local:
        print("=== IMAGE local mature-vs-SFW ===")
        print(f"  SFW:    {sfw_local}")
        print(f"  mature: {mature_local}")
        last, row = bench(
            "image_local_mature_detect",
            {
                "query": "explicit adult nude photography, NSFW, full nudity",
                "documents": [
                    {"image": as_data_url(mature_local)},
                    {"image": as_data_url(sfw_local)},
                    "A red bicycle on a rainy street, no people.",
                ],
                "prompt": "Determine how relevant the document is to adult NSFW / explicit nude content.",
                "sigmoid": True,
            },
            n=3,
        )
        rows.append(row)
        for lab, s in zip(["mature_img", "sfw_img", "sfw_text"], last["scores"]):
            print(f"    {lab}: {s:.4f}")
        print()

        print("=== IMAGE same docs vs NSFW vs SFW queries ===")
        docs = [
            {"image": as_data_url(mature_local)},
            {"image": as_data_url(sfw_local)},
        ]
        for qname, q in [
            ("nsfw_q", "explicit adult nude photography NSFW"),
            ("sfw_q", "wholesome family-friendly outdoor scene, no nudity"),
        ]:
            out, wall = post(
                "/score",
                {
                    "query": q,
                    "documents": docs,
                    "prompt": "Rate relevance to the query.",
                    "sigmoid": True,
                },
            )
            print(
                f"  {qname} wall={wall*1000:.0f}ms "
                f"mature_img={out['scores'][0]:.4f} sfw_img={out['scores'][1]:.4f}"
            )
            rows.append(
                {
                    "case": f"image_dual_{qname}",
                    "wall_ms": round(wall * 1000, 1),
                    "api_latency_s": out.get("latency_s"),
                    "scores": out["scores"],
                }
            )
        print()
    else:
        print("=== IMAGE local skipped ===", "sfw=", sfw_local, "mature=", mature_local, "\n")

    # Rough "pairs/sec" from large text case
    large = next(r for r in rows if r["case"] == "text_large_32docs")
    pairs = 32
    pps = pairs / (large["api_latency_s"] or large["wall_med_ms"] / 1000)
    print("=== SUMMARY ===")
    print(json.dumps(rows, indent=2))
    print(f"\nApprox pairs/sec on text_large_32docs (api latency): {pps:.1f}")
    print("Note: this is a reranker (scores), not a generative LLM — no tokens/sec.")
    out_path = ROOT / "cache" / "qwen3-vl-reranker-2b" / "_debug" / "smoke-latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"health": h, "rows": rows, "pairs_per_sec_large": pps}, indent=2) + "\n")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
