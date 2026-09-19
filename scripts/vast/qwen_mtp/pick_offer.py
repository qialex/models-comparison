#!/usr/bin/env python3
"""Rank Vast offers for Qwen MTP ngram (local :8112 recipe)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug  # noqa: E402

import json
import subprocess
import sys

OUT = cache_debug("qwen35-4b-mtp-ngram", "_debug") / "offers_ranked.json"
QUERY = (
    "gpu_ram>=8 cuda_max_good>=12.0 rented=False num_gpus=1 "
    "disk_space>=16 reliability>0.9"
)


def main() -> None:
    raw = subprocess.check_output(
        [
            "vastai",
            "search",
            "offers",
            QUERY,
            "-o",
            "dph_total-",
            "--limit",
            "40",
            "--raw",
        ],
        text=True,
    )
    offers = json.loads(raw)
    rows = []
    for o in offers:
        geo = str(o.get("geolocation") or "")
        if "Japan" in geo:
            continue
        bw = float(o.get("gpu_mem_bw") or 0)
        frac = o.get("gpu_frac") or 1.0
        if isinstance(frac, (list, tuple)):
            frac = float(frac[0] if frac else 1.0)
        else:
            frac = float(frac)
        price = float(o.get("dph_total") or 0) or 1e9
        score = (bw * frac) / (price * 100)
        rows.append(
            {
                "id": o["id"],
                "gpu_name": o.get("gpu_name"),
                "dph_total": price,
                "gpu_mem_bw": bw,
                "gpu_frac": frac,
                "score": round(score, 2),
                "reliability": o.get("reliability2") or o.get("reliability"),
                "static_ip": o.get("static_ip"),
                "geolocation": geo,
                "verified": o.get("verified"),
                "gpu_ram": o.get("gpu_ram"),
                "disk_space": o.get("disk_space"),
            }
        )
    rows.sort(key=lambda r: (-r["score"], r["dph_total"]))
    OUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"n={len(rows)} wrote {OUT}")
    for r in rows[:15]:
        print(
            f"{r['id']} {r['gpu_name']} ${r['dph_total']:.4f} "
            f"score={r['score']:.1f} bw={r['gpu_mem_bw']:.0f} "
            f"reli={r['reliability']} static={r['static_ip']} "
            f"{r['geolocation']} ver={r['verified']}"
        )


if __name__ == "__main__":
    main()
