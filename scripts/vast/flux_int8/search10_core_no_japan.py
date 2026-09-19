#!/usr/bin/env python3
"""10 repeated searches: cheapest Flux core, Japan excluded. No rent."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug  # noqa: E402

import json
import statistics
import subprocess
import time
from collections import Counter

OUT = cache_debug("flux2-klein-9b-kv-int8", "_debug") / "search10_core_cheapest_no_japan_live.json"
MIN_EFF = 11.5
MIN_RAM = 30
N = 10
QUERY = "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True"
EXCLUDE_OFFERS = {45607706}
EXCLUDE_MACHINES = {54116}


def is_japan(r) -> bool:
    if r["id"] in EXCLUDE_OFFERS or r.get("machine") in EXCLUDE_MACHINES:
        return True
    loc = (r.get("loc") or "").lower()
    return "japan" in loc or ", jp" in loc


def search():
    return json.loads(
        subprocess.check_output(
            [
                "vastai",
                "search",
                "offers",
                QUERY,
                "--order",
                "dph_total",
                "--limit",
                "80",
                "--raw",
                "-n",
            ],
            text=True,
        )
    )


def norm(o):
    vram = float(o.get("gpu_ram") or 0)
    vram_g = vram / 1024 if vram > 64 else vram
    frac = float(o.get("gpu_frac") if o.get("gpu_frac") is not None else 1.0)
    eff = frac * vram_g
    ram = float(o.get("cpu_ram") or 0)
    ram_g = ram / 1024 if ram > 512 else ram
    return {
        "id": o.get("id"),
        "machine": int(o.get("machine_id") or 0),
        "price": float(o.get("dph_total") or 99),
        "gpu": o.get("gpu_name"),
        "vram_g": round(vram_g, 2),
        "frac": round(frac, 4),
        "eff_vram": round(eff, 2),
        "ram_g": round(ram_g, 1),
        "ports": int(o.get("direct_port_count") or 0),
        "verified": bool(o.get("verified")),
        "static_ip": bool(o.get("static_ip")),
        "reli": round(float(o.get("reliability2") or o.get("reliability") or 0), 3),
        "loc": o.get("geolocation") or "",
    }


def fit(rows):
    return sorted(
        [
            r
            for r in rows
            if r["eff_vram"] >= MIN_EFF and r["ram_g"] > MIN_RAM and not is_japan(r)
        ],
        key=lambda r: r["price"],
    )


def main():
    rounds = []
    for i in range(1, N + 1):
        raw = search()
        rows = [norm(o) for o in raw]
        hits = fit(rows)
        top = hits[:8]
        print(f"\n=== search {i}/{N}  pool={len(rows)}  core_fit_no_jp={len(hits)} ===")
        for r in top:
            print(
                f"  {r['id']} ${r['price']:.4f} {r['gpu']} "
                f"frac={r['frac']:.3f} eff={r['eff_vram']:.1f}G ram={r['ram_g']:.0f}G "
                f"ports={r['ports']} static={r['static_ip']} reli={r['reli']:.2f} "
                f"ver={r['verified']} {r['loc']}"
            )
        if not hits:
            print("  (none)")
        rounds.append(
            {
                "i": i,
                "pool": len(rows),
                "fit_n": len(hits),
                "cheapest": hits[0] if hits else None,
                "top": top,
            }
        )
        if i < N:
            time.sleep(2)

    cheapest_prices = [r["cheapest"]["price"] for r in rounds if r["cheapest"]]
    ids = Counter(r["cheapest"]["id"] for r in rounds if r["cheapest"])
    summary = {
        "query": QUERY,
        "filters": "eff_vram>=11.5 AND ram_g>30, Japan/45607706/machine 54116 excluded",
        "n_searches": N,
        "rounds_with_fit": len(cheapest_prices),
        "cheapest_min": min(cheapest_prices) if cheapest_prices else None,
        "cheapest_max": max(cheapest_prices) if cheapest_prices else None,
        "cheapest_median": statistics.median(cheapest_prices) if cheapest_prices else None,
        "cheapest_mean": statistics.mean(cheapest_prices) if cheapest_prices else None,
        "top_offer_frequency": ids.most_common(10),
        "rounds": rounds,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n========== SUMMARY ==========")
    print(json.dumps({k: summary[k] for k in summary if k != "rounds"}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
