#!/usr/bin/env python3
"""20 searches for qwen3.5-4b-mtp-ngram: static_ip mandatory, frac OK.

Fit: effVRAM>=4GB, RAM>=4GB, disk>=8GB, static_ip=True. No rent.

Primary rank: score = (gpu_mem_bw * frac) / price_cents
  i.e. effective GPU mem BW (GB/s) per 1 cent of $/hr.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug  # noqa: E402

import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict

OUT = cache_debug("qwen35-4b-mtp-ngram", "_debug") / "search20_static_analysis.json"
TEMPLATE = "qwen3.5-4b-mtp-ngram"
MIN_EFF = 4.0
MIN_RAM = 4.0
MIN_DISK = 8
N = 20
# static_ip in Vast query; unverified included via -n
QUERY = (
    f"gpu_ram>={MIN_EFF} cpu_ram>={MIN_RAM} disk_space>={MIN_DISK} "
    "num_gpus=1 rentable=True static_ip=True"
)


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
                "100",
                "--raw",
                "-n",
            ],
            text=True,
        )
    )


def _gb(val: float, threshold: float) -> float:
    """Vast sometimes returns MiB (>threshold) else already GB."""
    return val / 1024 if val > threshold else val


def score_bw_per_cent(gpu_mem_bw: float, frac: float, price: float) -> float:
    """(gpu_mem_bw * frac) per 1 cent of $/hr."""
    cents = price * 100.0
    if cents <= 0:
        return 0.0
    return (gpu_mem_bw * frac) / cents


def norm(o: dict) -> dict:
    vram = float(o.get("gpu_ram") or 0)
    vram_g = _gb(vram, 64)
    frac = float(o.get("gpu_frac") if o.get("gpu_frac") is not None else 1.0)
    ram = float(o.get("cpu_ram") or 0)
    ram_g = _gb(ram, 512)
    price = float(o.get("dph_total") or 99)
    gpu_mem_bw = float(o.get("gpu_mem_bw") or 0)
    bw_frac = gpu_mem_bw * frac
    return {
        "id": o.get("id"),
        "machine": int(o.get("machine_id") or 0),
        "price": price,
        "gpu": o.get("gpu_name"),
        "vram_g": round(vram_g, 2),
        "frac": round(frac, 4),
        "eff_vram": round(frac * vram_g, 2),
        "ram_g": round(ram_g, 1),
        "disk": float(o.get("disk_space") or 0),
        "gpu_mem_bw": gpu_mem_bw,  # GB/s, per-GPU mem BW
        "bw_frac": round(bw_frac, 2),  # gpu_mem_bw * frac
        "score": round(score_bw_per_cent(gpu_mem_bw, frac, price), 4),
        "pcie_bw": float(o.get("pcie_bw") or 0),
        "inet_down": float(o.get("inet_down") or 0),
        "inet_up": float(o.get("inet_up") or 0),
        "static_ip": bool(o.get("static_ip")),
        "verified": bool(o.get("verified") or (o.get("verification") == "verified")),
        "reli": round(float(o.get("reliability2") or o.get("reliability") or 0), 3),
        "ports": int(o.get("direct_port_count") or 0),
        "loc": o.get("geolocation") or "",
    }


def fit(rows: list[dict]) -> list[dict]:
    # best value first: (gpu_mem_bw * frac) per cent of price
    return sorted(
        [
            r
            for r in rows
            if r["eff_vram"] >= MIN_EFF
            and r["ram_g"] >= MIN_RAM
            and r["disk"] >= MIN_DISK
            and r["static_ip"]
        ],
        key=lambda r: (-r["score"], r["price"]),
    )


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def price_stats(prices: list[float]) -> dict:
    if not prices:
        return {"n": 0}
    return {
        "n": len(prices),
        "min": min(prices),
        "p25": pct(prices, 25),
        "median": statistics.median(prices),
        "p75": pct(prices, 75),
        "mean": statistics.mean(prices),
        "max": max(prices),
    }


def analyze(all_hits: list[dict], rounds: list[dict]) -> dict:
    prices = [r["price"] for r in all_hits]
    scores = [r["score"] for r in all_hits]
    by_loc: dict[str, list[float]] = defaultdict(list)
    by_gpu: dict[str, list[float]] = defaultdict(list)
    by_ver: dict[str, list[float]] = defaultdict(list)
    by_eff_bucket: dict[str, list[float]] = defaultdict(list)
    score_by_loc: dict[str, list[float]] = defaultdict(list)

    for r in all_hits:
        loc = r["loc"] or "(unknown)"
        by_loc[loc].append(r["price"])
        score_by_loc[loc].append(r["score"])
        by_gpu[r["gpu"] or "?"].append(r["price"])
        by_ver["verified" if r["verified"] else "unverified"].append(r["price"])
        eff = r["eff_vram"]
        if eff < 6:
            bucket = "4-6G"
        elif eff < 8:
            bucket = "6-8G"
        elif eff < 12:
            bucket = "8-12G"
        elif eff < 16:
            bucket = "12-16G"
        else:
            bucket = "16G+"
        by_eff_bucket[bucket].append(r["price"])

    best_per_round = [rd["best"] for rd in rounds if rd.get("best")]
    round_scores = [c["score"] for c in best_per_round]
    round_prices = [c["price"] for c in best_per_round]

    uniq: dict[int, dict] = {}
    for r in all_hits:
        oid = r["id"]
        if oid not in uniq or r["score"] > uniq[oid]["score"]:
            uniq[oid] = r
    best_unique = sorted(uniq.values(), key=lambda r: (-r["score"], r["price"]))[:30]

    return {
        "score_formula": "(gpu_mem_bw * frac) / (price_usd * 100)  # GB/s-effective per cent $/hr",
        "all_hits_price": price_stats(prices),
        "all_hits_score": price_stats(scores),
        "best_per_round_score": price_stats(round_scores),
        "best_per_round_price": price_stats(round_prices),
        "by_location_price": {
            k: price_stats(v)
            for k, v in sorted(by_loc.items(), key=lambda kv: min(kv[1]))
        },
        "by_location_best_score": {
            k: {"n": len(v), "max": max(v), "median": statistics.median(v), "mean": statistics.mean(v)}
            for k, v in sorted(score_by_loc.items(), key=lambda kv: -max(kv[1]))
        },
        "by_gpu": {
            k: price_stats(v)
            for k, v in sorted(by_gpu.items(), key=lambda kv: min(kv[1]))
        },
        "by_verified": {k: price_stats(v) for k, v in by_ver.items()},
        "by_eff_vram_bucket": {
            k: price_stats(by_eff_bucket[k])
            for k in ("4-6G", "6-8G", "8-12G", "12-16G", "16G+")
            if k in by_eff_bucket
        },
        "top_offer_frequency": Counter(c["id"] for c in best_per_round).most_common(10),
        "best_score_unique_sample": [
            {
                "id": r["id"],
                "score": r["score"],
                "bw_frac": r["bw_frac"],
                "price": r["price"],
                "gpu": r["gpu"],
                "frac": r["frac"],
                "eff_vram": r["eff_vram"],
                "ram_g": r["ram_g"],
                "gpu_mem_bw": r["gpu_mem_bw"],
                "pcie_bw": r["pcie_bw"],
                "inet_down": r["inet_down"],
                "inet_up": r["inet_up"],
                "static_ip": r["static_ip"],
                "verified": r["verified"],
                "reli": r["reli"],
                "loc": r["loc"],
            }
            for r in best_unique
        ],
    }


def fmt(r: dict) -> str:
    return (
        f"{r['id']} score={r['score']:.2f} "
        f"(bw*frac={r['bw_frac']:.0f}/{r['price']*100:.2f}c) "
        f"${r['price']:.4f} {r['gpu']} frac={r['frac']:.3f} "
        f"eff={r['eff_vram']:.1f}G ram={r['ram_g']:.0f}G "
        f"gmem_bw={r['gpu_mem_bw']:.0f} pcie={r['pcie_bw']:.1f} "
        f"down={r['inet_down']:.0f} up={r['inet_up']:.0f} "
        f"static={r['static_ip']} ver={r['verified']} "
        f"reli={r['reli']:.2f} {r['loc']}"
    )


def main() -> int:
    rounds = []
    all_hits: list[dict] = []

    print(f"template={TEMPLATE}")
    print(f"query={QUERY}")
    print(
        f"fit: eff>={MIN_EFF} ram>={MIN_RAM} disk>={MIN_DISK} static_ip=True | "
        "sort: (gpu_mem_bw*frac)/cents\n"
    )

    for i in range(1, N + 1):
        rows = [norm(o) for o in search()]
        hits = fit(rows)
        top = hits[:10]
        print(f"=== search {i}/{N}  pool={len(rows)}  fit={len(hits)} ===")
        for r in top:
            print(f"  {fmt(r)}")
        if not hits:
            print("  (none)")
        best = hits[0] if hits else None
        rounds.append(
            {
                "i": i,
                "pool": len(rows),
                "fit_n": len(hits),
                "best": best,
                "top": top,
            }
        )
        all_hits.extend(hits)
        if i < N:
            time.sleep(2)

    analysis = analyze(all_hits, rounds)
    summary = {
        "template": TEMPLATE,
        "query": QUERY,
        "filters": (
            f"eff_vram>={MIN_EFF} AND ram_g>={MIN_RAM} AND disk>={MIN_DISK} "
            "AND static_ip=True (gpu_frac applied)"
        ),
        "rank": "(gpu_mem_bw * frac) / (dph_total * 100)",
        "n_searches": N,
        "rounds_with_fit": sum(1 for r in rounds if r["best"]),
        "analysis": analysis,
        "rounds": rounds,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n========== SCORE ANALYSIS ==========")
    print("formula:", analysis["score_formula"])
    print("best_per_round_score:", json.dumps(analysis["best_per_round_score"], indent=2))
    print("best_per_round_price:", json.dumps(analysis["best_per_round_price"], indent=2))
    print("all_hits_score:", json.dumps(analysis["all_hits_score"], indent=2))
    print("\nby_location (by max score):")
    for loc, st in list(analysis["by_location_best_score"].items())[:12]:
        print(f"  {loc}: n={st['n']} max_score={st['max']:.2f} med={st['median']:.2f}")
    print("\ntop by score (bw*frac per cent):")
    for r in analysis["best_score_unique_sample"][:12]:
        print(f"  {fmt(r)}")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
