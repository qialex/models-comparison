#!/usr/bin/env python3
"""Monte-Carlo / ablation study of Vast filters for Flux INT8 fit. Research only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug  # noqa: E402

import itertools
import json
import random
import subprocess
from collections import Counter

OUT = cache_debug("flux2-klein-9b-kv-int8", "_debug") / "montecarlo_results.json"
MIN_EFF = 11.5
MIN_RAM = 30
MIN_PORTS = 2
MIN_RELI = 0.90
PRICES = [0.05, 0.055, 0.065, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.40]


def fetch_pool():
    # Broad pool — no verified/static — then filter client-side
    q = "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True"
    offers = json.loads(
        subprocess.check_output(
            ["vastai", "search", "offers", q, "--order", "dph_total", "--limit", "200", "--raw", "-n"],
            text=True,
        )
    )
    return offers


def norm(o):
    vram = float(o.get("gpu_ram") or 0)
    vram_g = vram / 1024 if vram > 64 else vram
    frac = float(o.get("gpu_frac") if o.get("gpu_frac") is not None else 1.0)
    eff = frac * vram_g
    ram = float(o.get("cpu_ram") or 0)
    ram_g = ram / 1024 if ram > 512 else ram
    price = float(o.get("dph_total") or 99)
    ports = int(o.get("direct_port_count") or 0)
    reli = float(o.get("reliability2") or o.get("reliability") or 0)
    return {
        "id": o.get("id"),
        "price": price,
        "gpu": o.get("gpu_name"),
        "vram_g": round(vram_g, 2),
        "frac": round(frac, 4),
        "eff_vram": round(eff, 2),
        "ram_g": round(ram_g, 1),
        "ports": ports,
        "verified": bool(o.get("verified")),
        "static_ip": bool(o.get("static_ip")),
        "reli": round(reli, 3),
        "loc": o.get("geolocation") or "",
    }


def passes(r, *, max_price, need_eff, need_ram, need_ports, need_reli, need_static, need_verified):
    if r["price"] >= max_price:
        return False
    if need_eff and r["eff_vram"] < MIN_EFF:
        return False
    if need_ram and r["ram_g"] <= MIN_RAM:
        return False
    if need_ports and r["ports"] < MIN_PORTS:
        return False
    if need_reli and r["reli"] <= MIN_RELI:
        return False
    if need_static and not r["static_ip"]:
        return False
    if need_verified and not r["verified"]:
        return False
    return True


def main():
    raw = fetch_pool()
    rows = [norm(o) for o in raw]
    print(f"pool={len(rows)}")

    # Base: price-only counts
    price_curve = []
    for p in PRICES:
        n = sum(1 for r in rows if r["price"] < p)
        price_curve.append({"max_dph": p, "n": n})

    # Ablation under each price: start from full Flux-ish, drop one constraint
    ablations = []
    full = dict(
        need_eff=True,
        need_ram=True,
        need_ports=True,
        need_reli=True,
        need_static=True,
        need_verified=False,
    )
    variants = [
        ("full: eff+ram+ports+reli+static", full),
        ("drop static", {**full, "need_static": False}),
        ("drop reli", {**full, "need_reli": False}),
        ("drop ports", {**full, "need_ports": False}),
        ("drop effVRAM (use listed vram proxy)", {**full, "need_eff": False}),
        ("only price+eff+ram", dict(need_eff=True, need_ram=True, need_ports=False, need_reli=False, need_static=False, need_verified=False)),
        ("price+eff+ram+ports", dict(need_eff=True, need_ram=True, need_ports=True, need_reli=False, need_static=False, need_verified=False)),
        ("price+eff+ram+static", dict(need_eff=True, need_ram=True, need_ports=False, need_reli=False, need_static=True, need_verified=False)),
        ("price+eff+ram+reli", dict(need_eff=True, need_ram=True, need_ports=False, need_reli=True, need_static=False, need_verified=False)),
        ("price+eff+ram+verified", dict(need_eff=True, need_ram=True, need_ports=False, need_reli=False, need_static=False, need_verified=True)),
        ("verified+static+reli+eff+ram+ports", dict(need_eff=True, need_ram=True, need_ports=True, need_reli=True, need_static=True, need_verified=True)),
    ]

    for label, flags in variants:
        series = []
        for p in PRICES:
            n = sum(1 for r in rows if passes(r, max_price=p, **flags))
            series.append({"max_dph": p, "n": n})
        ablations.append({"label": label, "series": series})

    # Monte Carlo: random subsets of optional flags at fixed prices
    optional = ["need_ports", "need_reli", "need_static", "need_verified"]
    mc = []
    rng = random.Random(42)
    for p in [0.055, 0.10, 0.15, 0.25]:
        counts = Counter()
        for _ in range(200):
            flags = {
                "need_eff": True,
                "need_ram": True,
                "need_ports": False,
                "need_reli": False,
                "need_static": False,
                "need_verified": False,
            }
            chosen = []
            for k in optional:
                if rng.random() < 0.5:
                    flags[k] = True
                    chosen.append(k.replace("need_", ""))
            n = sum(1 for r in rows if passes(r, max_price=p, **flags))
            key = "+".join(sorted(chosen)) if chosen else "none"
            counts[key] += n
        # average n per combo
        combo_avg = []
        seen = Counter()
        sums = Counter()
        for _ in range(200):
            flags = {
                "need_eff": True,
                "need_ram": True,
                "need_ports": False,
                "need_reli": False,
                "need_static": False,
                "need_verified": False,
            }
            chosen = []
            for k in optional:
                if rng.random() < 0.5:
                    flags[k] = True
                    chosen.append(k.replace("need_", ""))
            n = sum(1 for r in rows if passes(r, max_price=p, **flags))
            key = "+".join(sorted(chosen)) if chosen else "(core only)"
            seen[key] += 1
            sums[key] += n
        for key in sorted(sums.keys(), key=lambda k: -sums[k] / max(seen[k], 1)):
            combo_avg.append(
                {
                    "combo": key,
                    "avg_n": round(sums[key] / seen[key], 2),
                    "samples": seen[key],
                }
            )
        mc.append({"max_dph": p, "combos": combo_avg[:12]})

    # Exhaustive optional flag grid at key prices (2^4 = 16)
    grid = []
    for p in [0.055, 0.10, 0.15]:
        cells = []
        for bits in itertools.product([False, True], repeat=4):
            ports, reli, static, verified = bits
            flags = dict(
                need_eff=True,
                need_ram=True,
                need_ports=ports,
                need_reli=reli,
                need_static=static,
                need_verified=verified,
            )
            n = sum(1 for r in rows if passes(r, max_price=p, **flags))
            label = []
            if ports:
                label.append("ports")
            if reli:
                label.append("reli")
            if static:
                label.append("static")
            if verified:
                label.append("verified")
            cells.append({"flags": "+".join(label) or "core", "n": n})
        cells.sort(key=lambda c: -c["n"])
        grid.append({"max_dph": p, "cells": cells})

    # Killer constraint under $0.15 among core-fit (eff+ram)
    core = [r for r in rows if r["price"] < 0.15 and r["eff_vram"] >= MIN_EFF and r["ram_g"] > MIN_RAM]
    killers = {
        "core_eff_ram_under_0.15": len(core),
        "lose_to_ports": sum(1 for r in core if r["ports"] < MIN_PORTS),
        "lose_to_reli": sum(1 for r in core if r["reli"] <= MIN_RELI),
        "lose_to_static": sum(1 for r in core if not r["static_ip"]),
        "lose_to_verified": sum(1 for r in core if not r["verified"]),
        "frac_lt_1": sum(1 for r in core if r["frac"] < 1.0),
        "frac_eq_1": sum(1 for r in core if r["frac"] >= 1.0),
    }

    # Cheapest survivors for interesting recipes
    recipes = {}
    for label, flags in variants[:8]:
        hits = [r for r in rows if passes(r, max_price=99, **flags)]
        hits.sort(key=lambda r: r["price"])
        recipes[label] = hits[:5]

    out = {
        "pool_n": len(rows),
        "query": "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True",
        "thresholds": {
            "min_eff_vram": MIN_EFF,
            "min_ram_gb": MIN_RAM,
            "min_ports": MIN_PORTS,
            "min_reliability": MIN_RELI,
        },
        "price_curve": price_curve,
        "ablations": ablations,
        "monte_carlo": mc,
        "flag_grid": grid,
        "killers_under_0.15_core": killers,
        "cheapest_by_recipe": {
            k: [
                {
                    "id": r["id"],
                    "price": r["price"],
                    "gpu": r["gpu"],
                    "eff_vram": r["eff_vram"],
                    "frac": r["frac"],
                    "ram_g": r["ram_g"],
                    "ports": r["ports"],
                    "reli": r["reli"],
                    "static_ip": r["static_ip"],
                    "verified": r["verified"],
                    "loc": r["loc"],
                }
                for r in v
            ]
            for k, v in recipes.items()
        },
        "pool_sample_cheapest": sorted(rows, key=lambda r: r["price"])[:20],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)

    # quick stdout summary
    print("\n=== counts under price for key recipes ===")
    for label, flags in variants:
        ns = [sum(1 for r in rows if passes(r, max_price=p, **flags)) for p in (0.055, 0.10, 0.15)]
        print(f"{label}: <0.055={ns[0]} <0.10={ns[1]} <0.15={ns[2]}")
    print("\nkillers under $0.15 core:", killers)


if __name__ == "__main__":
    main()
