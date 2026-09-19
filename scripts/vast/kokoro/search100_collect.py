#!/usr/bin/env python3
"""100 Vast searches for Kokoro-class boxes. Collect max info. NO RENT.

Query (template-aligned): gpu_ram>=4 cpu_ram>=4 disk_space>=8 1xGPU rentable.
Stores per-round snapshots + unique-offer ledger + light summary.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug  # noqa: E402

OUT_DIR = cache_debug("kokoro-82m", "_debug") / "search100"
N = 100
SLEEP_S = 2
LIMIT = 100
# Match kokoro template; -n includes unverified
QUERY = "gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 rentable=True"

# Fields worth keeping from raw offer (drop huge/useless)
KEEP = [
    "id",
    "machine_id",
    "host_id",
    "bundle_id",
    "gpu_name",
    "gpu_ram",
    "gpu_total_ram",
    "gpu_frac",
    "num_gpus",
    "gpu_mem_bw",
    "gpu_arch",
    "gpu_lanes",
    "gpu_max_power",
    "gpu_max_temp",
    "compute_cap",
    "cuda_max_good",
    "driver_version",
    "driver_vers",
    "cpu_name",
    "cpu_ram",
    "cpu_cores",
    "cpu_cores_effective",
    "cpu_ghz",
    "cpu_arch",
    "has_avx",
    "mobo_name",
    "disk_space",
    "disk_name",
    "disk_bw",
    "nw_disk_avg_bw",
    "nw_disk_min_bw",
    "nw_disk_max_bw",
    "pcie_bw",
    "pci_gen",
    "bw_nvlink",
    "inet_down",
    "inet_up",
    "inet_down_cost",
    "inet_up_cost",
    "internet_down_cost_per_tb",
    "internet_up_cost_per_tb",
    "dph_base",
    "dph_total",
    "dph_total_adj",
    "discounted_dph_total",
    "discounted_hourly",
    "discount_rate",
    "storage_cost",
    "storage_total_cost",
    "vram_costperhour",
    "min_bid",
    "is_bid",
    "dlperf",
    "dlperf_per_dphtotal",
    "total_flops",
    "flops_per_dphtotal",
    "reliability",
    "reliability2",
    "reliability_mult",
    "expected_reliability",
    "target_reliability",
    "static_ip",
    "direct_port_count",
    "geolocation",
    "geolocode",
    "public_ipaddr",
    "hostname",
    "hosting_type",
    "verification",
    "vericode",
    "rentable",
    "rented",
    "external",
    "duration",
    "start_date",
    "end_date",
    "time_remaining",
    "os_version",
    "vms_enabled",
    "is_vm_deverified",
    "score",
    "rn",
]


def search() -> list[dict]:
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
                str(LIMIT),
                "--raw",
                "-n",
            ],
            text=True,
        )
    )


def _gb(val: float, threshold: float) -> float:
    return val / 1024 if val > threshold else val


def derive(o: dict) -> dict:
    vram = float(o.get("gpu_ram") or 0)
    vram_g = _gb(vram, 64)
    frac = float(o["gpu_frac"] if o.get("gpu_frac") is not None else 1.0)
    ram = float(o.get("cpu_ram") or 0)
    ram_g = _gb(ram, 512)
    price = float(o.get("dph_total") or 99)
    bw = float(o.get("gpu_mem_bw") or 0)
    cents = price * 100.0
    return {
        "vram_g": round(vram_g, 3),
        "ram_g": round(ram_g, 2),
        "frac": round(frac, 4),
        "eff_vram_g": round(frac * vram_g, 3),
        "price": price,
        "bw_frac": round(bw * frac, 2),
        "score_bw_per_cent": round((bw * frac) / cents, 4) if cents > 0 else None,
        "verified": bool(
            o.get("verified")
            or str(o.get("verification") or "").lower() == "verified"
        ),
        "reli": float(o.get("reliability2") or o.get("reliability") or 0),
        "static_ip": bool(o.get("static_ip")),
        "ports": int(o.get("direct_port_count") or 0),
        "loc": o.get("geolocation") or "",
        "gpu": o.get("gpu_name"),
        "machine": int(o.get("machine_id") or 0),
        "host": int(o.get("host_id") or 0),
        "kokoro_core_ok": (frac * vram_g) >= 4.0 and ram_g >= 4.0 and float(o.get("disk_space") or 0) >= 8,
    }


def slim(o: dict) -> dict:
    row = {k: o.get(k) for k in KEEP}
    row["derived"] = derive(o)
    return row


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rounds_dir = OUT_DIR / "rounds"
    rounds_dir.mkdir(exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    unique: dict[int, dict] = {}
    round_summaries: list[dict] = []

    print(f"query={QUERY}")
    print(f"n={N} limit={LIMIT} sleep={SLEEP_S}s out={OUT_DIR}")

    for i in range(1, N + 1):
        t0 = time.perf_counter()
        try:
            raw = search()
        except Exception as e:
            print(f"[{i}/{N}] SEARCH FAIL {e}")
            round_summaries.append({"i": i, "error": str(e), "n": 0})
            time.sleep(SLEEP_S)
            continue

        rows = [slim(o) for o in raw]
        ts = datetime.now(timezone.utc).isoformat()
        elapsed = round(time.perf_counter() - t0, 3)

        for r in rows:
            oid = r["id"]
            d = r["derived"]
            if oid not in unique:
                unique[oid] = {
                    "first_seen": ts,
                    "last_seen": ts,
                    "seen_rounds": 1,
                    "price_min": d["price"],
                    "price_max": d["price"],
                    "price_last": d["price"],
                    "offer": r,
                }
            else:
                u = unique[oid]
                u["last_seen"] = ts
                u["seen_rounds"] += 1
                u["price_min"] = min(u["price_min"], d["price"])
                u["price_max"] = max(u["price_max"], d["price"])
                u["price_last"] = d["price"]
                u["offer"] = r  # refresh snapshot

        core = [r for r in rows if r["derived"]["kokoro_core_ok"]]
        prices = [r["derived"]["price"] for r in rows]
        core_prices = [r["derived"]["price"] for r in core]
        static_prices = [r["derived"]["price"] for r in core if r["derived"]["static_ip"]]
        verified_prices = [r["derived"]["price"] for r in core if r["derived"]["verified"]]

        cheapest = min(rows, key=lambda r: r["derived"]["price"]) if rows else None
        cheapest_core = min(core, key=lambda r: r["derived"]["price"]) if core else None
        best_score = (
            max(core, key=lambda r: r["derived"]["score_bw_per_cent"] or 0) if core else None
        )

        summary = {
            "i": i,
            "ts": ts,
            "search_s": elapsed,
            "n": len(rows),
            "n_core": len(core),
            "n_static_core": len(static_prices),
            "n_verified_core": len(verified_prices),
            "price_all": price_stats(prices),
            "price_core": price_stats(core_prices),
            "price_static_core": price_stats(static_prices),
            "price_verified_core": price_stats(verified_prices),
            "cheapest": {
                "id": cheapest["id"],
                **cheapest["derived"],
            }
            if cheapest
            else None,
            "cheapest_core": {
                "id": cheapest_core["id"],
                **cheapest_core["derived"],
            }
            if cheapest_core
            else None,
            "best_bw_score_core": {
                "id": best_score["id"],
                **best_score["derived"],
            }
            if best_score
            else None,
            "gpu_counts": Counter(r["derived"]["gpu"] for r in core).most_common(8),
            "loc_counts": Counter(r["derived"]["loc"] for r in core).most_common(8),
        }
        round_summaries.append(summary)

        (rounds_dir / f"round_{i:03d}.json").write_text(
            json.dumps({"summary": summary, "offers": rows}, indent=2) + "\n",
            encoding="utf-8",
        )

        cc = summary["cheapest_core"]
        print(
            f"[{i}/{N}] n={summary['n']} core={summary['n_core']} "
            f"core_min=${(summary['price_core'].get('min') or 0):.4f} "
            f"cheapest_core="
            + (
                f"{cc['id']} ${cc['price']:.4f} {cc['gpu']} "
                f"eff={cc['eff_vram_g']}G frac={cc['frac']} {cc['loc']}"
                if cc
                else "(none)"
            )
        )

        if i < N:
            time.sleep(SLEEP_S)

    # unique ledger
    (OUT_DIR / "unique_offers.json").write_text(
        json.dumps(unique, indent=2) + "\n", encoding="utf-8"
    )

    # aggregate
    core_mins = [
        s["price_core"]["min"]
        for s in round_summaries
        if s.get("price_core", {}).get("n")
    ]
    cheapest_ids = Counter(
        s["cheapest_core"]["id"] for s in round_summaries if s.get("cheapest_core")
    )

    # final unique core market snapshot
    core_unique = [
        u
        for u in unique.values()
        if u["offer"]["derived"]["kokoro_core_ok"]
    ]
    core_unique.sort(key=lambda u: u["price_last"])

    meta = {
        "started": started,
        "ended": datetime.now(timezone.utc).isoformat(),
        "query": QUERY,
        "n_searches": N,
        "limit": LIMIT,
        "sleep_s": SLEEP_S,
        "note": "NO RENT — research only. kokoro_core_ok = eff_vram>=4 AND ram>=4 AND disk>=8",
        "rounds_ok": sum(1 for s in round_summaries if s.get("n")),
        "unique_offers": len(unique),
        "unique_core": len(core_unique),
        "cheapest_core_min_over_rounds": min(core_mins) if core_mins else None,
        "cheapest_core_max_over_rounds": max(core_mins) if core_mins else None,
        "cheapest_core_median_over_rounds": statistics.median(core_mins) if core_mins else None,
        "top_cheapest_core_frequency": cheapest_ids.most_common(15),
        "cheapest_unique_core_now": [
            {
                "id": u["offer"]["id"],
                "seen_rounds": u["seen_rounds"],
                "price_min": u["price_min"],
                "price_max": u["price_max"],
                "price_last": u["price_last"],
                **u["offer"]["derived"],
            }
            for u in core_unique[:40]
        ],
        "round_summaries": round_summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("\n========== DONE ==========")
    print(json.dumps({k: meta[k] for k in meta if k != "round_summaries"}, indent=2))
    print("wrote", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
