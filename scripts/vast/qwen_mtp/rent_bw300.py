#!/usr/bin/env python3
"""Rent one Qwen3.5-4B MTP ngram instance matching hard filters.

Criteria (all required):
  - static_ip=True
  - dph_total < 0.05
  - disk_space >= 8  (create with --disk 8)
  - (gpu_mem_bw * gpu_frac) > 300
  - num_gpus=1, rentable
  - eff VRAM >= 4GB, system RAM >= 4GB

Rank: highest (bw * frac) / cents, then cheapest.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug, redact_create  # noqa: E402

ONSTART = str(
    Path(__file__).resolve().parents[3] / "vast" / "qwen35-4b-mtp-ngram" / "onstart.sh"
)
DISK = "8"
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
ENV = (
    "-p 8080:8080 -e PORT=8080 -e N_CTX=8192 -e N_PARALLEL=1 "
    "-e N_PREDICT=1536 -e N_GPU_LAYERS=99"
)
QUERY = (
    "gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 "
    "rentable=True static_ip=True dph_total<0.05"
)

MAX_PRICE = 0.05
MIN_BW_FRAC = 300.0  # gpu_mem_bw * frac
MIN_EFF = 4.0
MIN_RAM = 4.0
MIN_DISK = 8.0

MAX_ATTEMPTS = 900  # ~1h at SLEEP_S=4
SLEEP_S = 4
OUT_LOG = cache_debug("qwen35-4b-mtp-ngram", "_debug") / "rent_bw300.log"


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
                "100",
                "--raw",
                "-n",
            ],
            text=True,
        )
    )


def _gb(val: float, threshold: float) -> float:
    return val / 1024 if val > threshold else val


def norm(o: dict) -> dict:
    vram_g = _gb(float(o.get("gpu_ram") or 0), 64)
    frac = float(o["gpu_frac"] if o.get("gpu_frac") is not None else 1.0)
    ram_g = _gb(float(o.get("cpu_ram") or 0), 512)
    price = float(o.get("dph_total") or 99)
    bw = float(o.get("gpu_mem_bw") or 0)
    bw_frac = bw * frac
    cents = price * 100.0
    score = (bw * frac) / cents if cents > 0 else 0.0
    return {
        "id": o["id"],
        "price": price,
        "gpu": o.get("gpu_name"),
        "frac": frac,
        "eff": frac * vram_g,
        "ram": ram_g,
        "disk": float(o.get("disk_space") or 0),
        "bw": bw,
        "bw_frac": bw_frac,
        "score": round(score, 2),
        "static": bool(o.get("static_ip")),
        "loc": o.get("geolocation") or "",
        "reli": float(o.get("reliability2") or o.get("reliability") or 0),
    }


def fit(r: dict) -> bool:
    return (
        r["static"]
        and r["price"] < MAX_PRICE
        and r["bw_frac"] > MIN_BW_FRAC
        and r["eff"] >= MIN_EFF
        and r["ram"] >= MIN_RAM
        and r["disk"] >= MIN_DISK
    )


def fmt(r: dict) -> str:
    return (
        f"{r['id']} ${r['price']:.4f} {r['gpu']} "
        f"bw={r['bw']:.0f} frac={r['frac']:.3f} bw*frac={r['bw_frac']:.1f} "
        f"eff={r['eff']:.1f}G score={r['score']:.1f} "
        f"reli={r['reli']:.3f} static={r['static']} {r['loc']}"
    )


def create(oid: int) -> dict:
    cmd = [
        "vastai",
        "create",
        "instance",
        str(oid),
        "--image",
        IMAGE,
        "--disk",
        DISK,
        "--ssh",
        "--direct",
        "--env",
        ENV,
        "--onstart",
        ONSTART,
        "--cancel-unavail",
        "--raw",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output or str(e)
    try:
        return json.loads(out)
    except Exception:
        return {"error": True, "msg": out}


def log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    print(line, end="", flush=True)
    with OUT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    OUT_LOG.write_text("", encoding="utf-8")
    log(
        "rent_bw300 start\n"
        f"  static_ip + price<{MAX_PRICE} + (gpu_mem_bw*frac)>{MIN_BW_FRAC}\n"
        f"  disk_create={DISK} eff>={MIN_EFF} ram>={MIN_RAM}\n"
        f"  onstart={ONSTART} exists={Path(ONSTART).exists()}\n"
        f"  query={QUERY}\n"
    )
    tried: set[int] = set()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            rows = [norm(o) for o in search()]
        except Exception as e:
            log(f"[{attempt}/{MAX_ATTEMPTS}] search fail: {e}; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            continue

        hits = sorted(
            [r for r in rows if fit(r) and r["id"] not in tried],
            key=lambda r: (-r["score"], r["price"]),
        )
        peek = sorted(rows, key=lambda r: (-r["score"], r["price"]))[:5]

        log(
            f"\n[{attempt}/{MAX_ATTEMPTS}] offers={len(rows)} "
            f"fit={len(hits)} tried={len(tried)}"
        )
        for r in peek:
            tag = "FIT" if fit(r) else "miss"
            log(f"  peek [{tag}] {fmt(r)}")

        if not hits:
            log(f"  none fit; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            if attempt % 10 == 0:
                tried.clear()
            continue

        target = hits[0]
        tried.add(target["id"])
        log(f"  RENTING {fmt(target)}")
        res = create(target["id"])
        log(json.dumps(redact_create(res)))
        if res.get("success") and res.get("new_contract"):
            log(
                f"SUCCESS instance={res['new_contract']} "
                f"offer={target['id']} ${target['price']:.4f}"
            )
            return 0
        time.sleep(2)

    log("FAILED — no rent within window")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
