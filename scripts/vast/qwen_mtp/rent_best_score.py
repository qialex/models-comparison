#!/usr/bin/env python3
"""Rent best qwen fit by (gpu_mem_bw*frac)/cents. static_ip, eff>=4, etc."""
from __future__ import annotations

import json
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import redact_create  # noqa: E402

ONSTART = str(Path(__file__).resolve().parents[3] / "vast" / "qwen35-4b-mtp-ngram" / "onstart.sh")
DISK = "8"
IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
ENV = "-p 8080:8080 -e PORT=8080 -e N_CTX=8192 -e N_PARALLEL=1 -e N_PREDICT=1536 -e N_GPU_LAYERS=99"
QUERY = "gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 rentable=True static_ip=True"
MIN_EFF = 4.0
MIN_RAM = 4.0
PREFER_ID = 49061607


def search():
    return json.loads(
        subprocess.check_output(
            ["vastai", "search", "offers", QUERY, "--order", "dph_total", "--limit", "100", "--raw", "-n"],
            text=True,
        )
    )


def norm(o):
    v = float(o.get("gpu_ram") or 0)
    vg = v / 1024 if v > 64 else v
    frac = float(o["gpu_frac"] if o.get("gpu_frac") is not None else 1)
    ram = float(o.get("cpu_ram") or 0)
    rg = ram / 1024 if ram > 512 else ram
    price = float(o.get("dph_total") or 99)
    bw = float(o.get("gpu_mem_bw") or 0)
    score = (bw * frac) / (price * 100) if price > 0 else 0
    return {
        "id": o["id"],
        "price": price,
        "gpu": o.get("gpu_name"),
        "frac": frac,
        "eff": frac * vg,
        "ram": rg,
        "bw": bw,
        "score": score,
        "loc": o.get("geolocation") or "",
        "static": bool(o.get("static_ip")),
    }


def fit(rows):
    return sorted(
        [r for r in rows if r["eff"] >= MIN_EFF and r["ram"] >= MIN_RAM and r["static"]],
        key=lambda r: (-r["score"], r["price"]),
    )


def create(oid: int):
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


def main():
    print("onstart", ONSTART, "exists", Path(ONSTART).exists())
    tried: set[int] = set()
    for attempt in range(1, 41):
        hits = [r for r in fit(norm(o) for o in search()) if r["id"] not in tried]
        prefer = next((r for r in hits if r["id"] == PREFER_ID), None)
        target = prefer or (hits[0] if hits else None)
        print(f"\n[{attempt}/40] fit_untried={len(hits)} tried={len(tried)}")
        for r in hits[:5]:
            mark = "TARGET" if target and r["id"] == target["id"] else ""
            print(
                f"  {r['id']} score={r['score']:.1f} ${r['price']:.4f} {r['gpu']} "
                f"frac={r['frac']} eff={r['eff']:.1f} {r['loc']} {mark}"
            )
        if not target:
            print("none; sleep")
            time.sleep(3)
            if attempt % 10 == 0:
                tried.clear()
            continue
        if target["id"] != PREFER_ID:
            print(f"prefer {PREFER_ID} gone; renting best score {target['id']}")
        print(f"RENTING {target['id']} ...")
        tried.add(target["id"])
        res = create(target["id"])
        print(json.dumps(redact_create(res)))
        if res.get("success") and res.get("new_contract"):
            print(f"SUCCESS instance={res['new_contract']} offer={target['id']}")
            return 0
        time.sleep(1)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
