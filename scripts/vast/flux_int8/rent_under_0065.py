#!/usr/bin/env python3
"""Loop search+rent until an offer under MAX_DPH is secured."""
from __future__ import annotations

import json
import subprocess
import time

TEMPLATE = "1ec7dac83a206e2101e0fa6ec06d2cdf"
DISK = "20"
MAX_DPH = 0.065
QUERY = "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True"
MAX_ATTEMPTS = 40
SLEEP_S = 3


def search():
    cmd = [
        "vastai",
        "search",
        "offers",
        QUERY,
        "--order",
        "dph_total",
        "--limit",
        "40",
        "--raw",
        "-n",
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


def normalize(o):
    vram = float(o.get("gpu_ram") or 0)
    vram_g = vram / 1024 if vram > 64 else vram
    ram = float(o.get("cpu_ram") or 0)
    ram_g = ram / 1024 if ram > 512 else ram
    price = float(o.get("dph_total") or 99)
    return price, vram_g, ram_g, o


def candidates(offers):
    rows = []
    for o in offers:
        price, vram_g, ram_g, o = normalize(o)
        if price >= MAX_DPH:
            continue
        if vram_g <= 11.5 or ram_g <= 30:
            continue
        rows.append((price, vram_g, ram_g, o))
    rows.sort(key=lambda x: x[0])
    return rows


def create(offer_id: int):
    cmd = [
        "vastai",
        "create",
        "instance",
        str(offer_id),
        "--template_hash",
        TEMPLATE,
        "--disk",
        DISK,
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


def main() -> int:
    tried = set()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rows = candidates(search())
        cheap = [r for r in rows if r[3]["id"] not in tried]
        print(f"\n[{attempt}/{MAX_ATTEMPTS}] under ${MAX_DPH}: {len(rows)} fit, {len(cheap)} untried")
        for price, vram_g, ram_g, o in rows[:8]:
            mark = "tried" if o["id"] in tried else "new"
            print(
                f"  {o['id']} ${price:.4f} {o.get('gpu_name')} "
                f"vram={vram_g:.1f}G ram={ram_g:.0f}G {o.get('geolocation')} [{mark}]"
            )
        if not cheap:
            print(f"none left; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            tried.clear()  # market refreshed; allow retry
            continue

        price, vram_g, ram_g, o = cheap[0]
        oid = o["id"]
        tried.add(oid)
        print(f"renting {oid} @ ${price:.4f} ...")
        res = create(oid)
        print(json.dumps({k: res.get(k) for k in ("success", "new_contract", "error", "msg", "status_code") if k in res or res.get(k) is not None}))
        if res.get("success") and res.get("new_contract"):
            print(f"SUCCESS instance={res['new_contract']} offer={oid} ${price:.4f}")
            return 0
        time.sleep(1)

    print("FAILED: no offer under threshold secured")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
