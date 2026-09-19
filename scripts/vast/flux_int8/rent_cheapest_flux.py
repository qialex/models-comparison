#!/usr/bin/env python3
"""Find cheapest Flux-fit offer and rent with template ASAP."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import redact_create  # noqa: E402

TEMPLATE = "1ec7dac83a206e2101e0fa6ec06d2cdf"
DISK = "20"


def search():
    q = "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True"
    cmd = ["vastai", "search", "offers", q, "--order", "dph_total", "--limit", "30", "--raw", "-n"]
    return json.loads(subprocess.check_output(cmd, text=True))


def fit(offers):
    rows = []
    for o in offers:
        vram = float(o.get("gpu_ram") or 0)
        vram_g = vram / 1024 if vram > 64 else vram
        ram = float(o.get("cpu_ram") or 0)
        ram_g = ram / 1024 if ram > 512 else ram
        if vram_g <= 11.5 or ram_g <= 30:
            continue
        rows.append((float(o.get("dph_total") or 99), o))
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
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return json.loads(out)


def main() -> int:
    preferred = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else []
    offers = fit(search())
    print("top candidates:")
    for price, o in offers[:8]:
        print(
            f"  {o['id']} ${price:.4f} {o.get('gpu_name')} "
            f"vram={(o.get('gpu_ram') or 0)/1024 if (o.get('gpu_ram') or 0)>64 else o.get('gpu_ram')} "
            f"ram={(o.get('cpu_ram') or 0)/1024 if (o.get('cpu_ram') or 0)>512 else (o.get('cpu_ram') or 0)/1} "
            f"{o.get('geolocation')}"
        )

    tried = []
    # try preferred first if still present, then cheapest
    ids = []
    for pid in preferred:
        if any(o["id"] == pid for _, o in offers):
            ids.append(pid)
    for _, o in offers:
        if o["id"] not in ids:
            ids.append(o["id"])

    for oid in ids[:6]:
        print(f"\ntrying offer {oid} ...")
        try:
            res = create(oid)
        except subprocess.CalledProcessError as e:
            print("create failed:", e.output)
            continue
        print(json.dumps(redact_create(res), indent=2)[:1500])
        if res.get("error") or res.get("success") is False:
            msg = res.get("msg") or res
            print("not available:", msg)
            tried.append((oid, str(msg)))
            continue
        print("SUCCESS")
        return 0

    print("all attempts failed", tried)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
