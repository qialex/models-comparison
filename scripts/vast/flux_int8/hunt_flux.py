#!/usr/bin/env python3
"""Hunt+rent Flux INT8: static IP, reliability>0.9, effVRAM>=11.5, under MAX_DPH."""
from __future__ import annotations

import json
import subprocess
import time

TEMPLATE = "1ec7dac83a206e2101e0fa6ec06d2cdf"
DISK = "20"
MAX_DPH = 0.055
MIN_EFF_VRAM = 11.5
MIN_RAM = 30
MIN_PORTS = 2
MIN_RELI = 0.90
QUERY = (
    "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True "
    "direct_port_count>=2 static_ip=True reliability>0.90"
)
MAX_ATTEMPTS = 80
SLEEP_S = 5


def search():
    cmd = [
        "vastai",
        "search",
        "offers",
        QUERY,
        "--order",
        "dph_total",
        "--limit",
        "60",
        "--raw",
        "-n",
    ]
    return json.loads(subprocess.check_output(cmd, text=True))


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
    static_ip = bool(o.get("static_ip"))
    return price, vram_g, frac, eff, ram_g, ports, reli, static_ip, o


def candidates(offers):
    rows = []
    for o in offers:
        price, vram_g, frac, eff, ram_g, ports, reli, static_ip, o = norm(o)
        if price >= MAX_DPH:
            continue
        if eff < MIN_EFF_VRAM or ram_g <= MIN_RAM:
            continue
        if ports < MIN_PORTS:
            continue
        if not static_ip or reli <= MIN_RELI:
            continue
        rows.append((price, vram_g, frac, eff, ram_g, ports, reli, o))
    rows.sort(key=lambda x: (x[0], -x[3]))
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
    tried: set[int] = set()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rows = candidates(search())
        cheap = [r for r in rows if r[7]["id"] not in tried]
        print(
            f"\n[{attempt}/{MAX_ATTEMPTS}] <${MAX_DPH} static+reli>{MIN_RELI} "
            f"eff>={MIN_EFF_VRAM}: {len(rows)} fit, {len(cheap)} untried"
        )
        for price, vram_g, frac, eff, ram_g, ports, reli, o in rows[:10]:
            mark = "tried" if o["id"] in tried else "new"
            print(
                f"  {o['id']} ${price:.4f} {o.get('gpu_name')} "
                f"frac={frac:.3f} eff={eff:.1f}G ram={ram_g:.0f}G "
                f"ports={ports} reli={reli:.2f} {o.get('geolocation')} [{mark}]"
            )
        if not cheap:
            print(f"none; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            if attempt % 12 == 0:
                tried.clear()
            continue

        price, vram_g, frac, eff, ram_g, ports, reli, o = cheap[0]
        oid = o["id"]
        tried.add(oid)
        print(f"renting {oid} @ ${price:.4f} frac={frac:.3f} eff={eff:.1f}G ...")
        res = create(oid)
        print(
            json.dumps(
                {
                    k: res.get(k)
                    for k in ("success", "new_contract", "error", "msg", "status_code")
                    if k in res
                }
            )
        )
        if res.get("success") and res.get("new_contract"):
            print(f"SUCCESS instance={res['new_contract']} offer={oid} ${price:.4f}")
            return 0
        time.sleep(1)

    print("FAILED: no static+reliable offer under threshold")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
