#!/usr/bin/env python3
"""Hunt: show market under SHOW_DPH; only rent when <= RENT_DPH.
Excludes known-bad Japan hosts. reli>0.9 + effVRAM + RAM required to rent.
"""
from __future__ import annotations

import json
import subprocess
import time

TEMPLATE = "1ec7dac83a206e2101e0fa6ec06d2cdf"
DISK = "20"
SHOW_DPH = 0.10  # log window
RENT_DPH = 0.055  # only rent at/under this
MIN_EFF = 11.5
MIN_RAM = 30
MIN_RELI = 0.90
QUERY = "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True"
MAX_ATTEMPTS = 600
SLEEP_S = 3

# Known-bad: Japan docker_build failures (offer 45607706 / machine 54116)
EXCLUDE_OFFERS = {45607706}
EXCLUDE_MACHINES = {54116}
EXCLUDE_LOCS = ("japan", " jp")  # substring match on geolocation lowercased


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


def norm(o):
    vram = float(o.get("gpu_ram") or 0)
    vram_g = vram / 1024 if vram > 64 else vram
    frac = float(o.get("gpu_frac") if o.get("gpu_frac") is not None else 1.0)
    ram = float(o.get("cpu_ram") or 0)
    ram_g = ram / 1024 if ram > 512 else ram
    return {
        "id": o["id"],
        "machine": int(o.get("machine_id") or 0),
        "price": float(o.get("dph_total") or 99),
        "gpu": o.get("gpu_name"),
        "frac": frac,
        "eff": frac * vram_g,
        "ram": ram_g,
        "reli": float(o.get("reliability2") or o.get("reliability") or 0),
        "loc": o.get("geolocation") or "",
        "verified": bool(o.get("verified")),
        "static": bool(o.get("static_ip")),
        "o": o,
    }


def excluded(r) -> str | None:
    if r["id"] in EXCLUDE_OFFERS:
        return "bad_offer"
    if r["machine"] in EXCLUDE_MACHINES:
        return "bad_machine"
    loc = r["loc"].lower()
    for needle in EXCLUDE_LOCS:
        if needle in loc:
            return "japan"
    return None


def core_ok(r) -> bool:
    return r["eff"] >= MIN_EFF and r["ram"] > MIN_RAM


def fmt(r) -> str:
    return (
        f"{r['id']} ${r['price']:.4f} {r['gpu']} frac={r['frac']:.3f} "
        f"eff={r['eff']:.1f}G ram={r['ram']:.0f}G reli={r['reli']:.2f} "
        f"m={r['machine']} {r['loc']}"
    )


def create(oid: int):
    cmd = [
        "vastai",
        "create",
        "instance",
        str(oid),
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
        offers = [norm(o) for o in search()]
        under_show = [r for r in offers if r["price"] < SHOW_DPH]
        under_show.sort(key=lambda r: (r["price"], -r["eff"]))

        core = [r for r in under_show if core_ok(r)]
        core_reli = [r for r in core if r["reli"] > MIN_RELI]
        skipped_jp = [r for r in core_reli if excluded(r)]
        rentable = [
            r
            for r in core_reli
            if not excluded(r) and r["price"] <= RENT_DPH and r["id"] not in tried
        ]
        watch = [r for r in core_reli if not excluded(r)]

        print(
            f"\n[{attempt}/{MAX_ATTEMPTS}] market <${SHOW_DPH} | "
            f"rent<=${RENT_DPH} reli>{MIN_RELI} eff>={MIN_EFF} ram>{MIN_RAM}"
        )
        print(
            f"  stats: under_show={len(under_show)} core={len(core)} "
            f"core+reli={len(core_reli)} skip_jp={len(skipped_jp)} "
            f"rent_ready={len(rentable)} tried={len(tried)}"
        )

        if under_show:
            c0 = under_show[0]
            tag = excluded(c0) or ("core" if core_ok(c0) else "weak")
            print(f"  cheapest_any: {fmt(c0)} [{tag}]")
        else:
            print("  cheapest_any: (none under show window)")

        if watch:
            w0 = watch[0]
            print(f"  cheapest_fit: {fmt(w0)}" + (" [RENT]" if w0["price"] <= RENT_DPH else " [wait]"))
            for r in watch[:6]:
                mark = (
                    "RENT"
                    if r["price"] <= RENT_DPH and r["id"] not in tried
                    else ("tried" if r["id"] in tried else f"${r['price']:.3f}")
                )
                print(f"    {fmt(r)} [{mark}]")
        else:
            print("  cheapest_fit: (none - waiting for non-JP core+reli)")

        if skipped_jp[:2]:
            print("  skipped_jp:")
            for r in skipped_jp[:2]:
                print(f"    {fmt(r)} [{excluded(r)}]")

        if not rentable:
            print(f"  no rent yet; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            if attempt % 20 == 0:
                tried.clear()
            continue

        r = rentable[0]
        tried.add(r["id"])
        print(f"  RENTING {r['id']} @ ${r['price']:.4f} ...")
        res = create(r["id"])
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
            print(f"SUCCESS instance={res['new_contract']} offer={r['id']}")
            return 0
        time.sleep(1)

    print("FAILED — no rent under threshold")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
