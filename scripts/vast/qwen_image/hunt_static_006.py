#!/usr/bin/env python3
"""Hunt Vast offers until one fits Qwen-Image-2.1 template criteria; then rent.

Criteria:
  gpu_ram>=12 (eff VRAM = frac*vram >= 11.5)
  cpu_ram>=30
  disk_space>=20
  num_gpus=1
  static_ip=True
  reliability>0.90
  dph_total<=0.06
  rentable now (start_date not future)
  exclude Ukraine / deverified / is_vm_deverified
  disk create = 20
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # scripts/vast/qwen_image → repo
VAST = ROOT / "vast" / "qwen-image-2.1"
HASH = (VAST / ".vast_hash_id").read_text(encoding="utf-8").strip()
KEEP = {"52841731", "52842312"}  # Ohio + Korea — do not destroy

QUERY = (
    "gpu_ram>=12 cpu_ram>=30 disk_space>=20 num_gpus=1 rentable=True "
    "static_ip=True reliability>0.90 dph_total<=0.06"
)
MAX_DPH = 0.06
MIN_EFF = 11.5
MIN_RAM = 30
DISK = "20"
MAX_HOURS = 1.0
SLEEP_S = 20
MAX_ATTEMPTS = int((MAX_HOURS * 3600) / SLEEP_S) + 5


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


def candidates(offers, now: float):
    rows = []
    for o in offers:
        geo = o.get("geolocation") or ""
        if "Ukraine" in geo:
            continue
        if o.get("is_vm_deverified"):
            continue
        ver = str(o.get("verification") or "")
        if ver == "deverified":
            continue
        if not o.get("static_ip") or not o.get("rentable") or o.get("rented"):
            continue

        vram = float(o.get("gpu_ram") or 0)
        vram_g = vram / 1024 if vram > 64 else vram
        frac = float(o.get("gpu_frac") if o.get("gpu_frac") is not None else 1.0)
        ram = float(o.get("cpu_ram") or 0)
        ram_g = ram / 1024 if ram > 512 else ram
        price = float(o.get("dph_total") or 99)
        reli = float(o.get("reliability2") or o.get("reliability") or 0)
        down = float(o.get("inet_down") or 0)
        sd = float(o.get("start_date") or 0)
        eff = vram_g * frac

        if price > MAX_DPH or eff < MIN_EFF or ram_g <= MIN_RAM or reli <= 0.90:
            continue
        # available right now (not scheduled for later)
        if sd and sd > now + 3600:
            continue

        # prefer verified, then faster downlink, then cheaper
        score = (0 if ver == "verified" else 1, -down, price)
        rows.append((score, price, o, vram_g, frac, eff, ram_g, down, ver))
    rows.sort()
    return rows


def create(offer_id: int):
    cmd = [
        "vastai",
        "create",
        "instance",
        str(offer_id),
        "--template_hash",
        HASH,
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
    t0 = time.time()
    deadline = t0 + MAX_HOURS * 3600
    tried: set[int] = set()
    print(
        f"hunt start hash={HASH} keep={sorted(KEEP)} "
        f"max_dph={MAX_DPH} disk={DISK} until={datetime.fromtimestamp(deadline, timezone.utc).isoformat()}",
        flush=True,
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if time.time() >= deadline:
            print("TIMEOUT — no fit within hour", flush=True)
            return 1

        now = time.time()
        try:
            rows = candidates(search(), now)
        except Exception as e:
            print(f"[{attempt}] search error: {e}", flush=True)
            time.sleep(SLEEP_S)
            continue

        elapsed_m = (now - t0) / 60
        print(
            f"[{attempt}/{MAX_ATTEMPTS}] t={elapsed_m:.1f}m fit={len(rows)} tried={len(tried)}",
            flush=True,
        )
        for score, price, o, v, f, e, r, down, ver in rows[:8]:
            mark = "T" if o["id"] in tried else "N"
            print(
                f"  {mark} {o['id']} ${price:.4f} {o.get('gpu_name')} "
                f"eff={e:.1f}G down={down:.0f} {ver} {o.get('geolocation')}",
                flush=True,
            )

        cheap = [x for x in rows if x[2]["id"] not in tried]
        if not cheap:
            if attempt % 10 == 0:
                tried.clear()
                print("  cleared tried set", flush=True)
            time.sleep(SLEEP_S)
            continue

        score, price, o, v, f, e, r, down, ver = cheap[0]
        oid = int(o["id"])
        tried.add(oid)
        print(
            f"RENT {oid} ${price:.4f} disk={DISK} down={down:.0f} {ver} {o.get('geolocation')}",
            flush=True,
        )
        res = create(oid)
        # never log api key
        safe = {k: res.get(k) for k in ("success", "new_contract", "error", "msg", "status_code") if k in res}
        print(json.dumps(safe), flush=True)

        if res.get("success") and res.get("new_contract"):
            nid = res["new_contract"]
            print(
                f"SUCCESS instance={nid} offer={oid} ${price:.4f} "
                f"{o.get('gpu_name')} {o.get('geolocation')} down={down:.0f}",
                flush=True,
            )
            (VAST / ".vast_instance_id_3").write_text(f"{nid}\n", encoding="utf-8")
            (VAST / ".vast_offer_id_3").write_text(f"{oid}\n", encoding="utf-8")
            res.pop("instance_api_key", None)
            (VAST / "rent_result_3.json").write_text(
                json.dumps(res, indent=2) + "\n", encoding="utf-8"
            )
            return 0

        time.sleep(2)

    print("FAILED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
