#!/usr/bin/env python3
"""Miracle hunt+rent for Kokoro Vast template.

Rent when ALL of:
  - static_ip
  - reli > 0.9
  - kokoro core: eff_vram>=4, ram>=4, disk>=8
  - AND either:
      (price < 0.026 AND score > 70)
      OR (score > 90 AND price <= 0.036)

score = (gpu_mem_bw * frac) / (price * 100)
No Japan geo (optional safety). Runs up to ~1 hour.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug, redact_create  # noqa: E402

TEMPLATE = "e2d0e011d6945f72bdea985c0c2e9778"
DISK = "8"
QUERY = "gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 rentable=True static_ip=True"
MIN_EFF = 4.0
MIN_RAM = 4.0
MIN_RELI = 0.90
# Band A: ultra-cheap needs solid score
CHEAP_PRICE = 0.026
CHEAP_MIN_SCORE = 70.0
# Band B: higher score allows slightly higher price
HIGH_SCORE = 90.0
HIGH_PRICE_CAP = 0.036

MAX_ATTEMPTS = 900  # ~1h at SLEEP_S=4
SLEEP_S = 4
OUT_LOG = cache_debug("kokoro-82m", "_debug") / "hunt_miracle.log"


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
    return val / 1024 if val > threshold else val


def norm(o: dict) -> dict:
    vram = float(o.get("gpu_ram") or 0)
    vram_g = _gb(vram, 64)
    frac = float(o["gpu_frac"] if o.get("gpu_frac") is not None else 1.0)
    ram = float(o.get("cpu_ram") or 0)
    ram_g = _gb(ram, 512)
    price = float(o.get("dph_total") or 99)
    bw = float(o.get("gpu_mem_bw") or 0)
    cents = price * 100.0
    score = (bw * frac) / cents if cents > 0 else 0.0
    loc = o.get("geolocation") or ""
    return {
        "id": o["id"],
        "machine": int(o.get("machine_id") or 0),
        "price": price,
        "gpu": o.get("gpu_name"),
        "frac": frac,
        "eff": frac * vram_g,
        "ram": ram_g,
        "disk": float(o.get("disk_space") or 0),
        "bw": bw,
        "score": round(score, 2),
        "reli": float(o.get("reliability2") or o.get("reliability") or 0),
        "static": bool(o.get("static_ip")),
        "verified": bool(
            o.get("verified")
            or str(o.get("verification") or "").lower() == "verified"
        ),
        "loc": loc,
        "japan": "japan" in loc.lower() or ", jp" in loc.lower(),
    }


def band(r: dict) -> str | None:
    """Return band name if rentable under miracle rules."""
    if not r["static"]:
        return None
    if r["reli"] <= MIN_RELI:
        return None
    if r["eff"] < MIN_EFF or r["ram"] < MIN_RAM or r["disk"] < 8:
        return None
    if r["japan"]:
        return None
    if r["price"] < CHEAP_PRICE and r["score"] > CHEAP_MIN_SCORE:
        return "cheap_lt026_score70"
    if r["score"] > HIGH_SCORE and r["price"] <= HIGH_PRICE_CAP:
        return "score90_cap036"
    return None


def fmt(r: dict) -> str:
    return (
        f"{r['id']} score={r['score']:.2f} ${r['price']:.4f} {r['gpu']} "
        f"frac={r['frac']:.3f} eff={r['eff']:.1f}G reli={r['reli']:.3f} "
        f"static={r['static']} ver={r['verified']} {r['loc']}"
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


def log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    print(line, end="", flush=True)
    with OUT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    OUT_LOG.write_text("", encoding="utf-8")
    log(
        f"miracle hunt start template={TEMPLATE}\n"
        f"  static + reli>{MIN_RELI} + core\n"
        f"  bandA: price<{CHEAP_PRICE} AND score>{CHEAP_MIN_SCORE}\n"
        f"  bandB: score>{HIGH_SCORE} AND price<={HIGH_PRICE_CAP}\n"
        f"  max_attempts={MAX_ATTEMPTS} sleep={SLEEP_S}s (~{MAX_ATTEMPTS*SLEEP_S/3600:.1f}h)\n"
    )
    tried: set[int] = set()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            offers = [norm(o) for o in search()]
        except Exception as e:
            log(f"[{attempt}/{MAX_ATTEMPTS}] search fail: {e}; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            continue

        static = [r for r in offers if r["static"]]
        hits = []
        for r in static:
            b = band(r)
            if b and r["id"] not in tried:
                hits.append((b, r))
        hits.sort(key=lambda x: (-x[1]["score"], x[1]["price"]))

        # market peek: best static by score under 0.04 for logs
        peek = sorted(
            [r for r in static if r["eff"] >= MIN_EFF and r["ram"] >= MIN_RAM],
            key=lambda r: (-r["score"], r["price"]),
        )[:5]

        log(
            f"\n[{attempt}/{MAX_ATTEMPTS}] static={len(static)} "
            f"miracle_ready={len(hits)} tried={len(tried)}"
        )
        for r in peek:
            tag = band(r) or (
                f"score={r['score']:.0f}"
                if r["price"] <= HIGH_PRICE_CAP
                else f"${r['price']:.3f}"
            )
            log(f"  peek {fmt(r)} [{tag}]")

        if not hits:
            log(f"  no miracle yet; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            if attempt % 30 == 0:
                tried.clear()
            continue

        b, r = hits[0]
        tried.add(r["id"])
        log(f"  RENTING band={b} {fmt(r)}")
        res = create(r["id"])
        log(json.dumps(redact_create(res)))
        if res.get("success") and res.get("new_contract"):
            log(f"SUCCESS instance={res['new_contract']} offer={r['id']} band={b}")
            return 0
        time.sleep(2)

    log("FAILED — no miracle within window")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
