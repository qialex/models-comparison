#!/usr/bin/env python3
"""Rent cheapest Kokoro-82M instance matching hard filters.

Criteria (all required):
  - static_ip=True
  - dph_total < 0.035  (hard cap)
  - (gpu_mem_bw * gpu_frac) > 100
  - disk_space >= 8  (create with --disk 8)
  - num_gpus=1, rentable
  - eff VRAM >= 4GB, system RAM >= 4GB

Post-rent: probe direct http://public_ip:mapped_8080/health.
If the public port stays unreachable (common on cheap static_ip hosts),
destroy and try the next offer. App-up via SSH proxy alone is NOT enough.

Rank: cheapest first, then highest (bw * frac).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug, redact_create  # noqa: E402

TEMPLATE = "e2d0e011d6945f72bdea985c0c2e9778"
DISK = "8"
QUERY = (
    "gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 "
    "rentable=True static_ip=True dph_total<0.035"
)

MAX_PRICE = 0.035
MIN_BW_FRAC = 100.0  # gpu_mem_bw * frac
MIN_EFF = 4.0
MIN_RAM = 4.0
MIN_DISK = 8.0

MAX_ATTEMPTS = 900  # ~1h search loop
SLEEP_S = 4
OUT_LOG = cache_debug("kokoro-82m", "_debug") / "rent_bw100.log"

# After create: wait for port map, then poll /health on the public IP
PORT_WAIT_S = 90
HEALTH_WAIT_S = 420  # apt+pip+model can take several minutes
HEALTH_POLL_S = 8
HTTP_TIMEOUT_S = 6
# Consecutive hard timeouts after ports exist → treat as firewalled host
MAX_TIMEOUT_STREAK = 8


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
        "bw_frac": bw_frac,
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
        f"eff={r['eff']:.1f}G reli={r['reli']:.3f} "
        f"static={r['static']} mach={r['machine']} {r['loc']}"
    )


def create(oid: int) -> dict:
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


def destroy(iid: int) -> None:
    try:
        subprocess.check_output(
            ["vastai", "destroy", "instance", str(iid), "-y", "--raw"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        log(f"  destroy {iid} fail: {e.output or e}")


def show_instance(iid: int) -> dict | None:
    try:
        raw = subprocess.check_output(
            ["vastai", "show", "instance", str(iid), "--raw"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return json.loads(raw)
    except Exception as e:
        log(f"  show {iid} fail: {e}")
        return None


def mapped_http(inst: dict) -> tuple[str | None, int | None]:
    """Return (public_ip, host_port) for container 8080."""
    ip = (inst.get("public_ipaddr") or "").strip() or None
    ports = inst.get("ports") or {}
    entry = ports.get("8080/tcp") or ports.get("8080")
    if not entry:
        return ip, None
    if isinstance(entry, list) and entry:
        host_port = entry[0].get("HostPort")
        try:
            return ip, int(host_port)
        except (TypeError, ValueError):
            return ip, None
    return ip, None


def probe_health(ip: str, port: int) -> tuple[str, str]:
    """Return (kind, detail). kind: ok | loading | http | refuse | timeout | error."""
    url = f"http://{ip}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200 and '"status"' in body and "ok" in body:
                return "ok", body[:200]
            return "http", f"status={resp.status} body={body[:120]}"
    except urllib.error.HTTPError as e:
        if e.code == 503:
            return "loading", "503"
        return "http", f"HTTP {e.code}"
    except TimeoutError:
        return "timeout", "timeout"
    except urllib.error.URLError as e:
        reason = str(e.reason).lower()
        if "timed out" in reason or "timeout" in reason:
            return "timeout", reason
        if "refused" in reason:
            return "refuse", reason
        return "error", reason
    except Exception as e:
        msg = str(e).lower()
        if "timed out" in msg or "timeout" in msg:
            return "timeout", msg
        return "error", str(e)


def wait_direct_health(iid: int) -> tuple[bool, str]:
    """True if public /health is ok. False → destroy candidate."""
    t0 = time.time()
    ip: str | None = None
    port: int | None = None

    while time.time() - t0 < PORT_WAIT_S:
        inst = show_instance(iid)
        if not inst:
            time.sleep(HEALTH_POLL_S)
            continue
        ip, port = mapped_http(inst)
        status = inst.get("actual_status") or inst.get("cur_state")
        if ip and port:
            log(f"  ports ready ip={ip} port={port} status={status}")
            break
        log(f"  waiting ports status={status} ip={ip} ports={bool(inst.get('ports'))}")
        time.sleep(HEALTH_POLL_S)
    else:
        return False, "no_public_8080_mapping"

    timeout_streak = 0
    deadline = time.time() + HEALTH_WAIT_S
    while time.time() < deadline:
        kind, detail = probe_health(ip, port)  # type: ignore[arg-type]
        log(f"  health {ip}:{port} -> {kind} ({detail[:80]})")
        if kind == "ok":
            return True, f"http://{ip}:{port}/health"
        if kind == "timeout":
            timeout_streak += 1
            if timeout_streak >= MAX_TIMEOUT_STREAK:
                return False, f"direct_port_firewalled ({timeout_streak} timeouts)"
        else:
            # refuse / loading / http — app still coming up or transient
            timeout_streak = 0
        time.sleep(HEALTH_POLL_S)

    return False, "health_deadline_exceeded"


def log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    print(line, end="", flush=True)
    with OUT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    OUT_LOG.write_text("", encoding="utf-8")
    log(
        "rent_bw100 start (kokoro)\n"
        f"  static_ip + price<{MAX_PRICE} + (gpu_mem_bw*frac)>{MIN_BW_FRAC} + cheapest\n"
        f"  disk_create={DISK} eff>={MIN_EFF} ram>={MIN_RAM}\n"
        f"  post-rent: direct /health required "
        f"(port_wait={PORT_WAIT_S}s health_wait={HEALTH_WAIT_S}s "
        f"timeout_streak={MAX_TIMEOUT_STREAK})\n"
        f"  template={TEMPLATE}\n"
        f"  query={QUERY}\n"
    )
    tried_offers: set[int] = set()
    bad_machines: set[int] = set()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            rows = [norm(o) for o in search()]
        except Exception as e:
            log(f"[{attempt}/{MAX_ATTEMPTS}] search fail: {e}; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            continue

        hits = sorted(
            [
                r
                for r in rows
                if fit(r)
                and r["id"] not in tried_offers
                and r["machine"] not in bad_machines
            ],
            key=lambda r: (r["price"], -r["bw_frac"]),
        )
        peek = sorted(
            [r for r in rows if r["static"]],
            key=lambda r: (r["price"], -r["bw_frac"]),
        )[:5]

        log(
            f"\n[{attempt}/{MAX_ATTEMPTS}] offers={len(rows)} "
            f"fit={len(hits)} tried_offers={len(tried_offers)} "
            f"bad_machines={sorted(bad_machines)}"
        )
        for r in peek:
            tag = "FIT" if fit(r) and r["id"] not in tried_offers else "miss"
            if r["machine"] in bad_machines:
                tag = "badmach"
            log(f"  peek [{tag}] {fmt(r)}")

        if not hits:
            log(f"  none fit; sleep {SLEEP_S}s")
            time.sleep(SLEEP_S)
            if attempt % 10 == 0:
                # keep bad_machines; only clear offer tries so market can refresh
                tried_offers.clear()
            continue

        target = hits[0]
        tried_offers.add(target["id"])
        log(f"  RENTING {fmt(target)}")
        res = create(target["id"])
        log(json.dumps(redact_create(res)))
        if not (res.get("success") and res.get("new_contract")):
            time.sleep(2)
            continue

        iid = int(res["new_contract"])
        log(f"  created instance={iid}; probing direct /health ...")
        ok, why = wait_direct_health(iid)
        if ok:
            log(
                f"SUCCESS instance={iid} offer={target['id']} "
                f"${target['price']:.4f} health={why}"
            )
            return 0

        log(f"  FAIL direct health: {why} - destroying {iid}")
        destroy(iid)
        if target["machine"]:
            bad_machines.add(target["machine"])
            log(f"  blacklisted machine={target['machine']}")
        time.sleep(2)

    log("FAILED - no rent with reachable direct /health within window")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
