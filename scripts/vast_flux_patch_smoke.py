#!/usr/bin/env python3
"""Patch Vast flux INT8 instance and smoke-test."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "flux2-klein-9b-kv-int8" / "pipeline_flux2_klein_kv_offload.py"
APP = ROOT / "flux2-klein-9b-kv-int8" / "app.py"


def sh(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def patch_and_restart(ip: str, ssh_port: int) -> None:
    scp = sh(
        [
            "scp",
            "-P",
            str(ssh_port),
            "-o",
            "StrictHostKeyChecking=no",
            str(PIPE),
            str(APP),
            f"root@{ip}:/workspace/app/",
        ]
    )
    print("scp:", scp.returncode, (scp.stderr or "")[-200:])
    remote = r"""
set -e
mkdir -p /workspace/app
MODEL_DIR=$(ls -d /workspace/hf-cache/hub/models--albex123--flux2-klein-kv-qint8-offload/snapshots/* 2>/dev/null | head -n1 || true)
echo MODEL_DIR=$MODEL_DIR
# kill existing uvicorn without matching this shell
killall -q python || true
sleep 2
export MODEL_ID=albex123/flux2-klein-kv-qint8-offload
export HF_HOME=/workspace/hf-cache
export MODEL_DIR
export PORT=8127 HEIGHT=512 WIDTH=512 STEPS=4
cd /workspace/app
nohup python -m uvicorn app:app --host 0.0.0.0 --port 8127 >/workspace/uvicorn.log 2>&1 &
echo started pid=$!
for i in $(seq 1 90); do
  if curl -sf localhost:8127/health >/tmp/h.json; then
    echo HEALTH_OK
    cat /tmp/h.json
    exit 0
  fi
  sleep 2
done
echo HEALTH_FAIL
tail -80 /workspace/uvicorn.log || true
exit 1
"""
    r = sh(["ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", f"root@{ip}", remote], timeout=240)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("stderr:", (r.stderr or "")[-800:])
        raise SystemExit(f"restart failed rc={r.returncode}")


def wait_health(base: str, tries: int = 60) -> dict:
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(f"{base.rstrip('/')}/health", timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            last = e
            print(f"wait health {i}: {type(e).__name__}", flush=True)
            time.sleep(5)
    raise SystemExit(f"health never ok: {last}")


def smoke(base: str) -> dict:
    body = json.dumps(
        {"prompt": "a red apple on a wooden table", "height": 512, "width": 512, "steps": 4, "seed": 42, "format": "png"}
    ).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/generate", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read().decode())
            return {k: v for k, v in d.items() if k != "image_base64"}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"smoke HTTP {e.code}: {e.read()[:800].decode(errors='replace')}")


def main() -> int:
    # args: name ip ssh_port http_port
    name, ip, ssh_port, http_port = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    base = f"http://{ip}:{http_port}"
    print(f"=== {name} {base}", flush=True)
    try:
        h = wait_health(base, tries=6)
        print("pre health", h.get("status"), "load_s", h.get("load_s"), flush=True)
    except SystemExit as e:
        print("pre health not ready:", e, flush=True)
        print("waiting longer for boot...", flush=True)
        h = wait_health(base, tries=90)
        print("boot health", h.get("status"), flush=True)

    patch_and_restart(ip, ssh_port)
    h2 = wait_health(base, tries=30)
    print("post health", h2, flush=True)
    s = smoke(base)
    print("smoke", s, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
