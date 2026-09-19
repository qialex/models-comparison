#!/usr/bin/env python3
"""Submit ComfyUI API workflow and wait for completion; print VRAM via nvidia-smi if available."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def post(url: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8188")
    p.add_argument("--workflow", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    wf = json.loads(args.workflow.read_text(encoding="utf-8"))
    print("queueing...", flush=True)
    t0 = time.perf_counter()
    out = post(f"{args.base.rstrip('/')}/prompt", {"prompt": wf})
    prompt_id = out.get("prompt_id")
    print("prompt_id:", prompt_id, flush=True)
    if not prompt_id:
        print(out)
        return 1

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        hist = get(f"{args.base.rstrip('/')}/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or entry.get("outputs"):
                wall = time.perf_counter() - t0
                print(f"done wall_s={wall:.1f}", flush=True)
                print(json.dumps({"status": status, "outputs": list((entry.get("outputs") or {}).keys())}, indent=2))
                if status.get("status_str") == "error":
                    print(json.dumps(status, indent=2))
                    return 2
                return 0
            msgs = status.get("messages") or []
            for m in msgs:
                if m and m[0] == "execution_error":
                    print("EXEC ERROR:", json.dumps(m[1], indent=2)[:4000])
                    return 2
        time.sleep(2)
    print("timeout")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
