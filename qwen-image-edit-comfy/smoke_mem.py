#!/usr/bin/env python3
"""Smoke Qwen-Image-Edit ComfyUI on :8190; sample host RAM + GPU VRAM during run."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def post(url: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def nvidia() -> tuple[float, float]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, total = out.split(",")
    return float(used), float(total)


def comfy_ram(base: str) -> tuple[float, float] | None:
    try:
        s = get(f"{base.rstrip('/')}/system_stats")["system"]
        total = s["ram_total"] / (1024**3)
        free = s["ram_free"] / (1024**3)
        return total - free, total
    except Exception:
        return None


def host_ram_used_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().used / (1024**3)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8190")
    ap.add_argument(
        "--workflow",
        type=Path,
        default=ROOT / "qwen-image-edit-comfy" / "workflow_api_smoke.json",
    )
    ap.add_argument(
        "--image",
        type=Path,
        default=Path(
            r"C:\Users\Alex\.cursor\projects\c-Productivity-small-models\assets"
            r"\c__Users_Alex_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images"
            r"_6b90258f-2498-4999-89c2-bf7ee057bb51-6c7adc07-8b2c-4c94-9c22-86e4c12d58bb.png"
        ),
    )
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--no-image", action="store_true", help="skip copying a reference image (text-only)")
    args = ap.parse_args()

    if not args.no_image:
        if not args.image.is_file():
            print("missing image", args.image)
            return 1
        subprocess.check_call(
            [
                "docker",
                "cp",
                str(args.image),
                "qwen-image-edit-comfy-gpu:/app/ComfyUI/input/smoke_input.png",
            ]
        )
        print("copied smoke_input.png into container", flush=True)
    else:
        print("text-only: no reference image", flush=True)

    wf = json.loads(args.workflow.read_text(encoding="utf-8"))
    samples: list[dict] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            row: dict = {"t": round(time.time(), 2)}
            try:
                vu, vt = nvidia()
                row["vram_used_mib"] = vu
                row["vram_total_mib"] = vt
            except Exception as e:
                row["vram_err"] = str(e)
            cr = comfy_ram(args.base)
            if cr:
                row["comfy_ram_used_gb"] = round(cr[0], 2)
                row["comfy_ram_total_gb"] = round(cr[1], 2)
            hr = host_ram_used_gb()
            if hr is not None:
                row["host_ram_used_gb"] = round(hr, 2)
            samples.append(row)
            stop.wait(args.interval)

    thr = threading.Thread(target=sampler, daemon=True)
    thr.start()
    time.sleep(0.5)

    print("queueing...", flush=True)
    t0 = time.perf_counter()
    try:
        out = post(f"{args.base.rstrip('/')}/prompt", {"prompt": wf})
    except urllib.error.HTTPError as e:
        stop.set()
        print("HTTP", e.code, e.read().decode("utf-8", errors="replace")[:2000])
        return 1
    prompt_id = out.get("prompt_id")
    print("prompt_id:", prompt_id, flush=True)
    if not prompt_id:
        stop.set()
        print(out)
        return 1

    deadline = time.time() + args.timeout
    status_str = None
    while time.time() < deadline:
        try:
            hist = get(f"{args.base.rstrip('/')}/history/{prompt_id}")
        except Exception as e:
            print("history poll err:", type(e).__name__, e, flush=True)
            time.sleep(5)
            continue
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status") or {}
            status_str = status.get("status_str")
            if status.get("completed") or entry.get("outputs"):
                break
            for m in status.get("messages") or []:
                if m and m[0] == "execution_error":
                    stop.set()
                    wall = time.perf_counter() - t0
                    print("EXEC ERROR after", round(wall, 1), "s")
                    print(json.dumps(m[1], indent=2)[:4000])
                    _report(samples, wall, failed=True)
                    return 2
        time.sleep(2)
    else:
        stop.set()
        print("timeout")
        _report(samples, time.perf_counter() - t0, failed=True)
        return 3

    stop.set()
    thr.join(timeout=5)
    wall = time.perf_counter() - t0
    print(f"done wall_s={wall:.1f} status={status_str}", flush=True)
    _report(samples, wall, failed=False)
    return 0 if status_str != "error" else 2


def _report(samples: list[dict], wall: float, failed: bool) -> None:
    vrams = [s["vram_used_mib"] for s in samples if "vram_used_mib" in s]
    crams = [s["comfy_ram_used_gb"] for s in samples if "comfy_ram_used_gb" in s]
    hrams = [s["host_ram_used_gb"] for s in samples if "host_ram_used_gb" in s]
    summary = {
        "wall_s": round(wall, 1),
        "failed": failed,
        "samples": len(samples),
        "vram_peak_mib": max(vrams) if vrams else None,
        "vram_min_mib": min(vrams) if vrams else None,
        "comfy_ram_peak_gb": max(crams) if crams else None,
        "host_ram_peak_gb": max(hrams) if hrams else None,
    }
    out = ROOT / "results" / "qwen-image-edit-comfy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_mem.json").write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print("wrote", out / "smoke_mem.json")


if __name__ == "__main__":
    raise SystemExit(main())
