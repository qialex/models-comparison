#!/usr/bin/env python3
"""Multi-model image generation bench.

Writes under results/image-bench/<model>/<prompt_id>/ :
  prompt.txt  meta.json  image.png

Example:
  python tests/image-bench/run.py --model flux2-klein-4b --base http://127.0.0.1:8116
  python tests/image-bench/run.py --model flux2-klein-4b --base http://127.0.0.1:8116 --ids 01_studio_portrait,02_male_figure
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts_adult_nude.json"
DEFAULT_COMFY_WORKFLOW = ROOT / "z-image-comfy" / "workflow_api_512.json"
RESULTS_ROOT = ROOT / "results" / "image-bench"


def post_json(url: str, body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_flux(base: str, prompt: str, height: int, width: int, steps: int, seed: int, timeout: int) -> tuple[bytes, dict]:
    out = post_json(
        f"{base.rstrip('/')}/generate",
        {
            "prompt": prompt,
            "height": height,
            "width": width,
            "steps": steps,
            "seed": seed,
            "format": "png",
        },
        timeout,
    )
    raw = base64.b64decode(out["image_base64"])
    meta = {k: v for k, v in out.items() if k != "image_base64"}
    return raw, meta


def _comfy_get_bytes(url: str, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def run_comfy(
    base: str,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
    timeout: int,
    *,
    workflow_path: Path,
    filename_prefix: str = "zimage_bench",
) -> tuple[bytes, dict]:
    """ComfyUI API: mutate workflow JSON, queue, poll history, download via /view."""
    wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    # Node ids match z-image-comfy/workflow_api_512.json
    if "4" in wf and "inputs" in wf["4"]:
        wf["4"]["inputs"]["text"] = prompt
    if "7" in wf and "inputs" in wf["7"]:
        wf["7"]["inputs"]["width"] = width
        wf["7"]["inputs"]["height"] = height
    if "9" in wf and "inputs" in wf["9"]:
        wf["9"]["inputs"]["seed"] = seed
        wf["9"]["inputs"]["steps"] = steps
    if "11" in wf and "inputs" in wf["11"]:
        wf["11"]["inputs"]["filename_prefix"] = filename_prefix

    t0 = time.perf_counter()
    queued = post_json(f"{base.rstrip('/')}/prompt", {"prompt": wf}, min(timeout, 60))
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Comfy queue failed: {queued}")

    deadline = time.time() + timeout
    entry = None
    while time.time() < deadline:
        hist = get_json(f"{base.rstrip('/')}/history/{prompt_id}", 30)
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status") or {}
            msgs = status.get("messages") or []
            for m in msgs:
                if m and m[0] == "execution_error":
                    raise RuntimeError(f"Comfy execution_error: {m[1]}")
            if status.get("status_str") == "error":
                raise RuntimeError(f"Comfy error status: {status}")
            if status.get("completed") or entry.get("outputs"):
                break
        time.sleep(1.0)
    else:
        raise TimeoutError(f"Comfy prompt {prompt_id} timed out after {timeout}s")

    assert entry is not None
    outputs = entry.get("outputs") or {}
    images = None
    for node_out in outputs.values():
        if isinstance(node_out, dict) and node_out.get("images"):
            images = node_out["images"]
            break
    if not images:
        raise RuntimeError(f"Comfy produced no images: {list(outputs.keys())}")

    img_info = images[0]
    qs = urllib.parse.urlencode(
        {
            "filename": img_info["filename"],
            "subfolder": img_info.get("subfolder") or "",
            "type": img_info.get("type") or "output",
        }
    )
    raw = _comfy_get_bytes(f"{base.rstrip('/')}/view?{qs}", timeout)
    wall = time.perf_counter() - t0
    meta = {
        "latency_s": round(wall, 3),
        "images_per_sec": round(1.0 / wall, 4) if wall > 0 else None,
        "comfy_prompt_id": prompt_id,
        "comfy_filename": img_info["filename"],
        "height": height,
        "width": width,
        "steps": steps,
        "seed": seed,
    }
    return raw, meta


BACKENDS = {
    # Same HTTP shape: POST /generate with prompt/height/width/steps/seed
    "t2i": run_flux,
    "flux2-klein-4b": run_flux,
    "flux2-klein-9b-kv": run_flux,
    "flux2-klein-9b-kv-int8": run_flux,
    "juggernaut-xl-v9": run_flux,
    "juggernaut-xi-v11": run_flux,
    "realistic-vision-v5.1": run_flux,
    "animagine-xl-4.0": run_flux,
    "sdxl-turbo": run_flux,
    "z-image-turbo": run_flux,
    "z-image-comfy": run_comfy,
}


def check_health(base: str, backend: str) -> dict:
    base = base.rstrip("/")
    if backend == "z-image-comfy":
        return get_json(f"{base}/system_stats", timeout=30)
    return get_json(f"{base}/health", timeout=30)


def main() -> int:
    p = argparse.ArgumentParser(description="Image bench runner")
    p.add_argument("--model", required=True, help="Folder name under results/image-bench/")
    p.add_argument("--base", required=True, help="API base, e.g. http://127.0.0.1:8116")
    p.add_argument("--backend", default="flux2-klein-4b", choices=sorted(BACKENDS))
    p.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    p.add_argument("--ids", default="", help="Comma-separated prompt ids (default: all)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_COMFY_WORKFLOW,
        help="ComfyUI API workflow JSON (z-image-comfy backend)",
    )
    args = p.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if args.ids.strip():
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        prompts = [x for x in prompts if x["id"] in want]

    if not prompts:
        print("no prompts selected", file=sys.stderr)
        return 1

    try:
        health = check_health(args.base, args.backend)
        print("health:", json.dumps(health)[:500])
    except Exception as e:
        print(f"health failed: {e}", file=sys.stderr)
        return 1

    if args.backend == "z-image-comfy" and not args.workflow.is_file():
        print(f"workflow missing: {args.workflow}", file=sys.stderr)
        return 1

    model_dir = RESULTS_ROOT / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    runner = BACKENDS[args.backend]

    summary = []
    for i, item in enumerate(prompts):
        pid = item["id"]
        prompt = item["prompt"]
        out_dir = model_dir / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / "image.png"
        if args.skip_existing and img_path.exists():
            print(f"[{i+1}/{len(prompts)}] skip existing {pid}")
            continue

        (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        print(f"[{i+1}/{len(prompts)}] {pid} ...", flush=True)
        t0 = time.perf_counter()
        try:
            if args.backend == "z-image-comfy":
                raw, meta = run_comfy(
                    args.base,
                    prompt,
                    args.height,
                    args.width,
                    args.steps,
                    args.seed + i,
                    args.timeout,
                    workflow_path=args.workflow,
                    filename_prefix=f"bench_{pid}",
                )
            else:
                raw, meta = runner(
                    args.base, prompt, args.height, args.width, args.steps, args.seed + i, args.timeout
                )
            wall = time.perf_counter() - t0
            img_path.write_bytes(raw)
            meta = {
                **meta,
                "model": args.model,
                "backend": args.backend,
                "base": args.base,
                "prompt_id": pid,
                "wall_s": round(wall, 3),
                "bytes": len(raw),
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(
                f"  ok {meta.get('latency_s', wall)}s  "
                f"{meta.get('images_per_sec')} img/s  -> {img_path}"
            )
            summary.append({"id": pid, "ok": True, **{k: meta.get(k) for k in ("latency_s", "wall_s", "vram_max_allocated_mb")}})
        except Exception as e:
            wall = time.perf_counter() - t0
            err = {"ok": False, "error": f"{type(e).__name__}: {e}", "wall_s": round(wall, 3)}
            (out_dir / "meta.json").write_text(json.dumps(err, indent=2) + "\n", encoding="utf-8")
            print(f"  FAIL {err['error']}", file=sys.stderr)
            summary.append({"id": pid, **err})

    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ok = sum(1 for s in summary if s.get("ok"))
    print(f"\ndone {ok}/{len(summary)} -> {model_dir}")
    return 0 if ok == len(summary) else 2


if __name__ == "__main__":
    raise SystemExit(main())
