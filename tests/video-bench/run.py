#!/usr/bin/env python3
"""Video I2V bench runner for LTX-Video service.

Writes under results/video-bench/<model>/<suite>/<case_id>/ :
  prompt.txt  meta.json  video.mp4

Examples:
  python tests/video-bench/run.py --suite style_lock --base http://127.0.0.1:8122
  python tests/video-bench/run.py --suite all --base http://127.0.0.1:8122 --skip-existing
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "video-bench"
REFS_MANIFEST = BENCH / "refs_manifest.json"

SUITES = {
    "style_lock": BENCH / "cases_i2v_style_lock.json",
    "a_identity": BENCH / "cases_a_identity_stress.json",
    "b_camera": BENCH / "cases_b_camera_grammar.json",
    "c_action": BENCH / "cases_c_action_ladder.json",
    "d_style_trap": BENCH / "cases_d_style_trap.json",
    "e_multi": BENCH / "cases_e_multi_condition.json",
}


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


def load_refs() -> dict:
    data = json.loads(REFS_MANIFEST.read_text(encoding="utf-8"))
    return data["refs"]


def resolve_ref_path(ref_id: str, refs: dict, suite_data: dict) -> Path:
    # Prefer suite-local refs dict, else shared manifest
    suite_refs = suite_data.get("refs") or {}
    entry = suite_refs.get(ref_id) or refs.get(ref_id)
    if entry is None:
        raise KeyError(f"unknown ref id: {ref_id}")
    if isinstance(entry, str):
        rel = entry
    else:
        rel = entry["file"]
    path = BENCH / rel
    if not path.is_file():
        # fallback source_asset if present
        if isinstance(entry, dict) and entry.get("source_asset"):
            alt = Path(entry["source_asset"])
            if alt.is_file():
                return alt
        raise FileNotFoundError(path)
    return path


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    p = argparse.ArgumentParser(description="Video bench runner")
    p.add_argument("--model", default="ltxv-2b-0.9.8-distilled")
    p.add_argument("--base", required=True, help="API base, e.g. http://127.0.0.1:8122")
    p.add_argument(
        "--suite",
        default="style_lock",
        help="suite key, comma-list, or 'all'",
    )
    p.add_argument("--ids", default="", help="Comma-separated case ids")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=-1)
    args = p.parse_args()

    if args.suite.strip().lower() == "all":
        suite_keys = list(SUITES.keys())
    else:
        suite_keys = [x.strip() for x in args.suite.split(",") if x.strip()]

    unknown = [k for k in suite_keys if k not in SUITES]
    if unknown:
        print(f"unknown suites: {unknown}; known={list(SUITES)}", file=sys.stderr)
        return 1

    try:
        health = get_json(f"{args.base.rstrip('/')}/health", timeout=30)
        print("health:", json.dumps(health))
    except Exception as e:
        print(f"health failed: {e}", file=sys.stderr)
        return 1

    refs = load_refs()
    want_ids = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids.strip() else None

    grand = []
    for sk in suite_keys:
        suite_path = SUITES[sk]
        suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
        defaults = suite_data.get("defaults") or {}
        lock = (suite_data.get("lock_preamble") or "").strip()
        cases = suite_data.get("cases") or []
        if want_ids:
            cases = [c for c in cases if c["id"] in want_ids]
        if not cases:
            print(f"[{sk}] no cases selected")
            continue

        out_root = RESULTS_ROOT / args.model / sk
        out_root.mkdir(parents=True, exist_ok=True)
        summary = []

        for i, case in enumerate(cases):
            cid = case["id"]
            out_dir = out_root / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = out_dir / "video.mp4"
            if args.skip_existing and video_path.exists():
                print(f"[{sk} {i+1}/{len(cases)}] skip {cid}")
                continue

            ref_path = resolve_ref_path(case["ref"], refs, suite_data)
            prompt = case["prompt"]
            if lock:
                prompt = f"{lock}\n\n{prompt}"

            body = {
                "prompt": prompt,
                "negative_prompt": defaults.get("negative_prompt"),
                "image_base64": b64_file(ref_path),
                "height": args.height or defaults.get("height") or 576,
                "width": args.width or defaults.get("width") or 320,
                "num_frames": args.num_frames or defaults.get("num_frames") or 97,
                "fps": defaults.get("fps") or 24,
                "steps": args.steps or defaults.get("steps") or 8,
                "guidance_scale": (
                    args.guidance_scale
                    if args.guidance_scale >= 0
                    else defaults.get("guidance_scale", 1.0)
                ),
                "seed": args.seed if args.seed >= 0 else defaults.get("seed", 42),
            }
            if case.get("ref_last"):
                last_path = resolve_ref_path(case["ref_last"], refs, suite_data)
                body["last_image_base64"] = b64_file(last_path)

            (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            print(f"[{sk} {i+1}/{len(cases)}] {cid} ...", flush=True)
            t0 = time.perf_counter()
            try:
                out = post_json(f"{args.base.rstrip('/')}/generate", body, args.timeout)
                raw = base64.b64decode(out["video_base64"])
                video_path.write_bytes(raw)
                meta = {k: v for k, v in out.items() if k != "video_base64"}
                meta["suite"] = sk
                meta["case_id"] = cid
                meta["ref"] = case["ref"]
                meta["ref_last"] = case.get("ref_last")
                meta["ref_file"] = str(ref_path)
                meta["wall_s"] = round(time.perf_counter() - t0, 3)
                (out_dir / "meta.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"  ok {meta.get('latency_s')}s api / {meta['wall_s']}s wall "
                    f"vram_max={meta.get('vram_max_allocated_mb')} "
                    f"bytes={len(raw)}"
                )
                summary.append({"id": cid, "ok": True, **{k: meta.get(k) for k in ("latency_s", "wall_s", "vram_max_allocated_mb")}})
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"  FAIL {err}", file=sys.stderr)
                (out_dir / "meta.json").write_text(
                    json.dumps({"ok": False, "error": err}, indent=2) + "\n",
                    encoding="utf-8",
                )
                summary.append({"id": cid, "ok": False, "error": err})

        (out_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        grand.append({"suite": sk, "summary": summary})

    (RESULTS_ROOT / args.model / "summary_all.json").write_text(
        json.dumps(grand, indent=2) + "\n", encoding="utf-8"
    )
    failed = sum(1 for g in grand for s in g["summary"] if not s.get("ok"))
    print(f"done suites={len(grand)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
