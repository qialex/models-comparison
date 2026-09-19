#!/usr/bin/env python3
"""Selective download of LTX-Video 2B distilled (+ optional FP8) + Diffusers sidecars.

Skips 13B / old 2B / media / default transformer folder (~254GB full repo).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "Lightricks/LTX-Video"
TRANSFORMER = "ltxv-2b-0.9.8-distilled.safetensors"
TRANSFORMER_FP8 = "ltxv-2b-0.9.8-distilled-fp8.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", action="store_true", help=f"also download {TRANSFORMER_FP8}")
    ap.add_argument("--fp8-only", action="store_true", help=f"only download {TRANSFORMER_FP8}")
    args = ap.parse_args()

    hf_home = Path(os.environ.get("HF_HOME", Path.cwd() / "cache" / "ltx-video"))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or None
    print(f"HF_HOME={hf_home}")

    if not args.fp8_only:
        print(f"downloading Diffusers sidecars from {REPO} ...")
        snapshot_download(
            repo_id=REPO,
            token=token,
            cache_dir=str(hub_cache),
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "text_encoder/*",
                "vae/*",
            ],
            ignore_patterns=["media/*", "**/*.gif", "**/*.md"],
        )
        print(f"downloading {TRANSFORMER} ...")
        path = hf_hub_download(
            repo_id=REPO,
            filename=TRANSFORMER,
            token=token,
            cache_dir=str(hub_cache),
        )
        print(f"transformer -> {path}")
        print(f"downloading {UPSCALER} ...")
        up = hf_hub_download(
            repo_id=REPO,
            filename=UPSCALER,
            token=token,
            cache_dir=str(hub_cache),
        )
        print(f"upscaler -> {up}")

    if args.fp8 or args.fp8_only:
        print(f"downloading {TRANSFORMER_FP8} ...")
        fp8 = hf_hub_download(
            repo_id=REPO,
            filename=TRANSFORMER_FP8,
            token=token,
            cache_dir=str(hub_cache),
        )
        print(f"transformer_fp8 -> {fp8}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
