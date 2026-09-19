#!/usr/bin/env python3
"""Download Comfy-Org TE/VAE + BFL distilled FP8 DiT for Flux.2-klein 9B."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "cache" / "flux2-klein-9b-comfy"
SPLIT = CACHE_ROOT / "split_files"


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    SPLIT.mkdir(parents=True, exist_ok=True)

    print("TE + VAE from Comfy-Org/vae-text-encorder-for-flux-klein-9b ...", flush=True)
    for rel in (
        "split_files/text_encoders/qwen_3_8b_fp4mixed.safetensors",
        "split_files/vae/flux2-vae.safetensors",
    ):
        path = hf_hub_download(
            repo_id="Comfy-Org/vae-text-encorder-for-flux-klein-9b",
            filename=rel,
            local_dir=str(CACHE_ROOT),
            token=token,
        )
        print(f"  {path}", flush=True)

    print("DiT from black-forest-labs/FLUX.2-klein-9b-fp8 (gated) ...", flush=True)
    dit_dir = SPLIT / "diffusion_models"
    dit_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.2-klein-9b-fp8",
        filename="flux-2-klein-9b-fp8.safetensors",
        local_dir=str(dit_dir),
        token=token,
    )
    print(f"  {path}", flush=True)
    print(f"\nready under {SPLIT}")
    print("sizes:")
    for p in sorted(SPLIT.rglob("*.safetensors")):
        print(f"  {p.relative_to(SPLIT)}  {p.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
