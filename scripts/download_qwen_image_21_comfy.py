#!/usr/bin/env python3
"""Download Comfy-Org/Qwen-Image-2.1 weights (int8 DiT + TE + VAE).

Default pack (~17.4 GB):
  diffusion_models/qwen_image_2.1_int8_convrot.safetensors  (~7.3 GB)
  text_encoders/qwen3vl_8b_int8_convrot.safetensors         (~9.4 GB)
  vae/qwen_image_2.1_vae_bf16.safetensors                   (~0.68 GB)

Optional --with-pe adds ~9.5 GB prompt-enhancer TE (T2I).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("QWEN_IMAGE_21_CACHE", ROOT / "cache" / "qwen-image-2.1-comfy"))

REPO = "Comfy-Org/Qwen-Image-2.1"

DIT_CHOICES = {
    "int8": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    "bf16": "diffusion_models/qwen_image_2.1_bf16.safetensors",
}
TE_CHOICES = {
    "int8": "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "w4a8": "text_encoders/qwen3vl_8b_w4a8.safetensors",
    "bf16": "text_encoders/qwen3vl_8b_bf16.safetensors",
}
VAE = "vae/qwen_image_2.1_vae_bf16.safetensors"
PE_T2I = "text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default="int8", choices=sorted(DIT_CHOICES))
    ap.add_argument("--te", default="int8", choices=sorted(TE_CHOICES))
    ap.add_argument("--with-pe", action="store_true", help="also download T2I prompt enhancer TE")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "output").mkdir(parents=True, exist_ok=True)

    files = [DIT_CHOICES[args.dit], TE_CHOICES[args.te], VAE]
    if args.with_pe:
        files.append(PE_T2I)

    print(f"cache={CACHE_ROOT}", flush=True)
    for rel in files:
        print(f"{rel} from {REPO} ...", flush=True)
        print(hf_hub_download(REPO, rel, local_dir=str(CACHE_ROOT), token=token), flush=True)

    print("\nsizes:", flush=True)
    for p in sorted(CACHE_ROOT.rglob("*.safetensors")):
        print(f"  {p.relative_to(CACHE_ROOT)}  {p.stat().st_size / 1e9:.2f} GB", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
