#!/usr/bin/env python3
"""Download Qwen-Image-Edit ComfyUI weights for 12GB + --lowvram offload.

DiT: Comfy-Org/Qwen-Image-Edit_ComfyUI (default 2509 int8_convrot ~20.5GB)
TE+VAE: Comfy-Org/Qwen-Image_ComfyUI (fp8 TE ~9.4GB, VAE ~0.25GB)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("QWEN_EDIT_CACHE", ROOT / "cache" / "qwen-image-edit-comfy"))
SPLIT = CACHE_ROOT / "split_files"

EDIT_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
IMAGE_REPO = "Comfy-Org/Qwen-Image_ComfyUI"

DIT_CHOICES = {
    "2509-int8": "split_files/diffusion_models/qwen_image_edit_2509_int8_convrot.safetensors",
    "2509-fp8": "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
    "2511-int8": "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors",
    "2511-fp8mixed": "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
}
TE = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "split_files/vae/qwen_image_vae.safetensors"
LIGHTNING_REPO = "lightx2v/Qwen-Image-Lightning"
LIGHTNING_4STEP = (
    "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default="2509-int8", choices=sorted(DIT_CHOICES))
    ap.add_argument("--no-lightning", action="store_true", help="skip Lightning LoRA")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "output").mkdir(parents=True, exist_ok=True)
    (SPLIT / "loras").mkdir(parents=True, exist_ok=True)

    dit_rel = DIT_CHOICES[args.dit]
    print(f"cache={CACHE_ROOT}", flush=True)
    print(f"DiT {dit_rel} from {EDIT_REPO} ...", flush=True)
    print(hf_hub_download(EDIT_REPO, dit_rel, local_dir=str(CACHE_ROOT), token=token), flush=True)

    print(f"TE {TE} from {IMAGE_REPO} ...", flush=True)
    print(hf_hub_download(IMAGE_REPO, TE, local_dir=str(CACHE_ROOT), token=token), flush=True)

    print(f"VAE {VAE} from {IMAGE_REPO} ...", flush=True)
    print(hf_hub_download(IMAGE_REPO, VAE, local_dir=str(CACHE_ROOT), token=token), flush=True)

    if not args.no_lightning:
        print(f"Lightning LoRA {LIGHTNING_4STEP} from {LIGHTNING_REPO} ...", flush=True)
        p = hf_hub_download(
            LIGHTNING_REPO, LIGHTNING_4STEP, local_dir=str(CACHE_ROOT / "lightning"), token=token
        )
        dst = SPLIT / "loras" / Path(LIGHTNING_4STEP).name
        if not dst.exists() or dst.stat().st_size != Path(p).stat().st_size:
            import shutil

            shutil.copy2(p, dst)
        print(dst, flush=True)

    print("\nsizes:", flush=True)
    for p in sorted(SPLIT.rglob("*.safetensors")):
        print(f"  {p.relative_to(SPLIT)}  {p.stat().st_size / 1e9:.2f} GB", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
