"""Download smallest Ampere-safe Comfy-Org Z-Image-Turbo bundle and measure GPU weight footprint."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO = "Comfy-Org/z_image_turbo"
# Ampere-safe smallest practical set (NVFP4 DiT is smaller but needs Blackwell).
FILES = {
    "dit": "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
    "te": "split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors",
    "vae": "split_files/vae/ae.safetensors",
}
# Also fetch NVFP4 for disk-size comparison only (may not run on sm_86).
OPTIONAL = {
    "dit_nvfp4": "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
}

OUT = Path(os.environ.get("ZIMAGE_COMFY_CACHE", "cache/z-image-turbo-comfy"))
OUT.mkdir(parents=True, exist_ok=True)


def download(rel: str) -> Path:
    print(f"download {rel} ...", flush=True)
    p = hf_hub_download(
        REPO,
        rel,
        local_dir=str(OUT),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN") or True,
    )
    path = Path(p)
    print(f"  -> {path} ({path.stat().st_size / 1e9:.2f} GB)", flush=True)
    return path


def load_all_to_cuda(paths: dict[str, Path]) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    held: list[torch.Tensor] = []
    for name, path in paths.items():
        print(f"load {name} -> cuda ...", flush=True)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                held.append(t.to("cuda", non_blocking=True))
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  after {name}: alloc={alloc:.2f} GiB peak={peak:.2f} GiB tensors={len(held)}", flush=True)
    print(
        f"FINAL weights-only: alloc={torch.cuda.memory_allocated()/(1024**3):.2f} GiB "
        f"peak={torch.cuda.max_memory_allocated()/(1024**3):.2f} GiB "
        f"reserved={torch.cuda.memory_reserved()/(1024**3):.2f} GiB",
        flush=True,
    )


def main() -> int:
    paths = {k: download(v) for k, v in FILES.items()}
    disk = sum(p.stat().st_size for p in paths.values()) / 1e9
    print(f"DISK total Ampere-safe trio: {disk:.2f} GB", flush=True)
    try:
        download(OPTIONAL["dit_nvfp4"])
    except Exception as e:
        print(f"optional nvfp4 download skipped: {e}", flush=True)
    load_all_to_cuda(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
