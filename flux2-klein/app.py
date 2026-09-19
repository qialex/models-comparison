"""Minimal FLUX.2 [klein] 4B text-to-image API.

Image generation has no tok/s — we report wall latency and images/sec.
Official card: ~13GB VRAM with enable_model_cpu_offload (RTX 3090/4070+).
On a 12GB card, use smaller HEIGHT/WIDTH and CPU offload.
"""

from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
PORT = int(os.environ.get("PORT", "8116"))
HEIGHT = int(os.environ.get("HEIGHT", "512"))
WIDTH = int(os.environ.get("WIDTH", "512"))
STEPS = int(os.environ.get("STEPS", "4"))
GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", "1.0"))
# offload | cuda — offload is required for ~13GB envelope / 12GB cards
DEVICE_MODE = os.environ.get("DEVICE_MODE", "offload").strip().lower()

pipe = None
ready = False
load_s: float | None = None


def load_pipeline():
    global load_s
    t0 = time.perf_counter()
    # Flux2KleinPipeline needs recent diffusers (install from main if missing)
    from diffusers import Flux2KleinPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # Prefer dtype= (torch_dtype deprecated in newer diffusers)
    try:
        p = Flux2KleinPipeline.from_pretrained(MODEL_ID, dtype=dtype)
    except TypeError:
        p = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if torch.cuda.is_available():
        # 12GB cards: full cuda often works at 512²; offload for 1024² / safer peak.
        # sequential_cpu_offload avoids some enable_model_cpu_offload + Flux2 hook bugs.
        if DEVICE_MODE == "cuda":
            p.to("cuda")
        elif DEVICE_MODE == "sequential":
            p.enable_sequential_cpu_offload()
        else:
            try:
                p.enable_model_cpu_offload()
            except Exception:
                p.enable_sequential_cpu_offload()
    load_s = round(time.perf_counter() - t0, 2)
    return p


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipe, ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for FLUX.2-klein-4B")
    pipe = load_pipeline()
    ready = True
    yield
    ready = False
    pipe = None


app = FastAPI(title="flux2-klein-4b", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    # Optional reference image(s) for FLUX.2 multi-ref / identity edit.
    image_base64: str | list[str] | None = None
    height: int | None = Field(default=None, ge=256, le=1024)
    width: int | None = Field(default=None, ge=256, le=1024)
    steps: int | None = Field(default=None, ge=1, le=8)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=10.0)
    seed: int | None = None
    # png | jpeg
    format: str = Field(default="png")


def _decode_ref_images(raw: str | list[str] | None):
    if raw is None:
        return None
    from PIL import Image

    items = raw if isinstance(raw, list) else [raw]
    out = []
    for item in items:
        s = item.strip()
        if "," in s and s.lower().startswith("data:"):
            s = s.split(",", 1)[1]
        data = base64.b64decode(s)
        out.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return out[0] if len(out) == 1 else out


def _vram() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    return {
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
        "vram_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
    }


@app.get("/health")
def health():
    if not ready or pipe is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device_mode": DEVICE_MODE,
        "defaults": {"height": HEIGHT, "width": WIDTH, "steps": STEPS},
        "load_s": load_s,
        **_vram(),
    }


@app.post("/generate")
def generate(body: GenerateRequest):
    if not ready or pipe is None:
        raise HTTPException(status_code=503, detail="model not ready")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    h = body.height or HEIGHT
    w = body.width or WIDTH
    steps = body.steps or STEPS
    guidance = GUIDANCE if body.guidance_scale is None else body.guidance_scale
    fmt = body.format.lower().strip()
    if fmt not in ("png", "jpeg", "jpg"):
        raise HTTPException(status_code=400, detail="format must be png or jpeg")
    if fmt == "jpg":
        fmt = "jpeg"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # With CPU offload, keep generator on CPU to avoid device mismatch.
    gen_device = "cpu" if DEVICE_MODE != "cuda" else "cuda"
    gen = None
    if body.seed is not None:
        gen = torch.Generator(device=gen_device).manual_seed(int(body.seed))

    try:
        ref = _decode_ref_images(body.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image_base64: {type(e).__name__}: {e}") from e

    t0 = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": h,
            "width": w,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "generator": gen,
        }
        if ref is not None:
            kwargs["image"] = ref
        image = pipe(**kwargs).images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate failed: {type(e).__name__}: {e}") from e
    elapsed = time.perf_counter() - t0

    buf = io.BytesIO()
    image.save(buf, format=fmt.upper())
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")

    return {
        "image_base64": b64,
        "mime": f"image/{fmt}",
        "prompt": prompt,
        "height": h,
        "width": w,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": body.seed,
        "used_reference_image": ref is not None,
        "latency_s": round(elapsed, 3),
        "images_per_sec": round(1.0 / elapsed, 3) if elapsed > 0 else None,
        **_vram(),
    }


@app.post("/generate.png")
def generate_png(body: GenerateRequest):
    """Same as /generate but returns raw PNG bytes (for curl --output)."""
    out = generate(body)
    data = base64.b64decode(out["image_base64"])
    headers = {
        "X-Latency-Seconds": str(out["latency_s"]),
        "X-Images-Per-Sec": str(out.get("images_per_sec") or ""),
        "X-VRAM-Max-Allocated-MB": str(out.get("vram_max_allocated_mb") or ""),
    }
    return Response(content=data, media_type="image/png", headers=headers)
