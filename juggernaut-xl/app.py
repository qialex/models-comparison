"""Minimal Juggernaut XL v9 text-to-image API (SDXL).

Same shape as flux2-klein: GET /health, POST /generate, POST /generate.png.
Official: comfortable on ~8GB VRAM. Recommended 832x1216, 30-40 steps, CFG 3-7.
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

MODEL_ID = os.environ.get("MODEL_ID", "RunDiffusion/Juggernaut-XL-v9")
PORT = int(os.environ.get("PORT", "8117"))
HEIGHT = int(os.environ.get("HEIGHT", "1216"))
WIDTH = int(os.environ.get("WIDTH", "832"))
STEPS = int(os.environ.get("STEPS", "30"))
GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", "5.0"))
DEVICE_MODE = os.environ.get("DEVICE_MODE", "cuda").strip().lower()

pipe = None
ready = False
load_s: float | None = None


def load_pipeline():
    global load_s
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        p = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        )
    except Exception:
        p = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    # DPM++ 2M Karras (model card recommendation)
    p.scheduler = DPMSolverMultistepScheduler.from_config(
        p.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    if torch.cuda.is_available():
        if DEVICE_MODE == "sequential":
            p.enable_sequential_cpu_offload()
        elif DEVICE_MODE == "offload":
            p.enable_model_cpu_offload()
        else:
            p.to("cuda")
            try:
                p.enable_attention_slicing()
            except Exception:
                pass
    load_s = round(time.perf_counter() - t0, 2)
    return p


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipe, ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Juggernaut-XL-v9")
    pipe = load_pipeline()
    ready = True
    yield
    ready = False
    pipe = None


app = FastAPI(title="juggernaut-xl-v9", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    height: int | None = Field(default=None, ge=512, le=1536)
    width: int | None = Field(default=None, ge=512, le=1536)
    steps: int | None = Field(default=None, ge=1, le=50)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None
    format: str = Field(default="png")


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
        "defaults": {"height": HEIGHT, "width": WIDTH, "steps": STEPS, "guidance_scale": GUIDANCE},
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

    gen_device = "cpu" if DEVICE_MODE != "cuda" else "cuda"
    gen = None
    if body.seed is not None:
        gen = torch.Generator(device=gen_device).manual_seed(int(body.seed))

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": h,
        "width": w,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "generator": gen,
    }
    if body.negative_prompt:
        kwargs["negative_prompt"] = body.negative_prompt

    t0 = time.perf_counter()
    try:
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
        "latency_s": round(elapsed, 3),
        "images_per_sec": round(1.0 / elapsed, 3) if elapsed > 0 else None,
        **_vram(),
    }


@app.post("/generate.png")
def generate_png(body: GenerateRequest):
    out = generate(body)
    data = base64.b64decode(out["image_base64"])
    headers = {
        "X-Latency-Seconds": str(out["latency_s"]),
        "X-Images-Per-Sec": str(out.get("images_per_sec") or ""),
        "X-VRAM-Max-Allocated-MB": str(out.get("vram_max_allocated_mb") or ""),
    }
    return Response(content=data, media_type="image/png", headers=headers)
