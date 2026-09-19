"""Wan2.1 T2V 1.3B FastAPI (Diffusers WanPipeline).

Text-to-video only — image_base64 is accepted for bench compatibility but ignored.

GET /health
POST /generate  -> video_base64 (mp4) + meta
POST /generate.mp4 -> raw mp4 bytes
"""

from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
PORT = int(os.environ.get("PORT", "8123"))
HEIGHT = int(os.environ.get("HEIGHT", "360"))
WIDTH = int(os.environ.get("WIDTH", "640"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "9"))
FPS = int(os.environ.get("FPS", "16"))
STEPS = int(os.environ.get("STEPS", "50"))
GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", "6.0"))
FLOW_SHIFT = float(os.environ.get("FLOW_SHIFT", "3.0"))
DEVICE_MODE = os.environ.get("DEVICE_MODE", "cuda").strip().lower()
HF_HOME = Path(os.environ.get("HF_HOME", "/cache/huggingface"))

DEFAULT_NEG = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

pipe = None
ready = False
load_s: float | None = None


def load_pipeline():
    global load_s
    t0 = time.perf_counter()
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Wan2.1 T2V")

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=FLOW_SHIFT,
    )
    p = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    p.scheduler = scheduler
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
    HF_HOME.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline()
    ready = True
    yield
    ready = False
    pipe = None


app = FastAPI(title="wan-t2v-1.3b", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    # Accepted for video-bench compatibility; ignored (T2V).
    image_base64: str | None = None
    last_image_base64: str | None = None
    height: int | None = Field(default=None, ge=128, le=1280)
    width: int | None = Field(default=None, ge=128, le=1280)
    num_frames: int | None = Field(default=None, ge=5, le=161)
    fps: int | None = Field(default=None, ge=1, le=60)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None


def _round_frames(n: int) -> int:
    # Wan expects 4k+1 frames.
    if (n - 1) % 4 == 0:
        return n
    return max(5, ((n - 1) // 4) * 4 + 1)


def _round_hw(h: int, w: int) -> tuple[int, int]:
    # Wan / VAE: multiples of 16.
    h = max(128, ((h - 1) // 16 + 1) * 16)
    w = max(128, ((w - 1) // 16 + 1) * 16)
    return h, w


def _vram() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    return {
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
        "vram_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
    }


def _export_mp4(frames_fhwc: np.ndarray, fps: int) -> bytes:
    import imageio.v3 as iio

    buf = io.BytesIO()
    iio.imwrite(buf, frames_fhwc, extension=".mp4", fps=fps, codec="libx264")
    return buf.getvalue()


def _frames_to_uint8(frames) -> np.ndarray:
    # Diffusers may return list[PIL] or np ndarray float/uint8.
    if isinstance(frames, np.ndarray):
        arr = frames
        if arr.dtype != np.uint8:
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
        return arr
    out = []
    for fr in frames:
        if hasattr(fr, "convert"):
            out.append(np.asarray(fr.convert("RGB"), dtype=np.uint8))
        else:
            a = np.asarray(fr)
            if a.dtype != np.uint8:
                a = (a.clip(0, 1) * 255).astype(np.uint8)
            out.append(a)
    return np.stack(out, axis=0)


@app.get("/health")
def health():
    if not ready or pipe is None:
        raise HTTPException(status_code=503, detail="loading")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "backend": "diffusers-wan-t2v",
        "task": "t2v",
        "note": "image_base64 ignored (text-to-video)",
        "defaults": {
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "flow_shift": FLOW_SHIFT,
        },
        "load_s": load_s,
        **_vram(),
    }


@app.post("/generate")
def generate(body: GenerateRequest):
    if not ready or pipe is None:
        raise HTTPException(status_code=503, detail="not ready")

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    h, w = _round_hw(body.height or HEIGHT, body.width or WIDTH)
    num_frames = _round_frames(body.num_frames or NUM_FRAMES)
    fps = body.fps or FPS
    steps = body.steps or STEPS
    guidance = GUIDANCE if body.guidance_scale is None else float(body.guidance_scale)
    neg = body.negative_prompt or DEFAULT_NEG
    seed = 42 if body.seed is None else int(body.seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=neg,
            height=h,
            width=w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        frames = out.frames[0]
    except torch.cuda.OutOfMemoryError as e:
        raise HTTPException(status_code=507, detail=f"OOM: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"generate failed: {type(e).__name__}: {e}"
        ) from e
    elapsed = time.perf_counter() - t0

    video_np = _frames_to_uint8(frames)
    try:
        raw = _export_mp4(video_np, fps)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"export failed: {type(e).__name__}: {e}"
        ) from e

    return {
        "video_base64": base64.b64encode(raw).decode("ascii"),
        "mime": "video/mp4",
        "prompt": prompt,
        "negative_prompt": neg,
        "height": h,
        "width": w,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "image_ignored": body.image_base64 is not None,
        "backend": "diffusers-wan-t2v",
        "latency_s": round(elapsed, 3),
        "frames_per_sec": round(num_frames / elapsed, 3) if elapsed > 0 else None,
        **_vram(),
    }


@app.post("/generate.mp4")
def generate_mp4(body: GenerateRequest):
    out = generate(body)
    data = base64.b64decode(out["video_base64"])
    headers = {
        "X-Latency-S": str(out.get("latency_s")),
        "X-Height": str(out.get("height")),
        "X-Width": str(out.get("width")),
    }
    return Response(content=data, media_type="video/mp4", headers=headers)
