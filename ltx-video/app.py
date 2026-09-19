"""LTX-Video 2B distilled I2V API using the official Lightricks package.

Diffusers LTXImageToVideo / ConditionPipeline + FlowMatchEulerDiscreteScheduler
produced first-frame-only then noise. Official stack loads RectifiedFlowScheduler +
CausalVideoAutoencoder from the distilled safetensors (sampler=from_checkpoint)
and runs the multi-scale config from configs/ltxv-2b-0.9.8-distilled.yaml.

GET /health
POST /generate  -> video_base64 (mp4) + meta
POST /generate.mp4 -> raw mp4 bytes
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("MODEL_ID", "Lightricks/LTX-Video")
TRANSFORMER_FILE = os.environ.get(
    "TRANSFORMER_FILE", "ltxv-2b-0.9.8-distilled.safetensors"
)
SPATIAL_UPSCALER_FILE = os.environ.get(
    "SPATIAL_UPSCALER_FILE", "ltxv-spatial-upscaler-0.9.8.safetensors"
)
PORT = int(os.environ.get("PORT", "8122"))
HEIGHT = int(os.environ.get("HEIGHT", "576"))
WIDTH = int(os.environ.get("WIDTH", "320"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "97"))
FPS = int(os.environ.get("FPS", "24"))
GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", "1.0"))
IMAGE_COND_NOISE = float(os.environ.get("IMAGE_COND_NOISE_SCALE", "0.025"))
DEVICE_MODE = os.environ.get("DEVICE_MODE", "cuda").strip().lower()
HF_HOME = Path(os.environ.get("HF_HOME", "/cache/huggingface"))
PIPELINE_YAML = Path(os.environ.get("PIPELINE_YAML", "/app/pipeline.yaml"))

pipeline = None
pipeline_config: dict | None = None
ready = False
load_s: float | None = None
device: str = "cuda"


def _find_hub_file(filename: str) -> Path | None:
    explicit = os.environ.get("TRANSFORMER_PATH" if "distilled" in filename else "", "").strip()
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    matches = sorted(
        (HF_HOME / "hub").glob(f"**/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _resolve_file(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    found = _find_hub_file(filename)
    if found is not None:
        return str(found)
    return hf_hub_download(MODEL_ID, filename, cache_dir=str(HF_HOME / "hub"))


def _snapshot_dir() -> Path | None:
    hub = HF_HOME / "hub" / f"models--{MODEL_ID.replace('/', '--')}"
    snaps = hub / "snapshots"
    if not snaps.is_dir():
        return None
    cands = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if (c / "text_encoder").is_dir():
            return c
    return cands[0] if cands else None


def load_pipeline():
    global load_s, pipeline_config, device
    t0 = time.perf_counter()

    from ltx_video.inference import (
        create_latent_upsampler,
        create_ltx_video_pipeline,
        load_pipeline_config,
    )
    from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for LTX-Video")
    device = "cuda"

    cfg = load_pipeline_config(str(PIPELINE_YAML))
    # Prefer local HF snapshot for T5 so we do not re-download PixArt.
    snap = _snapshot_dir()
    if snap is not None:
        cfg["text_encoder_model_name_or_path"] = str(snap)

    ckpt = _resolve_file(TRANSFORMER_FILE)
    upscaler = _resolve_file(SPATIAL_UPSCALER_FILE)

    pipe = create_ltx_video_pipeline(
        ckpt_path=ckpt,
        precision=cfg["precision"],
        text_encoder_model_name_or_path=cfg["text_encoder_model_name_or_path"],
        sampler=cfg.get("sampler"),
        device=device,
        enhance_prompt=False,
    )
    latent_upsampler = create_latent_upsampler(upscaler, pipe.device)
    pipe = LTXMultiScalePipeline(pipe, latent_upsampler=latent_upsampler)

    pipeline_config = cfg
    load_s = round(time.perf_counter() - t0, 2)
    return pipe


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipeline, ready
    HF_HOME.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline()
    ready = True
    yield
    ready = False
    pipeline = None


app = FastAPI(title="ltx-video-2b-distilled", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    image_base64: str | None = None
    last_image_base64: str | None = None
    height: int | None = Field(default=None, ge=64, le=1024)
    width: int | None = Field(default=None, ge=64, le=1280)
    num_frames: int | None = Field(default=None, ge=9, le=257)
    fps: int | None = Field(default=None, ge=1, le=60)
    steps: int | None = Field(default=None, ge=1, le=50)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = None


def _decode_image(raw: str | None):
    if raw is None:
        return None
    from PIL import Image

    s = raw.strip()
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1]
    data = base64.b64decode(s)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _round_hw(h: int, w: int) -> tuple[int, int]:
    # LTX requires multiples of 32 (e.g. 9:16 → 320×576).
    h = max(64, ((h - 1) // 32 + 1) * 32)
    w = max(64, ((w - 1) // 32 + 1) * 32)
    return h, w


def _round_frames(n: int) -> int:
    if n % 8 == 1:
        return n
    return max(9, ((n - 2) // 8 + 1) * 8 + 1)


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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        iio.imwrite(
            tmp_path,
            frames_fhwc,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=[
                "-movflags",
                "+faststart",
                "-profile:v",
                "baseline",
                "-level",
                "3.1",
            ],
        )
        return Path(tmp_path).read_bytes()
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _pil_to_conditioning_path(image, path: Path) -> None:
    image.save(path)


@app.get("/health")
def health():
    if not ready or pipeline is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "transformer_file": TRANSFORMER_FILE,
        "backend": "lightricks-ltx-video",
        "defaults": {
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "guidance_scale": GUIDANCE,
            "device_mode": DEVICE_MODE,
        },
        "load_s": load_s,
        **_vram(),
    }


@app.post("/generate")
def generate(body: GenerateRequest):
    if not ready or pipeline is None or pipeline_config is None:
        raise HTTPException(status_code=503, detail="model not ready")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    try:
        first = _decode_image(body.image_base64)
        last = _decode_image(body.last_image_base64)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"invalid image_base64: {type(e).__name__}: {e}"
        ) from e

    if first is None:
        raise HTTPException(status_code=400, detail="image_base64 required for I2V")

    from ltx_video.inference import calculate_padding, prepare_conditioning
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

    req_h = body.height or HEIGHT
    req_w = body.width or WIDTH
    h, w = _round_hw(req_h, req_w)
    num_frames = _round_frames(body.num_frames or NUM_FRAMES)
    fps = body.fps or FPS
    neg = body.negative_prompt or (
        "worst quality, inconsistent motion, blurry, jittery, distorted"
    )
    seed = 42 if body.seed is None else int(body.seed)

    height_padded = ((h - 1) // 32 + 1) * 32
    width_padded = ((w - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
    padding = calculate_padding(h, w, height_padded, width_padded)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Official package expects conditioning media on disk.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        first_path = td_path / "first.png"
        _pil_to_conditioning_path(first.resize((w, h)), first_path)
        media_paths = [str(first_path)]
        start_frames = [0]
        if last is not None:
            last_path = td_path / "last.png"
            _pil_to_conditioning_path(last.resize((w, h)), last_path)
            media_paths.append(str(last_path))
            start_frames.append(max(0, num_frames - 1))

        conditioning_items = prepare_conditioning(
            conditioning_media_paths=media_paths,
            conditioning_strengths=[1.0] * len(media_paths),
            conditioning_start_frames=start_frames,
            height=h,
            width=w,
            num_frames=num_frames,
            padding=padding,
            pipeline=pipeline,
        )

        cfg = dict(pipeline_config)
        stg_mode = cfg.pop("stg_mode", "attention_values")
        if stg_mode.lower() in ("stg_av", "attention_values"):
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode.lower() in ("stg_as", "attention_skip"):
            skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode.lower() in ("stg_r", "residual"):
            skip_layer_strategy = SkipLayerStrategy.Residual
        else:
            skip_layer_strategy = SkipLayerStrategy.TransformerBlock

        # Drop non-pipeline kwargs from yaml.
        for k in (
            "pipeline_type",
            "checkpoint_path",
            "spatial_upscaler_model_path",
            "text_encoder_model_name_or_path",
            "precision",
            "sampler",
            "prompt_enhancement_words_threshold",
            "prompt_enhancer_image_caption_model_name_or_path",
            "prompt_enhancer_llm_model_name_or_path",
        ):
            cfg.pop(k, None)

        offload_to_cpu = DEVICE_MODE in ("offload", "cpu_offload", "sequential")
        gen_device = "cpu" if offload_to_cpu else device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        try:
            images = pipeline(
                **cfg,
                skip_layer_strategy=skip_layer_strategy,
                generator=generator,
                output_type="pt",
                callback_on_step_end=None,
                height=height_padded,
                width=width_padded,
                num_frames=num_frames_padded,
                frame_rate=fps,
                prompt=prompt,
                prompt_attention_mask=None,
                negative_prompt=neg,
                negative_prompt_attention_mask=None,
                media_items=None,
                conditioning_items=conditioning_items,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=IMAGE_COND_NOISE,
                mixed_precision=False,
                offload_to_cpu=offload_to_cpu,
                device=device,
                enhance_prompt=False,
            ).images
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"generate failed: {type(e).__name__}: {e}"
            ) from e
        elapsed = time.perf_counter() - t0

    # Crop padding / extra frames (same as official inference.py).
    pad_left, pad_right, pad_top, pad_bottom = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

    video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

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
        "guidance_scale": GUIDANCE if body.guidance_scale is None else body.guidance_scale,
        "seed": seed,
        "used_last_frame": last is not None,
        "backend": "lightricks-ltx-video",
        "latency_s": round(elapsed, 3),
        "frames_per_sec": round(num_frames / elapsed, 3) if elapsed > 0 else None,
        **_vram(),
    }


@app.post("/generate.mp4")
def generate_mp4(body: GenerateRequest):
    out = generate(body)
    data = base64.b64decode(out["video_base64"])
    headers = {
        "X-Latency-Seconds": str(out["latency_s"]),
        "X-Frames-Per-Sec": str(out.get("frames_per_sec") or ""),
        "X-VRAM-Max-Allocated-MB": str(out.get("vram_max_allocated_mb") or ""),
    }
    return Response(content=data, media_type="video/mp4", headers=headers)
