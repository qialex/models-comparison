"""Minimal T2I + edit API for Qwen-Image-2.1 ComfyUI — multipart → raw PNG.

  uvicorn api:app --host 127.0.0.1 --port 8192

  curl -o out.png http://127.0.0.1:8192/edit.png \
    -F "prompt=put her in a cafe" -F "image=@ref.png" -F "steps=40"

  curl -o out.png http://127.0.0.1:8192/generate.png \
    -F "prompt=a woman in a cafe" -F "width=352" -F "height=512" -F "steps=25"
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8191").rstrip("/")
_REPO = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(
    os.environ.get("COMFY_INPUT_DIR", str(_REPO / "cache" / "qwen-image-2.1-comfy" / "input"))
)
DIT = "qwen_image_2.1_int8_convrot.safetensors"
TE = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "25"))
DEFAULT_RES = int(os.environ.get("DEFAULT_SIZE", "1024"))

app = FastAPI(title="qwen-image-2.1")


def _comfy_ready() -> bool:
    try:
        r = httpx.get(f"{COMFY_URL}/system_stats", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def _queue_and_wait(wf: dict, timeout_s: float) -> bytes:
    r = httpx.post(f"{COMFY_URL}/prompt", json={"prompt": wf}, timeout=60.0)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:2000])
    prompt_id = r.json().get("prompt_id")
    if not prompt_id:
        raise HTTPException(500, f"queue failed: {r.text[:1000]}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = httpx.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30.0).json()
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise HTTPException(500, json.dumps(status)[:2000])
            for m in status.get("messages") or []:
                if m and m[0] == "execution_error":
                    raise HTTPException(500, json.dumps(m[1])[:2000])
            if status.get("completed") or entry.get("outputs"):
                for node_out in (entry.get("outputs") or {}).values():
                    for img in node_out.get("images") or []:
                        vr = httpx.get(
                            f"{COMFY_URL}/view",
                            params={
                                "filename": img["filename"],
                                "subfolder": img.get("subfolder") or "",
                                "type": img.get("type") or "output",
                            },
                            timeout=60.0,
                        )
                        vr.raise_for_status()
                        return vr.content
                raise HTTPException(500, "no image in outputs")
        time.sleep(1.0)
    raise HTTPException(504, "timeout")


def _loaders() -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
        "11": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TE, "type": "qwen_image", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }


def _build_edit(prompt: str, image_name: str, *, steps: int, seed: int, resolution: int) -> dict:
    # Official-style edit graph: TE gets image slots + VAE reference_latents;
    # QwenImage21Cache keeps the edit prefix between steps.
    w = _loaders()
    w["10"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    w["4"] = {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {
            "clip": ["2", 0],
            "prompt": prompt,
            "negative_prompt": "",
            "resolution": resolution,
            "images.image_1": ["10", 0],
            "vae": ["3", 0],
        },
    }
    w["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["11", 0],
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": ["4", 2],
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }
    w["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    w["8"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["7", 0], "filename_prefix": "api_edit"},
    }
    return w


def _build_t2i(prompt: str, *, width: int, height: int, steps: int, seed: int) -> dict:
    w = _loaders()
    resolution = max(width, height)
    w["4"] = {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {
            "clip": ["2", 0],
            "prompt": prompt,
            "negative_prompt": "",
            "resolution": resolution,
        },
    }
    w["5"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    w["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["11", 0],
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": ["5", 0],
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    }
    w["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    w["8"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["7", 0], "filename_prefix": "api_t2i"},
    }
    return w


@app.get("/health")
def health():
    ok = _comfy_ready()
    if not ok:
        raise HTTPException(503, "comfy not ready")
    return {
        "status": "ok",
        "model": "Comfy-Org/Qwen-Image-2.1",
        "comfy": COMFY_URL,
        "defaults": {"steps": DEFAULT_STEPS, "resolution": DEFAULT_RES},
    }


@app.post("/edit.png")
async def edit_png(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    steps: int = Form(DEFAULT_STEPS),
    resolution: int = Form(DEFAULT_RES),
    seed: int | None = Form(None),
    timeout_s: float = Form(1800),
):
    if not _comfy_ready():
        raise HTTPException(503, "comfy not ready")
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image")

    # Prefer Comfy input dir (compose mounts cache/.../input → container input)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "in.png").suffix.lower() or ".png"
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        suffix = ".png"
    name = f"api_{uuid.uuid4().hex}{suffix}"
    (INPUT_DIR / name).write_bytes(data)

    p = prompt.strip()
    # Prefer official instruction style; only inject if user omitted <image1>
    if "<image1>" not in p.lower():
        p = (
            f"Keep the character and pose in <image1> unchanged. {p} "
            f"Preserve facial features, hair, body shape, and outfit from <image1>."
        )
    sid = seed if seed is not None else int(uuid.uuid4().int % (2**31 - 1))
    wf = _build_edit(
        p,
        name,
        steps=max(1, min(80, int(steps))),
        seed=sid,
        resolution=max(256, min(2048, int(resolution))),
    )
    raw = _queue_and_wait(wf, float(timeout_s))
    return Response(content=raw, media_type="image/png")


@app.post("/generate.png")
async def generate_png(
    prompt: str = Form(...),
    width: int = Form(DEFAULT_RES),
    height: int = Form(DEFAULT_RES),
    steps: int = Form(DEFAULT_STEPS),
    seed: int | None = Form(None),
    timeout_s: float = Form(1800),
):
    if not _comfy_ready():
        raise HTTPException(503, "comfy not ready")
    p = prompt.strip()
    if not p:
        raise HTTPException(400, "empty prompt")
    sid = seed if seed is not None else int(uuid.uuid4().int % (2**31 - 1))
    wf = _build_t2i(
        p,
        width=max(256, min(2048, int(width))),
        height=max(256, min(2048, int(height))),
        steps=max(1, min(80, int(steps))),
        seed=sid,
    )
    raw = _queue_and_wait(wf, float(timeout_s))
    return Response(content=raw, media_type="image/png")
