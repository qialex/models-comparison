"""OmniVoice HTTP API (k2-fsa/OmniVoice) — zero-shot voice cloning TTS, 600+ languages.

POST /tts — synthesize wav (base64) from text + a reference voice.
Ships a bundled default reference clip (self-synthesized via Kokoro-82M,
Apache-2.0) so /tts works text-only out of the box; pass ref_audio_base64
(+ optional ref_text) to clone a different voice for that request instead.

NOTE: pretrained weights are CC-BY-NC (non-commercial use only); code is
Apache 2.0. See https://huggingface.co/k2-fsa/OmniVoice.
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
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_ID = os.environ.get("MODEL_ID", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 24000

REF_WAV = Path(__file__).with_name("reference.wav")
REF_TEXT = Path(__file__).with_name("reference.txt").read_text(encoding="utf-8").strip()

model = None
default_prompt = None
ready = False
load_s: float | None = None


def load_model():
    global load_s
    t0 = time.perf_counter()
    from omnivoice import OmniVoice

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but CUDA unavailable")
    dtype = torch.float16 if device == "cuda" else torch.float32
    m = OmniVoice.from_pretrained(REPO_ID, device_map=device, dtype=dtype)
    load_s = round(time.perf_counter() - t0, 2)
    return m


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, default_prompt, ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    model = load_model()
    default_prompt = model.create_voice_clone_prompt(ref_audio=str(REF_WAV), ref_text=REF_TEXT)
    ready = True
    yield
    ready = False
    model = None
    default_prompt = None


app = FastAPI(title="omnivoice", lifespan=lifespan)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str | None = Field(default=None, description="Language name; auto-detected from text if omitted")
    instruct: str | None = Field(default=None, description="Optional natural-language style/tone instruction")
    speed: float | None = Field(default=None, description="Speaking-rate multiplier; model default if omitted")
    duration: float | None = Field(default=None, description="Target duration in seconds, if supported")
    ref_audio_base64: str | None = Field(
        default=None, description="WAV bytes, base64 — clone this voice instead of the bundled default"
    )
    ref_text: str | None = Field(
        default=None, description="Transcript of ref_audio_base64; auto-transcribed (ASR) if omitted"
    )
    return_audio: bool = Field(default=True, description="If false, timing only (no wav payload)")


def _vram() -> dict[str, Any]:
    if not torch.cuda.is_available() or DEVICE != "cuda":
        return {}
    return {
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
    }


def synthesize(req: TtsRequest) -> tuple[np.ndarray, dict]:
    assert model is not None
    t0 = time.perf_counter()
    prompt = default_prompt
    cloned = False
    if req.ref_audio_base64:
        wav_bytes = base64.b64decode(req.ref_audio_base64)
        ref_audio, ref_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        ref_tensor = torch.from_numpy(np.asarray(ref_audio, dtype=np.float32)).reshape(1, -1)
        prompt = model.create_voice_clone_prompt(ref_audio=(ref_tensor, ref_sr), ref_text=req.ref_text)
        cloned = True
    with torch.inference_mode():
        wavs = model.generate(
            text=req.text,
            language=req.language,
            voice_clone_prompt=prompt,
            instruct=req.instruct,
            duration=req.duration,
            speed=req.speed,
        )
    gen_s = time.perf_counter() - t0
    if not wavs:
        raise HTTPException(500, "no audio produced")
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    audio_s = float(len(wav) / SAMPLE_RATE)
    return wav, {
        "generate_s": round(gen_s, 3),
        "audio_s": round(audio_s, 3),
        "rtf": round(gen_s / audio_s, 4) if audio_s > 0 else None,
        "samples": int(len(wav)),
        "sample_rate": SAMPLE_RATE,
        "cloned_voice": cloned,
    }


@app.get("/health")
def health():
    if not ready or model is None:
        raise HTTPException(503, "loading")
    return {
        "status": "ok",
        "model": REPO_ID,
        "device": DEVICE,
        "load_s": load_s,
        "cuda": torch.cuda.is_available(),
        "license_note": "pretrained weights are CC-BY-NC (non-commercial use only)",
        **_vram(),
    }


@app.get("/languages")
def languages():
    if model is None:
        raise HTTPException(503, "loading")
    return {"languages": sorted(model.supported_language_names())}


@app.post("/tts")
def tts(req: TtsRequest):
    if not ready or model is None:
        raise HTTPException(503, "loading")
    try:
        wav, metrics = synthesize(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    out: dict[str, Any] = {
        "language": req.language,
        "device": DEVICE,
        **metrics,
        **_vram(),
    }
    if req.return_audio:
        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format="WAV")
        out["wav_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        out["format"] = "wav"
    return out
