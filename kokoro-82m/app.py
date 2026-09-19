"""Kokoro-82M TTS HTTP API (hexgrad/Kokoro-82M).

POST /tts — synthesize wav (base64) or return timing-only metrics.
Presets: 54 voices. Tunables: speed, voice mix (comma-avg), lang_code, split_pattern.
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

REPO_ID = os.environ.get("MODEL_ID", "hexgrad/Kokoro-82M")
LANG_CODE = os.environ.get("LANG_CODE", "a")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "af_heart")
SAMPLE_RATE = 24000

pipeline = None
ready = False
load_s: float | None = None


def load_pipeline():
    global load_s
    t0 = time.perf_counter()
    from kokoro import KPipeline

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but CUDA unavailable")
    pipe = KPipeline(lang_code=LANG_CODE, repo_id=REPO_ID, device=device)
    load_s = round(time.perf_counter() - t0, 2)
    return pipe


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipeline, ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline()
    # warm voice download
    pipeline.load_voice(DEFAULT_VOICE)
    ready = True
    yield
    ready = False
    pipeline = None


app = FastAPI(title="kokoro-82m", lifespan=lifespan)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field(default=DEFAULT_VOICE, description="Preset voice or comma-mix e.g. af_bella,af_jessica")
    speed: float = Field(default=1.0, gt=0.1, le=4.0)
    split_pattern: str | None = Field(
        default=r"\n+",
        description="Regex to split text into segments; null = no split",
    )
    return_audio: bool = Field(default=True, description="If false, timing only (no wav payload)")


def _vram() -> dict[str, Any]:
    if not torch.cuda.is_available() or DEVICE != "cuda":
        return {}
    return {
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
    }


def synthesize(text: str, voice: str, speed: float, split_pattern: str | None) -> tuple[np.ndarray, dict]:
    assert pipeline is not None
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    n_segments = 0
    with torch.inference_mode():
        for _gs, _ps, audio in pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern,
        ):
            n_segments += 1
            if audio is None:
                continue
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    gen_s = time.perf_counter() - t0
    if not chunks:
        raise HTTPException(500, "no audio produced")
    wav = np.concatenate(chunks)
    audio_s = float(len(wav) / SAMPLE_RATE)
    return wav, {
        "generate_s": round(gen_s, 3),
        "audio_s": round(audio_s, 3),
        "rtf": round(gen_s / audio_s, 4) if audio_s > 0 else None,
        "segments": n_segments,
        "samples": int(len(wav)),
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/health")
def health():
    if not ready or pipeline is None:
        raise HTTPException(503, "loading")
    return {
        "status": "ok",
        "model": REPO_ID,
        "device": DEVICE,
        "lang_code": LANG_CODE,
        "default_voice": DEFAULT_VOICE,
        "load_s": load_s,
        "cuda": torch.cuda.is_available(),
        **_vram(),
    }


@app.get("/voices")
def voices():
    """Documented presets (v1.0). Mixing: comma-separate names to average embeddings."""
    return {
        "note": "54 presets; mix with comma e.g. af_bella,af_jessica. Tunables: speed, split_pattern, lang_code.",
        "tunables": {
            "voice": "preset name or comma-averaged mix",
            "speed": "float (or callable in library); default 1.0",
            "split_pattern": "regex; default \\n+",
            "lang_code": "a/b/e/f/h/i/p/j/z (pipeline init)",
            "phonemes": "generate_from_tokens bypasses G2P (library only)",
        },
        "american_en": [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
            "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
            "am_onyx", "am_puck", "am_santa",
        ],
        "british_en": [
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        ],
    }


@app.post("/tts")
def tts(req: TtsRequest):
    if not ready or pipeline is None:
        raise HTTPException(503, "loading")
    try:
        wav, metrics = synthesize(req.text, req.voice, req.speed, req.split_pattern)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    out: dict[str, Any] = {
        "voice": req.voice,
        "speed": req.speed,
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
