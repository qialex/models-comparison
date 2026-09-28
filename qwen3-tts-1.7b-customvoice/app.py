"""Qwen3-TTS-12Hz-1.7B-CustomVoice HTTP API (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).

POST /tts — synthesize wav (base64) from text + preset speaker + language.
9 preset speakers, 10 languages, optional natural-language style `instruct`.
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

REPO_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "English")
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "Ryan")
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")

SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

model = None
ready = False
load_s: float | None = None


def load_model():
    global load_s
    t0 = time.perf_counter()
    from qwen_tts import Qwen3TTSModel

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but CUDA unavailable")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    m = Qwen3TTSModel.from_pretrained(
        REPO_ID,
        device_map=device,
        dtype=dtype,
        attn_implementation=ATTN_IMPL,
    )
    load_s = round(time.perf_counter() - t0, 2)
    return m


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    model = load_model()
    ready = True
    yield
    ready = False
    model = None


app = FastAPI(title="qwen3-tts-1.7b-customvoice", lifespan=lifespan)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default=DEFAULT_LANGUAGE, description=f"One of {LANGUAGES}")
    speaker: str = Field(default=DEFAULT_SPEAKER, description=f"One of {SPEAKERS}")
    instruct: str | None = Field(default=None, description="Optional natural-language style/tone instruction")
    return_audio: bool = Field(default=True, description="If false, timing only (no wav payload)")


def _vram() -> dict[str, Any]:
    if not torch.cuda.is_available() or DEVICE != "cuda":
        return {}
    return {
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
    }


def synthesize(text: str, language: str, speaker: str, instruct: str | None) -> tuple[np.ndarray, int, dict]:
    assert model is not None
    t0 = time.perf_counter()
    with torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
    gen_s = time.perf_counter() - t0
    if not wavs:
        raise HTTPException(500, "no audio produced")
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    audio_s = float(len(wav) / sr)
    return wav, sr, {
        "generate_s": round(gen_s, 3),
        "audio_s": round(audio_s, 3),
        "rtf": round(gen_s / audio_s, 4) if audio_s > 0 else None,
        "samples": int(len(wav)),
        "sample_rate": sr,
    }


@app.get("/health")
def health():
    if not ready or model is None:
        raise HTTPException(503, "loading")
    return {
        "status": "ok",
        "model": REPO_ID,
        "device": DEVICE,
        "attn_impl": ATTN_IMPL,
        "load_s": load_s,
        "cuda": torch.cuda.is_available(),
        **_vram(),
    }


@app.get("/voices")
def voices():
    return {
        "speakers": SPEAKERS,
        "languages": LANGUAGES,
        "tunables": {
            "language": f"one of {LANGUAGES}; default {DEFAULT_LANGUAGE}",
            "speaker": f"one of {SPEAKERS}; default {DEFAULT_SPEAKER}",
            "instruct": "optional free-text tone/style instruction",
        },
    }


@app.post("/tts")
def tts(req: TtsRequest):
    if not ready or model is None:
        raise HTTPException(503, "loading")
    try:
        wav, sr, metrics = synthesize(req.text, req.language, req.speaker, req.instruct)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    out: dict[str, Any] = {
        "language": req.language,
        "speaker": req.speaker,
        "instruct": req.instruct,
        "device": DEVICE,
        **metrics,
        **_vram(),
    }
    if req.return_audio:
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        out["wav_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        out["format"] = "wav"
    return out
