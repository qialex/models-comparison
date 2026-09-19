"""Qwen3-Reranker-0.6B HTTP API (text relevance / yes-no logits).

POST /score and POST /rank — Sentence Transformers CrossEncoder.
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-Reranker-0.6B")
PORT = int(os.environ.get("PORT", "8129"))
DEFAULT_PROMPT = os.environ.get(
    "RERANK_PROMPT",
    "Given a web search query, retrieve relevant passages that answer the query",
)

model = None
ready = False
load_s: float | None = None


def load_model():
    global load_s
    t0 = time.perf_counter()
    from sentence_transformers import CrossEncoder

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for qwen3-reranker-0.6b")
    m = CrossEncoder(MODEL_ID, device="cuda")
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


app = FastAPI(title="qwen3-reranker-0.6b", lifespan=lifespan)


class ScoreRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)
    prompt: str | None = None
    sigmoid: bool = False


class RankRequest(ScoreRequest):
    top_k: int | None = Field(default=None, ge=1)


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
    if not ready or model is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": "cuda",
        "default_prompt": DEFAULT_PROMPT,
        "load_s": load_s,
        **_vram(),
    }


@app.post("/score")
def score(body: ScoreRequest):
    if not ready or model is None:
        raise HTTPException(status_code=503, detail="model not ready")

    prompt = body.prompt or DEFAULT_PROMPT
    pairs = [(body.query, doc) for doc in body.documents]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {"prompt": prompt}
        if body.sigmoid:
            kwargs["activation_fn"] = torch.nn.Sigmoid()
        scores = model.predict(pairs, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"score failed: {type(e).__name__}: {e}") from e
    elapsed = time.perf_counter() - t0
    score_list = [float(s) for s in scores]
    return {
        "query": body.query,
        "prompt": prompt,
        "sigmoid": body.sigmoid,
        "scores": score_list,
        "latency_s": round(elapsed, 3),
        **_vram(),
    }


@app.post("/rank")
def rank(body: RankRequest):
    if not ready or model is None:
        raise HTTPException(status_code=503, detail="model not ready")

    prompt = body.prompt or DEFAULT_PROMPT
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        rankings = model.rank(body.query, body.documents, prompt=prompt, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rank failed: {type(e).__name__}: {e}") from e
    elapsed = time.perf_counter() - t0
    out = []
    for row in rankings:
        item = dict(row)
        if "score" in item:
            item["score"] = float(item["score"])
        out.append(item)
    return {
        "query": body.query,
        "prompt": prompt,
        "rankings": out,
        "latency_s": round(elapsed, 3),
        **_vram(),
    }
