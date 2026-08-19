import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "boltuix/bert-emotion")
EXPECTED_LABELS = 13

classifier = None
ready = False


def load_classifier():
    return pipeline(
        task="text-classification",
        model=MODEL_ID,
        top_k=None,
        device=-1,
    )


def normalize_one(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [
        {"label": str(item["label"]), "score": float(item["score"])}
        for item in raw
    ]
    labels.sort(key=lambda x: x["score"], reverse=True)
    if len(labels) != EXPECTED_LABELS:
        raise RuntimeError(f"expected {EXPECTED_LABELS} labels, got {len(labels)}")
    return labels


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global classifier, ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    classifier = load_classifier()
    ready = True
    yield
    ready = False
    classifier = None


app = FastAPI(title="bert-emotion", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = Field(default=None)

    @model_validator(mode="after")
    def require_text(self):
        if self.text is None and not self.texts:
            raise ValueError("provide text or texts")
        if self.text is not None and self.texts is not None:
            raise ValueError("provide text or texts, not both")
        return self


@app.get("/health")
def health():
    if not ready or classifier is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ok", "model": MODEL_ID, "labels": EXPECTED_LABELS}


@app.post("/classify")
def classify(body: ClassifyRequest):
    if not ready or classifier is None:
        raise HTTPException(status_code=503, detail="model not ready")

    inputs = [body.text] if body.text is not None else list(body.texts or [])
    if not inputs or any(not isinstance(t, str) or not t.strip() for t in inputs):
        raise HTTPException(status_code=400, detail="text(s) must be non-empty strings")

    raw = classifier(inputs)
    if inputs and isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = [raw]

    return {"results": [normalize_one(item) for item in raw]}
