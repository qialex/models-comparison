import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "RobroKools/vad-bert")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))

tokenizer = None
model = None
device = "cpu"
ready = False


def load_stack():
    global tokenizer, model, device
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()


def predict_one(text: str) -> dict[str, Any]:
    assert tokenizer is not None and model is not None
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().tolist()
    return {
        "V": float(logits[0]),
        "A": float(logits[1]),
        "D": float(logits[2]),
        "text": text,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    load_stack()
    ready = True
    yield
    ready = False


app = FastAPI(title="vad-bert", lifespan=lifespan)


class PredictRequest(BaseModel):
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
    if not ready or model is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "dims": ["V", "A", "D"],
        "note": "raw model logits; no post-transform",
    }


@app.post("/predict")
def predict(body: PredictRequest):
    if not ready or model is None:
        raise HTTPException(status_code=503, detail="model not ready")

    if body.text is not None:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="text must be non-empty")
        return predict_one(body.text)

    texts = list(body.texts or [])
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise HTTPException(status_code=400, detail="texts must be non-empty strings")

    predictions = [predict_one(t) for t in texts]
    return {"predictions": predictions, "count": len(predictions)}
