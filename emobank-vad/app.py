import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "BenRongey/deberta-v3-base-emobank-vad")
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "microsoft/deberta-v3-base")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))

tokenizer = None
model = None
quantile_transformer = None
transformer_metadata: dict[str, Any] | None = None
device = "cpu"
ready = False


def load_stack():
    global tokenizer, model, quantile_transformer, transformer_metadata, device
    from peft import PeftModel

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Adapter-only repo — must load base + apply LoRA (num_labels=3 regression head).
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=3,
        problem_type="regression",
    )
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model.to(device)
    model.eval()

    qt_path = hf_hub_download(repo_id=MODEL_ID, filename="quantile_transformer.pkl")
    quantile_transformer = joblib.load(qt_path)

    try:
        meta_path = hf_hub_download(repo_id=MODEL_ID, filename="transformer_metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            transformer_metadata = json.load(f)
    except Exception:
        transformer_metadata = None


def predict_one(text: str, apply_transformation: bool = True) -> dict[str, Any]:
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
        logits = model(**inputs).logits[0].cpu().numpy()

    raw = {
        "V_raw": float(logits[0]),
        "A_raw": float(logits[1]),
        "D_raw": float(logits[2]),
    }

    if apply_transformation and quantile_transformer is not None:
        # Space API: map model outputs (~EmoBank 1–5) → uniform [0, 1] for display.
        # Suite gold is 1–5 — use apply_transformation=false to score on raw logits.
        mapped = quantile_transformer.transform(logits.reshape(1, -1))[0]
        return {
            "V": float(mapped[0]),
            "A": float(mapped[1]),
            "D": float(mapped[2]),
            **raw,
            "transformed": True,
            "text": text,
        }

    return {
        "V": float(logits[0]),
        "A": float(logits[1]),
        "D": float(logits[2]),
        "V_raw": None,
        "A_raw": None,
        "D_raw": None,
        "transformed": False,
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


app = FastAPI(title="emobank-vad", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = Field(default=None)
    apply_transformation: bool = True

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
        "base_model": BASE_MODEL_ID,
        "device": device,
        "peft": True,
        "transformer_loaded": quantile_transformer is not None,
        "transformer_metadata": transformer_metadata,
        "dims": ["V", "A", "D"],
    }


@app.post("/predict")
def predict(body: PredictRequest):
    if not ready or model is None:
        raise HTTPException(status_code=503, detail="model not ready")

    if body.text is not None:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="text must be non-empty")
        return predict_one(body.text, body.apply_transformation)

    texts = list(body.texts or [])
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise HTTPException(status_code=400, detail="texts must be non-empty strings")

    predictions = [predict_one(t, body.apply_transformation) for t in texts]
    return {"predictions": predictions, "count": len(predictions)}
