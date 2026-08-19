import os
from contextlib import asynccontextmanager
from os import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field, model_validator
from tokenizers import Tokenizer

MODEL_ID = os.environ.get("MODEL_ID", "SamLowe/roberta-base-go_emotions-onnx")
MODEL_FILE = os.environ.get("MODEL_FILE", "onnx/model_quantized.onnx")
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "SamLowe/roberta-base-go_emotions")
EXPECTED_LABELS = 28

# Ordered labels from go_emotions (model card)
LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

session: ort.InferenceSession | None = None
tokenizer: Tokenizer | None = None
ready = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_model() -> tuple[ort.InferenceSession, Tokenizer]:
    model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILE)
    tok = Tokenizer.from_pretrained(TOKENIZER_ID)
    params = {**tok.padding, "length": None}
    tok.enable_padding(**params)

    options = ort.SessionOptions()
    n = max(1, (cpu_count() or 4) // 2)
    options.inter_op_num_threads = n
    options.intra_op_num_threads = n
    sess = ort.InferenceSession(
        path_or_bytes=model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return sess, tok


def score_texts(texts: list[str]) -> list[list[dict[str, Any]]]:
    assert session is not None and tokenizer is not None
    tokens_obj = tokenizer.encode_batch(texts)
    input_feed = {
        "input_ids": [t.ids for t in tokens_obj],
        "attention_mask": [t.attention_mask for t in tokens_obj],
    }
    # Some exports also require token_type_ids
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        input_feed["token_type_ids"] = [
            getattr(t, "type_ids", None) or [0] * len(t.ids) for t in tokens_obj
        ]

    output_names = [session.get_outputs()[0].name]
    logits = session.run(output_names=output_names, input_feed=input_feed)[0]
    probs = sigmoid(np.asarray(logits, dtype=np.float64))

    results: list[list[dict[str, Any]]] = []
    for row in probs:
        if len(row) != EXPECTED_LABELS:
            raise RuntimeError(f"expected {EXPECTED_LABELS} scores, got {len(row)}")
        labels = [
            {"label": LABELS[i], "score": float(row[i])}
            for i in range(EXPECTED_LABELS)
        ]
        labels.sort(key=lambda x: x["score"], reverse=True)
        results.append(labels)
    return results


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global session, tokenizer, ready
    Path(os.environ.get("HF_HOME", "/cache/huggingface")).mkdir(parents=True, exist_ok=True)
    session, tokenizer = load_model()
    ready = True
    yield
    ready = False
    session = None
    tokenizer = None


app = FastAPI(title="go_emotions ONNX", lifespan=lifespan)


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
    if not ready or session is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "file": MODEL_FILE,
        "labels": EXPECTED_LABELS,
    }


@app.post("/classify")
def classify(body: ClassifyRequest):
    if not ready or session is None:
        raise HTTPException(status_code=503, detail="model not ready")

    inputs = [body.text] if body.text is not None else list(body.texts or [])
    if not inputs or any(not isinstance(t, str) or not t.strip() for t in inputs):
        raise HTTPException(status_code=400, detail="text(s) must be non-empty strings")

    return {"results": score_texts(inputs)}
