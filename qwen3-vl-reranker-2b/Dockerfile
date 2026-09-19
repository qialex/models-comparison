FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8128 \
    MODEL_ID=Qwen/Qwen3-VL-Reranker-2B \
    HF_HOME=/cache/huggingface \
    RERANK_PROMPT="Retrieve images or text relevant to the user's query." \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8128

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
