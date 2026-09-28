FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl sox \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8196 \
    MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    DEVICE=cuda \
    DEFAULT_LANGUAGE=English \
    DEFAULT_SPEAKER=Ryan \
    ATTN_IMPL=sdpa \
    HF_HOME=/cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8196

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
