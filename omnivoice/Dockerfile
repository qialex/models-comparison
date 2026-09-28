FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py reference.wav reference.txt .

ENV PORT=8198 \
    MODEL_ID=k2-fsa/OmniVoice \
    DEVICE=cuda \
    HF_HOME=/cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8198

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
