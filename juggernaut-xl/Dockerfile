FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8117 \
    MODEL_ID=RunDiffusion/Juggernaut-XL-v9 \
    HF_HOME=/cache/huggingface \
    HEIGHT=1216 \
    WIDTH=832 \
    STEPS=30 \
    GUIDANCE_SCALE=5.0 \
    DEVICE_MODE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8117

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
