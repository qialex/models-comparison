FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8125 \
    MODEL_ID=RunDiffusion/Juggernaut-XI-v11 \
    HF_HOME=/cache/huggingface \
    HEIGHT=1216 \
    WIDTH=832 \
    STEPS=35 \
    GUIDANCE_SCALE=5.0 \
    DEVICE_MODE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8125

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
