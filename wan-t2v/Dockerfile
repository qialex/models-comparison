FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8123 \
    MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    HF_HOME=/cache/huggingface \
    HEIGHT=360 \
    WIDTH=640 \
    NUM_FRAMES=9 \
    FPS=16 \
    STEPS=50 \
    GUIDANCE_SCALE=6.0 \
    FLOW_SHIFT=3.0 \
    DEVICE_MODE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8123

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
