FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY pipeline.yaml .

ENV PORT=8122 \
    MODEL_ID=Lightricks/LTX-Video \
    TRANSFORMER_FILE=ltxv-2b-0.9.8-distilled.safetensors \
    SPATIAL_UPSCALER_FILE=ltxv-spatial-upscaler-0.9.8.safetensors \
    HF_HOME=/cache/huggingface \
    HEIGHT=640 \
    WIDTH=360 \
    NUM_FRAMES=9 \
    FPS=24 \
    GUIDANCE_SCALE=1.0 \
    IMAGE_COND_NOISE_SCALE=0.025 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8122

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
