FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8118 \
    MODEL_ID=SG161222/Realistic_Vision_V5.1_noVAE \
    VAE_ID=stabilityai/sd-vae-ft-mse \
    HF_HOME=/cache/huggingface \
    HEIGHT=768 \
    WIDTH=512 \
    STEPS=30 \
    GUIDANCE_SCALE=5.0 \
    DEVICE_MODE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8118

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
