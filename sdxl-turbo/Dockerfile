FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8120 \
    MODEL_ID=stabilityai/sdxl-turbo \
    VAE_ID=madebyollin/sdxl-vae-fp16-fix \
    HF_HOME=/cache/huggingface \
    HEIGHT=512 \
    WIDTH=512 \
    STEPS=4 \
    GUIDANCE_SCALE=0.0 \
    DEVICE_MODE=cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8120

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
