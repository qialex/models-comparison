FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Flux2KleinPipeline landed recently — install diffusers from main.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git"

COPY app.py .

ENV PORT=8116 \
    MODEL_ID=black-forest-labs/FLUX.2-klein-4B \
    HF_HOME=/cache/huggingface \
    HEIGHT=512 \
    WIDTH=512 \
    STEPS=4 \
    GUIDANCE_SCALE=1.0 \
    DEVICE_MODE=sequential \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8116

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
