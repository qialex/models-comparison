FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Z-Image needs recent diffusers (ZImagePipeline); install from git.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/huggingface/diffusers"

COPY app.py .

ENV PORT=8121 \
    MODEL_ID=Tongyi-MAI/Z-Image-Turbo \
    HF_HOME=/cache/huggingface \
    HEIGHT=512 \
    WIDTH=512 \
    STEPS=9 \
    GUIDANCE_SCALE=0.0 \
    DEVICE_MODE=offload \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8121

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
