FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git"

COPY pipeline_flux2_klein_kv_offload.py .
COPY app.py .

ENV PORT=8127 \
    MODEL_ID=albex123/flux2-klein-kv-qint8-offload \
    HF_HOME=/cache/huggingface \
    HEIGHT=512 \
    WIDTH=512 \
    STEPS=4 \
    GUIDANCE_SCALE=1.0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8127

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
