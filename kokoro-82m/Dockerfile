FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8130 \
    MODEL_ID=hexgrad/Kokoro-82M \
    LANG_CODE=a \
    DEVICE=cuda \
    DEFAULT_VOICE=af_heart \
    HF_HOME=/cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8130

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
