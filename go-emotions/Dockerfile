FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8095 \
    MODEL_ID=SamLowe/roberta-base-go_emotions-onnx \
    MODEL_FILE=onnx/model_quantized.onnx \
    HF_HOME=/cache/huggingface

EXPOSE 8095

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
