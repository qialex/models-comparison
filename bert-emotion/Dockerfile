FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (much smaller than CUDA wheels)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8096 \
    MODEL_ID=boltuix/bert-emotion \
    HF_HOME=/cache/huggingface

EXPOSE 8096

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
