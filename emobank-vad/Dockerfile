FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8097 \
    MODEL_ID=BenRongey/deberta-v3-base-emobank-vad \
    HF_HOME=/cache/huggingface \
    MAX_LENGTH=512

EXPOSE 8097

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
