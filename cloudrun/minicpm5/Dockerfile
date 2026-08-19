FROM ghcr.io/ggml-org/llama.cpp:server

USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Bake GGUF into the image so Cloud Run 1Gi does not download 657MB at startup
ARG MODEL_URL=https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/resolve/main/MiniCPM5-1B-Q4_K_M.gguf
RUN mkdir -p /models \
 && curl -fL --retry 5 --retry-delay 2 \
      -o /models/MiniCPM5-1B-Q4_K_M.gguf \
      "${MODEL_URL}"

ENV MODEL_DIR=/models \
    MODEL_FILE=MiniCPM5-1B-Q4_K_M.gguf \
    N_CTX=1024 \
    N_THREADS=1 \
    N_PREDICT=128 \
    N_PARALLEL=1 \
    N_GPU_LAYERS=0 \
    REASONING=off \
    REASONING_BUDGET=0 \
    HOST=0.0.0.0 \
    PORT=8080 \
    LD_LIBRARY_PATH=/app

# Do not declare VOLUME /models — an empty mount would hide the baked GGUF
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
