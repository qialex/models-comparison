FROM ghcr.io/ggml-org/llama.cpp:server

USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODEL_DIR=/models \
    N_CTX=2048 \
    N_THREADS=4 \
    N_PREDICT=256 \
    N_PARALLEL=1 \
    HOST=0.0.0.0 \
    LD_LIBRARY_PATH=/app

VOLUME ["/models"]
EXPOSE 8080 8081 8082 8083 8084

ENTRYPOINT ["/entrypoint.sh"]
