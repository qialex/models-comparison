FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ffmpeg build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# RTX 3060 = Ampere sm_86. Q8 setup.py normally probes a live GPU (unavailable in
# docker build), so we pin the arch and patch the probe.
ENV TORCH_CUDA_ARCH_LIST="8.6" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda

COPY requirements.txt .
COPY patch_q8_setup.py /tmp/patch_q8_setup.py
COPY patch_q8_ampere.py /tmp/patch_q8_ampere.py

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir packaging wheel ninja setuptools \
    && git clone --depth 1 --recurse-submodules \
         https://github.com/Lightricks/LTX-Video-Q8-Kernels.git /tmp/q8 \
    && python /tmp/patch_q8_setup.py \
    && pip install --no-cache-dir --no-build-isolation /tmp/q8 \
    && python /tmp/patch_q8_ampere.py \
    && rm -rf /tmp/q8 /tmp/patch_q8_setup.py /tmp/patch_q8_ampere.py

COPY app.py .
COPY pipeline.yaml .

ENV PORT=8124 \
    MODEL_ID=Lightricks/LTX-Video \
    TRANSFORMER_FILE=ltxv-2b-0.9.8-distilled-fp8.safetensors \
    SPATIAL_UPSCALER_FILE=ltxv-spatial-upscaler-0.9.8.safetensors \
    HF_HOME=/cache/huggingface \
    HEIGHT=576 \
    WIDTH=320 \
    NUM_FRAMES=97 \
    FPS=24 \
    GUIDANCE_SCALE=1.0 \
    IMAGE_COND_NOISE_SCALE=0.025 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8124

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
