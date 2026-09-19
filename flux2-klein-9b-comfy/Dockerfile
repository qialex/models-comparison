FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# comfy_kitchen needs torch custom_op schema support for list[int] (Torch >= 2.7)
RUN pip install --no-cache-dir --upgrade \
    "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu126

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git \
    && pip install --no-cache-dir -r ComfyUI/requirements.txt

WORKDIR /app/ComfyUI

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CC=gcc \
    CXX=g++

EXPOSE 8189

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8189", "--disable-auto-launch"]
