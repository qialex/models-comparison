#!/usr/bin/env python3
"""Build Vast onstart + template.json for kokoro-82m (4GB VRAM / 8GB disk)."""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "kokoro-82m" / "app.py"
OUT_DIR = ROOT / "vast" / "kokoro-82m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

APP_B64 = base64.b64encode(gzip.compress(APP.read_bytes(), compresslevel=9)).decode("ascii")

ONSTART_TMPL = r"""#!/bin/bash
# Vast onstart: Kokoro-82M TTS API (same as local :8131)
set -eu
export DEBIAN_FRONTEND=noninteractive
export PORT="${PORT:-8080}"
export MODEL_ID="${MODEL_ID:-hexgrad/Kokoro-82M}"
export LANG_CODE="${LANG_CODE:-a}"
export DEVICE="${DEVICE:-cuda}"
export DEFAULT_VOICE="${DEFAULT_VOICE:-af_heart}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

echo "[kokoro] installing espeak-ng + pip deps..."
apt-get update -qq
apt-get install -y -qq espeak-ng curl >/dev/null
pip install -q --no-cache-dir \
  'kokoro>=0.9.4' 'misaki[en]>=0.9.4' soundfile \
  'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'huggingface_hub>=0.26'

mkdir -p "$HF_HOME" /workspace
python - <<'PY'
import base64, gzip
from pathlib import Path
b = '___APP_B64___'
Path("/workspace/app.py").write_bytes(gzip.decompress(base64.b64decode(b)))
print("[kokoro] wrote /workspace/app.py")
PY

cd /workspace
echo "[kokoro] starting uvicorn on :$PORT device=$DEVICE"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
"""

ONSTART = ONSTART_TMPL.replace("___APP_B64___", APP_B64)
(OUT_DIR / "onstart.sh").write_text(ONSTART, encoding="utf-8", newline="\n")

TEMPLATE = {
    "name": "kokoro-82m",
    "desc": (
        "Kokoro-82M TTS (hexgrad). FastAPI /health /voices /tts on :8080. "
        "Targets >=4GB VRAM (serial use), 8GB disk. Local recipe :8131."
    ),
    "image": "pytorch/pytorch",
    "tag": "2.6.0-cuda12.4-cudnn9-runtime",
    "env": (
        "-p 8080:8080 "
        "-e PORT=8080 "
        "-e MODEL_ID=hexgrad/Kokoro-82M "
        "-e LANG_CODE=a "
        "-e DEVICE=cuda "
        "-e DEFAULT_VOICE=af_heart "
        "-e HF_HOME=/workspace/hf-cache"
    ),
    "runtype": "ssh",
    "ssh_direct": True,
    "use_ssh": True,
    "recommended_disk_space": 8,
    "private": True,
    "search_params": "gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 rented=False",
    "onstart": ONSTART,
}

(OUT_DIR / "template.json").write_text(
    json.dumps(TEMPLATE, indent=2) + "\n", encoding="utf-8"
)
print("wrote", OUT_DIR / "onstart.sh", "chars", len(ONSTART))
print("wrote", OUT_DIR / "template.json")
