#!/bin/bash
# Vast onstart: Qwen3.5-4B MTP + ngram-mod (same recipe as local :8112)
set -eu

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-4B-MTP-Q4_K_M.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
# Defaults match local :8112. Override for small cards, e.g. A2000 6GB:
#   N_CTX=1024 N_PARALLEL=4 N_PREDICT=8
N_CTX="${N_CTX:-8192}"
N_THREADS="${N_THREADS:-4}"
N_PREDICT="${N_PREDICT:-1536}"
N_PARALLEL="${N_PARALLEL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

mkdir -p "${MODEL_DIR}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Downloading ${MODEL_FILE}..."
  if [ -n "${HF_TOKEN:-}" ]; then
    curl -fL --retry 5 --retry-delay 2 \
      -H "Authorization: Bearer ${HF_TOKEN}" \
      -o "${MODEL_PATH}.partial" \
      "${MODEL_URL}"
  else
    curl -fL --retry 5 --retry-delay 2 \
      -o "${MODEL_PATH}.partial" \
      "${MODEL_URL}"
  fi
  mv "${MODEL_PATH}.partial" "${MODEL_PATH}"
  echo "Download complete."
else
  echo "Model already present: ${MODEL_PATH}"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/app}"

exec /app/llama-server \
  -m "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  -c "${N_CTX}" \
  -t "${N_THREADS}" \
  -n "${N_PREDICT}" \
  -np "${N_PARALLEL}" \
  -ngl "${N_GPU_LAYERS}" \
  --reasoning off \
  --reasoning-budget 0 \
  --reasoning-format deepseek \
  --jinja \
  -fa on \
  --no-mmproj \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 2 \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 12 \
  --spec-ngram-mod-n-max 48
