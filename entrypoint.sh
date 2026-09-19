#!/bin/sh
set -eu

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILE="${MODEL_FILE:-qwen3-0.6b-q4_k_m.gguf}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Antigma/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf}"
N_CTX="${N_CTX:-2048}"
N_THREADS="${N_THREADS:-4}"
N_PREDICT="${N_PREDICT:-256}"
N_PARALLEL="${N_PARALLEL:-1}"
# 0 = CPU only; 99 / -1 = offload all layers (CUDA image)
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
# auto | on | off — Qwen3.5 / LFM often ignore soft /no_think
REASONING="${REASONING:-auto}"
# -1 unrestricted; 0 force end thinking immediately (helps LFM2.5)
REASONING_BUDGET="${REASONING_BUDGET:--1}"
# auto | none | deepseek | deepseek-legacy
REASONING_FORMAT="${REASONING_FORMAT:-auto}"
# Optional custom jinja (e.g. LFM pre-closed </think>)
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
# Comma-separated reverse-prompts / stop strings (e.g. OpenELM: [INST])
REVERSE_PROMPTS="${REVERSE_PROMPTS:-}"
# Optional DSpark / speculative draft GGUF (Prism fork)
DRAFT_FILE="${DRAFT_FILE:-}"
DRAFT_URL="${DRAFT_URL:-}"
SPEC_TYPE="${SPEC_TYPE:-}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-}"
N_GPU_LAYERS_DRAFT="${N_GPU_LAYERS_DRAFT:-99}"
SPEC_NGRAM_MOD_N_MATCH="${SPEC_NGRAM_MOD_N_MATCH:-}"
SPEC_NGRAM_MOD_N_MIN="${SPEC_NGRAM_MOD_N_MIN:-}"
SPEC_NGRAM_MOD_N_MAX="${SPEC_NGRAM_MOD_N_MAX:-}"
# empty | on | off | auto
FLASH_ATTN="${FLASH_ATTN:-}"
# 1 / true / yes → --no-mmproj (vLLM --language-model-only)
NO_MMPROJ="${NO_MMPROJ:-}"
# Optional multimodal projector (download + --mmproj). Ignored if NO_MMPROJ is set.
MMPROJ_FILE="${MMPROJ_FILE:-}"
MMPROJ_URL="${MMPROJ_URL:-}"
CHECKPOINT_MIN_STEP="${CHECKPOINT_MIN_STEP:-}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-}"
N_BATCH="${N_BATCH:-}"
N_UBATCH="${N_UBATCH:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

mkdir -p "${MODEL_DIR}"

download_gguf() {
  dest="$1"
  url="$2"
  label="$3"
  if [ -f "${dest}" ]; then
    echo "Model already present: ${dest}"
    return 0
  fi
  echo "Downloading ${label}..."
  if [ -n "${HF_TOKEN:-}" ]; then
    curl -fL --progress-bar \
      -H "Authorization: Bearer ${HF_TOKEN}" \
      -o "${dest}.partial" \
      "${url}"
  else
    curl -fL --progress-bar \
      -o "${dest}.partial" \
      "${url}"
  fi
  mv "${dest}.partial" "${dest}"
  echo "Download complete."
}

download_gguf "${MODEL_PATH}" "${MODEL_URL}" "${MODEL_FILE}"

DRAFT_PATH=""
if [ -n "${DRAFT_FILE}" ]; then
  DRAFT_PATH="${MODEL_DIR}/${DRAFT_FILE}"
  if [ -z "${DRAFT_URL}" ]; then
    echo "DRAFT_FILE set but DRAFT_URL is empty" >&2
    exit 1
  fi
  download_gguf "${DRAFT_PATH}" "${DRAFT_URL}" "${DRAFT_FILE}"
fi

MMPROJ_PATH=""
if [ -n "${MMPROJ_FILE}" ]; then
  if [ -z "${MMPROJ_URL}" ]; then
    echo "MMPROJ_FILE set but MMPROJ_URL is empty" >&2
    exit 1
  fi
  MMPROJ_PATH="${MODEL_DIR}/${MMPROJ_FILE}"
  download_gguf "${MMPROJ_PATH}" "${MMPROJ_URL}" "${MMPROJ_FILE}"
fi

set -- /app/llama-server \
  -m "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  -c "${N_CTX}" \
  -t "${N_THREADS}" \
  -n "${N_PREDICT}" \
  -np "${N_PARALLEL}" \
  -ngl "${N_GPU_LAYERS}" \
  --reasoning "${REASONING}" \
  --reasoning-budget "${REASONING_BUDGET}" \
  --reasoning-format "${REASONING_FORMAT}" \
  --jinja

if [ -n "${FLASH_ATTN}" ]; then
  set -- "$@" -fa "${FLASH_ATTN}"
fi

# Optional prompt-cache checkpoints (DSpark/DFlash cannot seq_rm; default cms is 256
# but llama.cpp skips mid-prompt snapshots unless batches are small enough to be
# "near the end").
if [ -n "${CHECKPOINT_MIN_STEP}" ]; then
  set -- "$@" --checkpoint-min-step "${CHECKPOINT_MIN_STEP}"
fi
if [ -n "${CTX_CHECKPOINTS}" ]; then
  set -- "$@" --ctx-checkpoints "${CTX_CHECKPOINTS}"
fi
if [ -n "${N_BATCH}" ]; then
  set -- "$@" -b "${N_BATCH}"
fi
if [ -n "${N_UBATCH}" ]; then
  set -- "$@" -ub "${N_UBATCH}"
fi

if [ -n "${NO_MMPROJ}" ]; then
  set -- "$@" --no-mmproj
elif [ -n "${MMPROJ_PATH}" ]; then
  set -- "$@" --mmproj "${MMPROJ_PATH}"
fi

if [ -n "${DRAFT_PATH}" ]; then
  set -- "$@" \
    -md "${DRAFT_PATH}" \
    --spec-type "${SPEC_TYPE:-draft-dspark}" \
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX}" \
    -ngld "${N_GPU_LAYERS_DRAFT}"
  if [ -n "${SPEC_DRAFT_P_MIN}" ]; then
    set -- "$@" --spec-draft-p-min "${SPEC_DRAFT_P_MIN}"
  fi
elif [ -n "${SPEC_TYPE}" ]; then
  # MTP / ngram: spec heads live in the target GGUF (no -md)
  set -- "$@" --spec-type "${SPEC_TYPE}" --spec-draft-n-max "${SPEC_DRAFT_N_MAX}"
  if [ -n "${SPEC_DRAFT_P_MIN}" ]; then
    set -- "$@" --spec-draft-p-min "${SPEC_DRAFT_P_MIN}"
  fi
fi

if [ -n "${SPEC_DRAFT_N_MIN}" ]; then
  set -- "$@" --spec-draft-n-min "${SPEC_DRAFT_N_MIN}"
fi
if [ -n "${SPEC_NGRAM_MOD_N_MATCH}" ]; then
  set -- "$@" --spec-ngram-mod-n-match "${SPEC_NGRAM_MOD_N_MATCH}"
fi
if [ -n "${SPEC_NGRAM_MOD_N_MIN}" ]; then
  set -- "$@" --spec-ngram-mod-n-min "${SPEC_NGRAM_MOD_N_MIN}"
fi
if [ -n "${SPEC_NGRAM_MOD_N_MAX}" ]; then
  set -- "$@" --spec-ngram-mod-n-max "${SPEC_NGRAM_MOD_N_MAX}"
fi

if [ -n "${CHAT_TEMPLATE_FILE}" ]; then
  set -- "$@" --chat-template-file "${CHAT_TEMPLATE_FILE}"
fi

if [ -n "${REVERSE_PROMPTS}" ]; then
  OLD_IFS=$IFS
  IFS=,
  for rp in ${REVERSE_PROMPTS}; do
    set -- "$@" -r "${rp}"
  done
  IFS=$OLD_IFS
fi

exec "$@"
