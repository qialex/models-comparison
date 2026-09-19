# Local LLM servers (CPU llama.cpp + optional GPU) + emotion classifiers

OpenAI-compatible APIs via **llama.cpp** GGUF (CPU by default; GPU services use CUDA), plus emotion classifiers (`go_emotions`, `bert_emotion`) and VAD regressors (`emobank_vad`, `vad_bert`).

| Service | Model | Port | Quant | RAM cap |
| --- | --- | --- | --- | --- |
| `qwen` | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | **8080** | Q4_K_M | 2g |
| `qwen35` | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | **8081** | Q4_K_M (text) | 2g |
| `lfm25` | [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | **8082** | Q4_0 (~1.59GB) | 3g |
| `qwen35_2b` | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | **8083** | Q4_K_M (~1.28GB) | 3g |
| `qwen35_2b_gpu` | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (CUDA, profile `gpu`) | **8100** | Q4_K_M (~1.28GB) | GPU VRAM |
| `qwen35_4b_gpu` | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (CUDA, profile `gpu`) | **8102** | Q4_K_M (~2.74GB) | GPU VRAM |
| `qwen35_4b_nommproj_gpu` | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (same as `:8102` + `--no-mmproj`) | **8111** | Q4_K_M (~2.74GB) | GPU VRAM |
| `qwen35_4b_spec_gpu` | [Qwen3.5-4B MTP](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF) (`draft-mtp`, n_max=2, CUDA, profile `gpu`) | **8110** | MTP Q4_K_M | GPU VRAM |
| `qwen35_4b_ngram_gpu` | [Qwen3.5-4B MTP](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF) (`draft-mtp,ngram-mod`, CUDA, profile `gpu`) | **8112** | MTP Q4_K_M | GPU VRAM |
| `qwen35_4b_ngram_np4_gpu` | Same as `:8112` but `N_PARALLEL=4` (concurrency test) | **8115** | MTP Q4_K_M | GPU VRAM |
| `ministral3_gpu` | [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) (CUDA, profile `gpu`) | **8114** | Q4_K_M (~2.15GB) | GPU VRAM |
| `granite41_8b_gpu` | [Granite 4.1 8B](https://huggingface.co/ibm-granite/granite-4.1-8b) (CUDA, profile `gpu`) | **8113** | Q4_K_M (~5.35GB) | GPU VRAM |
| `qwen35_4b_dflash_gpu` | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) + [DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) ([AtomicChat GGUF](https://huggingface.co/AtomicChat/Qwen3.5-4B-DFlash-GGUF), CUDA, profile `gpu`) | **8109** | Q4_K_M + DFlash Q8_0 (~2.74+0.69GB) | GPU VRAM |
| `flux2_klein_gpu` | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) text-to-image (CUDA, profile `gpu`) | **8116** | BF16 + CPU offload | GPU VRAM (~13GB official; 512² on 12GB) |
| `flux2_klein_9b_kv_gpu` | [FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) (gated, NCL) | **8126** | BF16 + sequential offload | GPU VRAM (~29GB official; 12GB = experiment only) |
| `flux2_klein_9b_kv_int8_gpu` | [9b-kv INT8 quanto](https://huggingface.co/albex123/flux2-klein-kv-qint8-offload) TE↔DiT swap | **8127** | INT8 + component swap | ~12GB target (not full dual-resident) |
| `qwen3_vl_reranker_2b_gpu` | [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) multimodal rerank | **8128** | BF16 | GPU VRAM (~2B VL) |
| `qwen3_reranker_0_6b_gpu` | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) text rerank | **8129** | BF16 | GPU VRAM (~0.6B) |
| `juggernaut_xl_gpu` | [Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) SDXL text-to-image (CUDA, profile `gpu`) | **8117** | FP16 | GPU VRAM (~8GB) |
| `juggernaut_xi_gpu` | [Juggernaut-XI-v11](https://huggingface.co/RunDiffusion/Juggernaut-XI-v11) SDXL (gated, CC BY-NC-ND) | **8125** | FP16 | GPU VRAM (~8GB; needs `HF_TOKEN`) |
| `realistic_vision_gpu` | [Realistic_Vision_V5.1](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE) SD1.5 + MSE VAE (CUDA, profile `gpu`) | **8118** | FP16 | GPU VRAM (~2–4GB) |
| `animagine_xl_gpu` | [Animagine XL 4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0) anime SDXL (CUDA, profile `gpu`) | **8119** | FP16 | GPU VRAM (~8GB) |
| `sdxl_turbo_gpu` | [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) distilled SDXL (CUDA, profile `gpu`) | **8120** | FP16 | GPU VRAM (~6–8GB) |
| `z_image_turbo_gpu` | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 6B DiT turbo (CUDA, profile `gpu`) | **8121** | BF16 + offload | GPU VRAM (~16GB official; 512² + offload on 12GB) |
| `ltx_video_gpu` | [LTX-Video 2B distilled](https://huggingface.co/Lightricks/LTX-Video) I2V (CUDA, profile `gpu`) | **8122** | BF16 + offload | GPU VRAM (12GB; stop other GPU image/video svcs) |
| `qwen35_9b_gpu` | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (CUDA, profile `gpu`) | **8103** | Q4_K_M (~5.68GB) | GPU VRAM |
| `qwen35_9b_mmproj_gpu` | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) + Unsloth `mmproj-F16` (vision) | **8132** | Q4_K_M + mmproj (~5.68+0.88GB) | GPU VRAM (solo on 12GB) |
| `phi4_mini_gpu` | [Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning) ([Bartowski GGUF](https://huggingface.co/bartowski/microsoft_Phi-4-mini-reasoning-GGUF), CUDA, profile `gpu`) | **8107** | Q4_K_M (~2.49GB) | GPU VRAM |
| `ministral3` | [Ministral-3-3B-Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF) | **8084** | Q4_K_M (~2.15GB, text) | **4g** |
| `minicpm5` | [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | **8085** | Q4_K_M (~657MB) | 2g |
| `gemma3` | [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) | **8086** | Q4_K_M (~806MB) | 2g |
| `llama32` | [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | **8087** | Q4_K_M (~808MB) | 2g |
| `qwen17` | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | **8088** | Q4_K_M (~1.1GB) | 3g |
| `openelm` | [OpenELM-1.1B-Instruct](https://huggingface.co/apple/OpenELM-1_1B-Instruct) | **8089** | Q4_K_M (~0.7GB) | 2g |
| `bonsai17` | [Bonsai-1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) (GGUF Q1_0) | **8091** | Q1_0 (~248MB) | 2g |
| `bonsai4` | [Bonsai-4B](https://huggingface.co/prism-ml/Bonsai-4B-gguf) | **8092** | Q1_0 (~572MB) | 2g |
| `bonsai8` | [Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) | **8093** | Q1_0 (~1.16GB) | 3g |
| `bonsai8_gpu` | [Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) (CUDA, profile `gpu`) | **8099** | Q1_0 (~1.16GB) | GPU VRAM |
| `ternary_bonsai8_gpu` | [Ternary-Bonsai-8B](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf) (Prism CUDA, profile `gpu`) | **8104** | Q2_0 (~2.18GB) | GPU VRAM |
| `ternary_bonsai4_gpu` | [Ternary-Bonsai-4B](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf) (Prism CUDA, profile `gpu`) | **8105** | Q2_0 (~1.07GB) | GPU VRAM |
| `ternary_bonsai17_gpu` | [Ternary-Bonsai-1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) (Prism CUDA, profile `gpu`) | **8106** | Q2_0 (~0.46GB) | GPU VRAM |
| `tinyllama` | [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | **8094** | Q4_K_M (~669MB) | 2g |
| `go_emotions` | [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) ([ONNX INT8](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx)) | **8095** | INT8 (~125MB) | 1g |
| `bert_emotion` | [bert-emotion](https://huggingface.co/boltuix/bert-emotion) | **8096** | FP32 (~20MB) | 1g |
| `emobank_vad` | [deberta-v3-base-emobank-vad](https://huggingface.co/BenRongey/deberta-v3-base-emobank-vad) (PEFT LoRA on DeBERTa-v3-base, VAD) | **8097** | FP32 + LoRA | 3g |
| `vad_bert` | [vad-bert](https://huggingface.co/RobroKools/vad-bert) (VAD regression, raw logits) | **8098** | FP32 (~420MB) | 2g |
| `bonsai27` | [Bonsai-27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) (profile `bonsai27`, no mmproj) | **8090** | Q1_0 (~3.5–3.9GB) | **8g** |
| `bonsai27_gpu` | [Bonsai-27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) (CUDA, profile `gpu`, no mmproj) | **8101** | Q1_0 (~3.5–3.9GB) | GPU VRAM (~5 GB at 2K ctx) |
| `bonsai27_spec_gpu` | [Bonsai-27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) + DSpark drafter (Prism CUDA, profile `gpu`) | **8108** | Q1_0 + dspark Q4_1 (~3.8+1.79GB) | GPU VRAM |

## Prerequisites

- Docker Desktop with Compose
- NVIDIA GPU + Docker GPU support for GPU services (profile `gpu`)

## Start

```powershell
copy .env.example .env
docker compose up -d --build
docker compose logs -f
```

Start one only:

```powershell
docker compose up -d qwen
docker compose up -d qwen35
docker compose up -d lfm25
docker compose up -d qwen35_2b
docker compose up -d ministral3
docker compose up -d minicpm5
docker compose up -d gemma3
docker compose up -d llama32
docker compose up -d qwen17
docker compose up -d openelm
docker compose up -d bonsai17
docker compose up -d bonsai4
docker compose up -d bonsai8
docker compose up -d tinyllama
# NVIDIA GPU: docker compose --profile gpu up -d --build qwen35_2b_gpu qwen35_4b_gpu qwen35_4b_nommproj_gpu qwen35_4b_spec_gpu qwen35_4b_ngram_gpu granite41_8b_gpu qwen35_4b_dflash_gpu qwen35_9b_gpu bonsai8_gpu bonsai27_gpu ternary_bonsai8_gpu ternary_bonsai4_gpu ternary_bonsai17_gpu phi4_mini_gpu
docker compose up -d go_emotions
docker compose up -d bert_emotion
docker compose up -d emobank_vad
docker compose up -d vad_bert
# optional CPU 27B: docker compose --profile bonsai27 up -d bonsai27
```

## Test

```powershell
curl.exe -s http://localhost:8080/v1/models
curl.exe -s http://localhost:8081/v1/models
curl.exe -s http://localhost:8082/v1/models
curl.exe -s http://localhost:8083/v1/models
curl.exe -s http://localhost:8084/v1/models
curl.exe -s http://localhost:8085/v1/models
curl.exe -s http://localhost:8086/v1/models
curl.exe -s http://localhost:8087/v1/models
curl.exe -s http://localhost:8088/v1/models
curl.exe -s http://localhost:8089/v1/models
curl.exe -s http://localhost:8091/v1/models
curl.exe -s http://localhost:8092/v1/models
curl.exe -s http://localhost:8093/v1/models
curl.exe -s http://localhost:8099/v1/models
curl.exe -s http://localhost:8100/v1/models
curl.exe -s http://localhost:8101/v1/models
curl.exe -s http://localhost:8102/v1/models
curl.exe -s http://localhost:8111/v1/models
curl.exe -s http://localhost:8110/v1/models
curl.exe -s http://localhost:8112/v1/models
curl.exe -s http://localhost:8113/v1/models
curl.exe -s http://localhost:8109/v1/models
curl.exe -s http://localhost:8103/v1/models
curl.exe -s http://localhost:8132/v1/models
curl.exe -s http://localhost:8104/v1/models
curl.exe -s http://localhost:8105/v1/models
curl.exe -s http://localhost:8106/v1/models
curl.exe -s http://localhost:8107/v1/models
curl.exe -s http://localhost:8094/v1/models
curl.exe -s http://localhost:8095/health
curl.exe -s http://localhost:8096/health
curl.exe -s http://localhost:8097/health
curl.exe -s http://localhost:8098/health
curl.exe -s http://localhost:8116/health
```

```powershell
@'
{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"temperature":0.1}
'@ | Set-Content -Encoding ascii req.json
curl.exe -s http://localhost:8094/v1/chat/completions -H "Content-Type: application/json" --data-binary "@req.json"
```

```powershell
@'
{"text":"I am not having a great day"}
'@ | Set-Content -Encoding ascii classify.json
curl.exe -s http://localhost:8095/classify -H "Content-Type: application/json" --data-binary "@classify.json"
curl.exe -s http://localhost:8096/classify -H "Content-Type: application/json" --data-binary "@classify.json"
curl.exe -s http://localhost:8097/predict -H "Content-Type: application/json" --data-binary "@classify.json"
curl.exe -s http://localhost:8098/predict -H "Content-Type: application/json" --data-binary "@classify.json"
```

API bases: `http://localhost:8080/v1` … `:8089/v1`, plus `:8091`–`:8094` (Bonsai 1.7B/4B/8B + TinyLlama), `:8099` (Bonsai 8B CUDA), `:8100` (Qwen3.5 2B CUDA), `:8101` (Bonsai 27B CUDA), `:8102` (Qwen3.5 4B CUDA), `:8111` (Qwen3.5 4B CUDA `--no-mmproj`), `:8110` (Qwen3.5 4B CUDA MTP), `:8112` (Qwen3.5 4B CUDA MTP + ngram-mod), `:8113` (Granite 4.1 8B CUDA), `:8109` (Qwen3.5 4B CUDA + DFlash), `:8103` (Qwen3.5 9B CUDA), `:8132` (Qwen3.5 9B CUDA + mmproj vision), `:8104` (Ternary-Bonsai 8B CUDA), `:8105` (Ternary-Bonsai 4B CUDA), `:8106` (Ternary-Bonsai 1.7B CUDA, Prism llama.cpp), and `:8107` (Phi-4-mini-reasoning CUDA) — GPU services use profile `gpu`. Classifiers: `POST :8095/classify` (go_emotions, **28** scores), `POST :8096/classify` (bert-emotion, **13** scores). VAD: `POST :8097/predict` — PEFT DeBERTa (optional quantile); `POST :8098/predict` — [vad-bert](https://huggingface.co/RobroKools/vad-bert) **raw logits only** `{V,A,D}` (no transform). CPU Bonsai-27B is opt-in on `:8090` via profile `bonsai27`.

**GPU:** Docker Desktop needs NVIDIA + WSL2 GPU. GPU services offload all layers (`N_GPU_LAYERS=99`). Same GGUF as CPU counterparts where those exist (`:8083`, `:8093`, `:8090`). Qwen3.5-9B Q4_K_M is ~5.68 GB — stop other GPU servers first on a 12 GB card. Vision variant `:8132` adds Unsloth `mmproj-F16` (~0.88 GB); do not run `:8103` and `:8132` together. Ternary Q2_0 on `:8104`–`:8106` uses `Dockerfile.prism-cuda` ([PrismML fork](https://docs.prismml.com/run/llamacpp)); stock `server-cuda` cannot load it.

**Thinking:** Qwen3 / Qwen3.5 / Ministral / MiniCPM5 use `--reasoning off` (Qwen3 & MiniCPM5 are hybrid think/no-think). LFM uses a patched jinja that pre-closes `<think></think>`. OpenELM uses an Alpaca jinja + reverse-prompts (`### Explanation:` / `### Instruction:`) because the GGUF ChatML template is wrong and the model rarely emits EOS. Mistral recommends temp **below 0.1** for production. OpenELM is under Apple’s AMLR license (research).

## Memory knobs

| Env | Default | Notes |
| --- | --- | --- |
| `N_CTX` | `2048` | Needs ≥ prompt + max_tokens; lower if OOM |
| `N_THREADS` | `4` | Per container — leave headroom if several run |
| `N_PREDICT` | `256` | Max new tokens default |
| `N_PARALLEL` | `1` | Keep 1 so one request gets the full context |
| `N_GPU_LAYERS` | `99` (GPU services) | GPU offload; CPU services stay at `0` |

## Cloud Run (MiniCPM5 1B CPU test)

Curiosity deploy of the same Q4_K_M GGUF: **1 / 2 / 4 vCPU** (Cloud Run has no 3 vCPU). Guide: [`cloudrun/minicpm5/README.md`](cloudrun/minicpm5/README.md).

## Tok/s suite (Bonsai 27B GPU)

[prism-ml/Bonsai-27B-mlx-1bit](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit) is **MLX** (Apple Silicon). This Windows/CUDA stack runs the GGUF twin on `:8101`. Stop other GPU servers first, then:

```powershell
docker compose --profile gpu stop bonsai8_gpu qwen35_2b_gpu qwen35_4b_gpu qwen35_9b_gpu
docker compose --profile gpu up -d bonsai27_gpu
python tests/toksec_bonsai27.py
```

Soak is at least **30 seconds** of generation (`--min-seconds 30`).

DSpark speculative decode (Prism fork, `:8108`) loads Q1_0 plus `Bonsai-27B-dspark-Q4_1.gguf`. It does not replace the 27B:

```powershell
docker compose --profile gpu stop bonsai8_gpu qwen35_2b_gpu qwen35_4b_gpu qwen35_9b_gpu bonsai27_gpu ternary_bonsai8_gpu ternary_bonsai4_gpu ternary_bonsai17_gpu phi4_mini_gpu
docker compose --profile gpu up -d --build bonsai27_spec_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8108
```

Qwen3.5-4B MTP (`:8110`) — see **GPU benchmark notes** below for numbers and tuning.

```powershell
docker compose --profile gpu stop bonsai8_gpu qwen35_2b_gpu qwen35_4b_gpu qwen35_4b_dflash_gpu qwen35_9b_gpu bonsai27_gpu bonsai27_spec_gpu ternary_bonsai8_gpu ternary_bonsai4_gpu ternary_bonsai17_gpu phi4_mini_gpu
docker compose --profile gpu up -d --build qwen35_4b_spec_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8110
```

Qwen3.5-4B MTP + ngram-mod (`:8112`) — recommended 4B setup for agent/code-rewrite loops. Full write-up in **GPU benchmark notes**.

```powershell
docker compose --profile gpu stop bonsai8_gpu qwen35_2b_gpu qwen35_4b_gpu qwen35_4b_spec_gpu qwen35_4b_dflash_gpu qwen35_4b_nommproj_gpu qwen35_9b_gpu bonsai27_gpu bonsai27_spec_gpu ternary_bonsai8_gpu ternary_bonsai4_gpu ternary_bonsai17_gpu phi4_mini_gpu
docker compose --profile gpu up -d --build qwen35_4b_ngram_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8112
```

Qwen3.5-4B + DFlash (`:8109`) uses the same Unsloth Q4_K_M target as `:8102` plus [AtomicChat's](https://huggingface.co/AtomicChat/Qwen3.5-4B-DFlash-GGUF) Q8_0 of [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash). It does not replace `:8102`. Ctx is **32768**. Expect the same prefix-cache tax as DSpark on short scoring suites.

```powershell
docker compose --profile gpu stop bonsai8_gpu qwen35_2b_gpu qwen35_4b_gpu qwen35_9b_gpu bonsai27_gpu bonsai27_spec_gpu ternary_bonsai8_gpu ternary_bonsai4_gpu ternary_bonsai17_gpu phi4_mini_gpu
docker compose --profile gpu up -d --build qwen35_4b_dflash_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8109
```

## GPU benchmark notes (RTX 3060 12 GB, exclusive)

All numbers below are from **one GPU server at a time** on this box unless noted. Use `chat_template_kwargs.enable_thinking: false` for Qwen3.5. For decode speed, trust **first-shot unique prompts** — not repeated-prompt benches.

### Qwen3.5-4B speculative decode ladder

| Service | Port | What | Chat 512 | Code 512 | Soak 30s | VRAM idle |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Plain 4B | `:8102` | baseline | 82.6 | — | 82.6 | ~5.6 GB |
| `--no-mmproj` | `:8111` | same GGUF as `:8102` | 83.8 | 83.7 | 83.1 | ~5.4 GB |
| MTP n=2 | `:8110` | Unsloth MTP GGUF | 86.8 | 119 | 91 | ~5.8 GB |
| MTP n=6 | `:8110` | (tested, too high) | 53.5 | 91 | 50 | ~5.8 GB |
| DFlash | `:8109` | separate drafter, ctx 32k | 74.9 | **247** | — | ~8.9 GB |
| 0.8B draft-simple | `:8110` old | removed | 35 | 49 | 34 | — |

**Findings:**

- **`NO_MMPROJ` is not the win.** `:8111` ≈ `:8102`. The MTP bump on `:8110` comes from draft heads, not skipping mmproj.
- **MTP sweet spot is `n_max=2`.** Raising to 6 drafts more tokens than get accepted (~25–55% accept, mean len ~2.1).
- **DFlash helps code, hurts short chat/scoring.** It kills prefix cache (`cache_n=0`); great for long codegen, bad for cached emotional/scene scoring.
- **0.8B draft-simple loses to plain 4B** on this 12 GB card — verify cost > draft savings even at 88% code accept.

### `:8112` — recommended Qwen 4B GPU setup

**What it is:** same Unsloth [MTP GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF) as `:8110`, plus llama.cpp self-speculation:

```text
--spec-type draft-mtp,ngram-mod
--spec-draft-n-max 2
--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 48
-fa on --no-mmproj --reasoning off --reasoning-format deepseek
```

No extra draft file. Inspired by the [Reddit ngram-mod 665% code-edit thread](https://www.reddit.com/r/LocalLLaMA/comments/1sq7grd/speculative_decoding_question_665_speed_increase/) — see [llama.cpp speculative docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

**Measured decode (first-shot vs iterative):**

| Scenario | Decode tok/s | Notes |
| --- | ---: | --- |
| Chat 512 (unique) | **88** | ~same as MTP-only |
| Code 512 (first pass) | **120** | MTP helps |
| Code rewrite (same file) | **209** | ngram-mod fires |
| Tiny edit → dump whole file | **636** | ~95% repeated output |

**Production chat (real app, ~1.9–2.2k prompt + ~350–380 out):** ~**5.4–5.8 s** wall → **~66–70 tok/s end-to-end** including prefill. Matches the ~88 tok/s decode baseline once prompt time is included.

**When to use `:8112` vs `:8110`:**

- **`:8110`** — diverse one-off prompts, scoring suites, general chat.
- **`:8112`** — agent loops that rewrite the same code/file many times; ngram-mod is free speed when output repeats.

**Caveat:** benches that repeat the same 16 prompts (see `tests/parallel_bench.py`) inflate tok/s to 300–600 because ngram-mod + KV cache replay prior answers. Do **not** use those numbers for capacity planning.

```powershell
docker compose --profile gpu stop bonsai8_gpu qwen35_2b_gpu qwen35_4b_gpu qwen35_4b_spec_gpu qwen35_4b_dflash_gpu qwen35_4b_nommproj_gpu qwen35_9b_gpu bonsai27_gpu bonsai27_spec_gpu ternary_bonsai8_gpu ternary_bonsai4_gpu ternary_bonsai17_gpu phi4_mini_gpu
docker compose --profile gpu up -d qwen35_4b_ngram_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8112
python tests/parallel_bench.py --base http://127.0.0.1:8112 --concurrency 4
```

### Parallelism (`N_PARALLEL`)

| Service | Port | Slots | Idle VRAM | Notes |
| --- | --- | ---: | ---: | --- |
| MTP + ngram | `:8112` | 1 | ~5.6 GB | default |
| MTP + ngram | `:8115` | 4 | ~6.2 GB | +~600 MB for 3 extra KV slots @ ctx 8192 |

With **64 requests, concurrency 4, max 128 out** (`tests/parallel_bench.py`):

| Metric | np=1 `:8112` | np=4 `:8115` |
| --- | ---: | ---: |
| Aggregate output tok/s | 305* | 178 |
| Aggregate prompt tok/s | 55* | 34 |
| Requests/sec | 2.73* | 1.61 |
| Avg per-request decode | 390* | 67 |

\* np=1 wins this bench only because the same prompts repeat 4× — one slot keeps warm KV + ngram hash pool. For **unique prompts**, expect ~**88 tok/s decode shared across active slots**, not per slot.

**Rule of thumb:** GPU decode bandwidth is ~80–90 tok/s total on this card. `-np N` does not multiply that; it queues/batches N conversations. Prefill can batch; decode round-robins one token per active slot per cycle.

### Other GPU models (same card, exclusive)

| Model | Port | Soak decode | VRAM idle |
| --- | --- | ---: | ---: |
| Qwen3.5-2B | `:8100` | ~156 | ~3 GB |
| Ministral-3-3B | `:8114` | ~98 | ~5.2 GB |
| Qwen3.5-4B plain | `:8102` | ~83 | ~5.6 GB |
| Granite 4.1 8B | `:8113` | ~50 | ~8.5 GB |
| Qwen3.5-9B | `:8103` | ~53 | ~9 GB |
| Bonsai-27B | `:8101` | ~36 | ~5 GB |

### Small-card prod hint (RTX A2000 6 GB)

Qwen 4B MTP + ngram (`:8112` recipe) with `-c 1024 -np 4 -n 8`:

- Weights ~2.7 GB + KV for 4 slots @ 1024 ≈ fits in 6 GB.
- Plan on **~65–88 tok/s decode** for unique work, not 300+.
- Short outputs (≤8 tokens) are prefill-dominated; parallelism helps throughput more than decode.

**Vast.ai template:** [`vast/qwen35-4b-mtp-ngram/`](vast/qwen35-4b-mtp-ngram/) — paste `onstart.sh` into a new template using `ghcr.io/ggml-org/llama.cpp:server-cuda`, port **8080**.

### FLUX.2 [klein] 9B-KV INT8 (`:8127`)

[albex123/flux2-klein-kv-qint8-offload](https://huggingface.co/albex123/flux2-klein-kv-qint8-offload) — qint8 + **TE↔DiT GPU swap** (VAE stays on GPU). Aims for **~12 GB** without layer-sequential thrash. ~5 s/img class on 3080 Ti. Quant of NCL 9B-KV.

```powershell
docker compose --profile gpu stop flux2_klein_9b_kv_gpu z_image_turbo_gpu flux2_klein_gpu
docker compose --profile gpu up -d --build flux2_klein_9b_kv_int8_gpu
curl.exe -s http://localhost:8127/health
```

**Vast.ai template:** [`vast/flux2-klein-9b-kv-int8/`](vast/flux2-klein-9b-kv-int8/) — **ID 706179** on account. Port **8127**. Needs **≥12 GB VRAM**, **≥32 GB system RAM**, **≥20 GB** instance disk.

### FLUX.2 [klein] 9B Comfy FP4 TE (`:8189`)

ComfyUI pack: smallest TE [`qwen_3_8b_fp4mixed`](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b) (~6.8 GB) + [`flux2-vae`](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b) + gated distilled DiT [`flux-2-klein-9b-fp8`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) (~9.4 GB). **Not** the KV/INT8 FastAPI service — standard klein 9B T2I (4 steps). FP4 TE is native on Blackwell; Ampere emulates. Accept BFL terms + `HF_TOKEN` for the DiT.

```powershell
python flux2-klein-9b-comfy/download_models.py
docker compose --profile gpu stop z_image_comfy_gpu flux2_klein_9b_kv_int8_gpu
docker compose --profile gpu up -d --build flux2_klein_9b_comfy_gpu
python flux2-klein-9b-comfy/smoke_api.py --workflow flux2-klein-9b-comfy/workflow_api_512.json
```

### FLUX.2 [klein] 9B-KV (`:8126`)

[black-forest-labs/FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) — KV-cache multi-ref edit variant (`Flux2KleinKVPipeline`). **Gated** + **FLUX Non-Commercial**. Official **~29 GB** VRAM; on a 12 GB card use **512²** + `DEVICE_MODE=sequential` and expect slow thrashing or OOM. Needs `HF_TOKEN`.

```powershell
docker compose --profile gpu stop flux2_klein_gpu juggernaut_xi_gpu z_image_turbo_gpu ltx_video_gpu
docker compose --profile gpu up -d --build flux2_klein_9b_kv_gpu
curl.exe -s http://localhost:8126/health
python tests/image-bench/run.py --model flux2-klein-9b-kv --backend flux2-klein-9b-kv --base http://127.0.0.1:8126 --height 512 --width 512 --steps 4 --timeout 900
```

### FLUX.2 [klein] 4B (`:8116`)

[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — Apache 2.0 text-to-image (diffusers `Flux2KleinPipeline`). **Not tok/s** — measure **latency / images-per-sec**.

| | |
| --- | --- |
| Official VRAM | **~13 GB** with `enable_model_cpu_offload` (RTX 3090/4070+) |
| This box (3060 **12 GB**) | Use `DEVICE_MODE=sequential` (default). Full `cuda` loads ~15 GB and **thrashes** (~156 s/img at 512²) |
| Measured (sequential, 512², 4 steps) | **~11 s/image** (~0.09 img/s), peak VRAM during gen **~1 GB** (layers streamed) |
| Port | **8116** |

```powershell
docker compose --profile gpu stop qwen35_9b_gpu bonsai27_gpu qwen35_4b_ngram_gpu
docker compose --profile gpu up -d --build flux2_klein_gpu
curl.exe -s http://localhost:8116/health
python tests/flux_latency.py --base http://127.0.0.1:8116
```

```powershell
@'
{"prompt":"A red apple on a wooden table","height":512,"width":512,"steps":4,"seed":0}
'@ | Set-Content -Encoding ascii flux.json
curl.exe -s http://localhost:8116/generate.png -H "Content-Type: application/json" --data-binary "@flux.json" -o flux-out.png
```

Endpoints: `GET /health`, `POST /generate` (JSON + base64), `POST /generate.png` (raw PNG).

### Juggernaut XL v9 (`:8117`)

[RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) — SDXL photoreal (CreativeML Open RAIL-M). Same HTTP API as `:8116`. Defaults: **832×1216**, **30** steps, CFG **5**, DPM++ 2M Karras. Comfortable on **~8 GB** VRAM.

```powershell
docker compose --profile gpu stop flux2_klein_gpu
docker compose --profile gpu up -d --build juggernaut_xl_gpu
curl.exe -s http://localhost:8117/health
python tests/image-bench/run.py --model juggernaut-xl-v9 --backend juggernaut-xl-v9 --base http://127.0.0.1:8117 --height 1216 --width 832 --steps 30
```

### Juggernaut XI v11 (`:8125`)

[RunDiffusion/Juggernaut-XI-v11](https://huggingface.co/RunDiffusion/Juggernaut-XI-v11) — latest SDXL Juggernaut. **Gated** (accept terms + `HF_TOKEN`). License **CC BY-NC-ND 4.0** (non-commercial). Defaults: **832×1216**, **35** steps, CFG **5**, DPM++ 2M Karras.

```powershell
docker compose --profile gpu stop juggernaut_xl_gpu ltx_video_gpu ltx_video_fp8_gpu wan_t2v_gpu
docker compose --profile gpu up -d --build juggernaut_xi_gpu
curl.exe -s http://localhost:8125/health
python tests/image-bench/run.py --model juggernaut-xi-v11 --backend juggernaut-xi-v11 --base http://127.0.0.1:8125 --height 1216 --width 832 --steps 35
```

### Realistic Vision V5.1 (`:8118`)

[SG161222/Realistic_Vision_V5.1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE) — SD 1.5 photoreal + [MSE VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse). Same HTTP API. Defaults: **512×768**, **30** steps, CFG **5**, DPM++ 2M Karras, card negative prompt. Light on VRAM (~2–4 GB).

```powershell
docker compose --profile gpu stop juggernaut_xl_gpu
docker compose --profile gpu up -d --build realistic_vision_gpu
curl.exe -s http://localhost:8118/health
python tests/image-bench/run.py --model realistic-vision-v5.1 --backend realistic-vision-v5.1 --base http://127.0.0.1:8118 --height 768 --width 512 --steps 30
```

### Animagine XL 4.0 (`:8119`)

[cagliostrolab/animagine-xl-4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0) — anime SDXL (CreativeML Open RAIL++-M). Same HTTP API. Defaults: **832×1216**, **28** steps, CFG **5**, Euler a, card negative prompt. Prefers **tag-style** prompts (`1girl, …, masterpiece, high score, great score, absurdres`).

```powershell
docker compose --profile gpu stop realistic_vision_gpu
docker compose --profile gpu up -d --build animagine_xl_gpu
curl.exe -s http://localhost:8119/health
python tests/image-bench/run.py --model animagine-xl-4.0 --backend animagine-xl-4.0 --base http://127.0.0.1:8119 --height 1216 --width 832 --steps 28
```

### SDXL-Turbo (`:8120`)

[stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) — distilled SDXL. Same HTTP API. Defaults: **512×512**, **4** steps, **CFG 0**, VAE [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix). More than ~4 steps often *hurts* Turbo. Prefer 512²; commercial use needs a [Stability license](https://stability.ai/license).

```powershell
docker compose --profile gpu stop animagine_xl_gpu
docker compose --profile gpu up -d --build sdxl_turbo_gpu
curl.exe -s http://localhost:8120/health
python tests/image-bench/run.py --model sdxl-turbo --backend sdxl-turbo --base http://127.0.0.1:8120 --height 512 --width 512 --steps 4
```

### Z-Image-Turbo (`:8121`)

[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — 6B single-stream DiT turbo (~8 NFEs). Same HTTP API. Defaults: **512×512** (card 1024²), **9** steps, **CFG 0**. Official envelope ~**16 GB**; on a 12 GB card we use `DEVICE_MODE=offload`. Needs recent diffusers (`ZImagePipeline`).

```powershell
docker compose --profile gpu stop flux2_klein_gpu sdxl_turbo_gpu
docker compose --profile gpu up -d --build z_image_turbo_gpu
curl.exe -s http://localhost:8121/health
python tests/image-bench/run.py --model z-image-turbo --backend z-image-turbo --base http://127.0.0.1:8121 --height 512 --width 512 --steps 9
```

### LTX-Video 2B distilled (`:8122`)

[Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) — image-to-video (and first+last frame). Use **`ltxv-2b-0.9.8-distilled.safetensors`** only (not 13B / not full 254 GB tree). Selective cache: `cache/ltx-video/` via `python scripts/download_ltx_video.py`. Stop other GPU image/video services first.

```powershell
$env:HF_HOME = "$PWD\cache\ltx-video"
python scripts/download_ltx_video.py
docker compose --profile gpu stop flux2_klein_gpu z_image_turbo_gpu
docker compose --profile gpu up -d --build ltx_video_gpu
curl.exe -s http://localhost:8122/health
python tests/video-bench/run.py --suite all --base http://127.0.0.1:8122 --skip-existing
```

See `tests/video-bench/README.md` for suites A–E.

## Tok/s suite (Ternary-Bonsai GPU)

Q2_0 needs the [PrismML llama.cpp fork](https://docs.prismml.com/run/llamacpp) (`Dockerfile.prism-cuda`). Run **one** GPU server at a time on a 12 GB card:

```powershell
docker compose --profile gpu stop bonsai8_gpu qwen35_2b_gpu qwen35_4b_gpu qwen35_9b_gpu bonsai27_gpu ternary_bonsai8_gpu ternary_bonsai4_gpu ternary_bonsai17_gpu phi4_mini_gpu

docker compose --profile gpu up -d ternary_bonsai8_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8104

docker compose --profile gpu stop ternary_bonsai8_gpu
docker compose --profile gpu up -d ternary_bonsai4_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8105

docker compose --profile gpu stop ternary_bonsai4_gpu
docker compose --profile gpu up -d ternary_bonsai17_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8106

docker compose --profile gpu stop ternary_bonsai17_gpu
docker compose --profile gpu up -d phi4_mini_gpu
python tests/toksec_bonsai27.py --base http://127.0.0.1:8107
```

## Stop

```powershell
docker compose down
```
