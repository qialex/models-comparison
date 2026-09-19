# Image generation bench (prompts + runner are tracked; outputs are not)

## Layout

```text
tests/image-bench/
  prompts_adult_nude.json          # 20 prompts: 01-10 photoreal, 11-20 anime style
  prompts_adult_nude_ceo_harden.json  # same scenes + CEO-review nudity harden
  run.py

results/image-bench/        # gitignored
  <model-name>/
    summary.json
    <prompt_id>/
      prompt.txt
      meta.json
      image.png
```

## Run (FLUX.2-klein 4B on :8116)

```powershell
docker compose --profile gpu up -d flux2_klein_gpu
python tests/image-bench/run.py --model flux2-klein-4b --base http://127.0.0.1:8116
```

## Run (FLUX.2-klein 9B-KV on :8126)

Gated + FLUX Non-Commercial. Official ~29 GB VRAM — 12 GB is experimental (`sequential` offload, 512²). Set `HF_TOKEN` after accepting the model terms.

```powershell
docker compose --profile gpu stop flux2_klein_gpu
docker compose --profile gpu up -d --build flux2_klein_9b_kv_gpu
python tests/image-bench/run.py --model flux2-klein-9b-kv --backend flux2-klein-9b-kv --base http://127.0.0.1:8126 --height 512 --width 512 --steps 4 --timeout 900
```

## Run (Juggernaut XL v9 on :8117)

```powershell
docker compose --profile gpu stop flux2_klein_gpu
docker compose --profile gpu up -d juggernaut_xl_gpu
python tests/image-bench/run.py --model juggernaut-xl-v9 --backend juggernaut-xl-v9 --base http://127.0.0.1:8117 --height 1216 --width 832 --steps 30
```

## Run (Juggernaut XI v11 on :8125)

Gated HF model — accept terms on the model page and set `HF_TOKEN`. License is **CC BY-NC-ND 4.0** (non-commercial).

```powershell
docker compose --profile gpu stop juggernaut_xl_gpu ltx_video_gpu ltx_video_fp8_gpu wan_t2v_gpu
docker compose --profile gpu up -d --build juggernaut_xi_gpu
python tests/image-bench/run.py --model juggernaut-xi-v11 --backend juggernaut-xi-v11 --base http://127.0.0.1:8125 --height 1216 --width 832 --steps 35
```

## Run (Realistic Vision V5.1 on :8118)

```powershell
docker compose --profile gpu stop juggernaut_xl_gpu
docker compose --profile gpu up -d realistic_vision_gpu
python tests/image-bench/run.py --model realistic-vision-v5.1 --backend realistic-vision-v5.1 --base http://127.0.0.1:8118 --height 768 --width 512 --steps 30
```

## Run (Animagine XL 4.0 on :8119)

```powershell
docker compose --profile gpu stop realistic_vision_gpu
docker compose --profile gpu up -d animagine_xl_gpu
python tests/image-bench/run.py --model animagine-xl-4.0 --backend animagine-xl-4.0 --base http://127.0.0.1:8119 --height 1216 --width 832 --steps 28
```

## Run (SDXL-Turbo on :8120)

```powershell
docker compose --profile gpu stop animagine_xl_gpu
docker compose --profile gpu up -d sdxl_turbo_gpu
python tests/image-bench/run.py --model sdxl-turbo --backend sdxl-turbo --base http://127.0.0.1:8120 --height 512 --width 512 --steps 4
```

## Run (Z-Image-Turbo Diffusers on :8121)

```powershell
docker compose --profile gpu stop flux2_klein_gpu sdxl_turbo_gpu
docker compose --profile gpu up -d z_image_turbo_gpu
python tests/image-bench/run.py --model z-image-turbo --backend z-image-turbo --base http://127.0.0.1:8121 --height 512 --width 512 --steps 9
```

## Run (Z-Image-Turbo ComfyUI INT8/FP4 on :8188)

Ampere-safe Comfy-Org packs (`z_image_turbo_int8_convrot` + `qwen_3_4b_fp4_mixed` + `ae`).

```powershell
docker compose --profile gpu stop z_image_turbo_gpu flux2_klein_9b_kv_int8_gpu
docker compose --profile gpu up -d --build z_image_comfy_gpu
python tests/image-bench/run.py --model z-image-comfy --backend z-image-comfy --base http://127.0.0.1:8188 --height 512 --width 512 --steps 9 --timeout 600
```

Subset:

```powershell
python tests/image-bench/run.py --model flux2-klein-4b --base http://127.0.0.1:8116 --ids 01_studio_portrait,05_beach_sunset
```

Use a different `--model` folder name per checkpoint/backend so results don't overwrite each other.
