# Qwen-Image-Edit (ComfyUI)

[Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) DiT pack + TE/VAE from [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI).

| | |
| --- | --- |
| Port | **8190** |
| Mode | `--novram` (CPU offload; `--lowvram` OOMs on 12 GB) |
| Default DiT | `qwen_image_edit_2509_int8_convrot` (~20.5 GB) |
| Lightning | `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16` (required for 4-step / CFG 1) |
| TE | `qwen_2.5_vl_7b_fp8_scaled` (~9.4 GB) |
| VAE | `qwen_image_vae` (~0.25 GB) |
| Target | 12 GB VRAM + plenty of system RAM |

```powershell
# ~30 GB download
python scripts/download_qwen_image_edit_comfy.py
# optional: --dit 2511-int8 | 2509-fp8 | 2511-fp8mixed

docker compose --profile gpu stop ltx_video_gpu wan_t2v_gpu z_image_comfy_gpu flux2_klein_9b_comfy_gpu
docker compose --profile gpu up -d --build qwen_image_edit_comfy_gpu
curl.exe -s http://localhost:8190/system_stats
```

UI: http://127.0.0.1:8190 — load Comfy template **Qwen-Image-Edit** / **2509**.

Smoke API graph notes (aligned with official Lightning 2509):
- `TextEncodeQwenImageEditPlus` (not the older single-image encoder)
- `ImageScaleToTotalPixels` → ~1 MP before `VAEEncode` (avoids ref/sampler grid mismatch)
- Lightning LoRA + 4 steps / CFG 1 + `CFGNorm`
- Prompts should be **edit instructions** (“make lighting warmer…”), not full T2I scene essays
