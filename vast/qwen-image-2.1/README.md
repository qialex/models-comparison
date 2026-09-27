# Vast.ai template — Qwen-Image-2.1

Same as local **`:8192`** edit API + Comfy (`--novram` for 12 GB). No worker/S3 — API + port only.

| | |
| --- | --- |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| API | **8192** — `GET /health`, `POST /edit.png`, `POST /generate.png` |
| Weights | ~**17 GB** int8 DiT + TE + VAE (first boot) |
| Filter | `gpu_ram>=12 cpu_ram>=32 disk_space>=20 num_gpus=1 static_ip=True reliability>0.90` |

**Disk:** 20 GB is tight. Weights download straight into Comfy (no HF cache dup). Prefer ≥25 GB if first boot fails.

## Rebuild

```powershell
python scripts/build_vast_qwen21_template.py
```

## Create on Vast

```powershell
$on = (Get-Content -Raw vast\qwen-image-2.1\onstart.gz.b64).Trim()
vastai create template --name qwen-image-2.1 `
  --image pytorch/pytorch --image_tag 2.6.0-cuda12.4-cudnn9-runtime `
  --env "-p 8192:8192 -e PORT=8192 -e COMFY_PORT=8188 -e COMFY_URL=http://127.0.0.1:8188 -e DEFAULT_STEPS=25 -e DEFAULT_SIZE=1024 -e COMFY_ROOT=/workspace/ComfyUI" `
  --onstart-cmd ("#!/bin/bash`npython -c `"import base64,gzip,os;p='/tmp/_vast_onstart.sh';open(p,'wb').write(gzip.decompress(base64.b64decode('$on')));os.chmod(p,0o755);os.execv('/bin/bash',['bash',p])`"") `
  --desc "Qwen-Image-2.1 Comfy int8 API :8192. >=12GB VRAM >=32GB RAM >=20GB disk." `
  --ssh --direct --disk_space 20 `
  --search_params "gpu_ram>=12 cpu_ram>=32 disk_space>=20 num_gpus=1 rented=False static_ip=True reliability>0.90"
```

Or paste `onstart.sh` / `template.json` in the Vast UI.

## Client

```powershell
curl.exe -s http://<IP>:<mapped_8192>/health
curl.exe -s -o out.png http://<IP>:<mapped_8192>/edit.png `
  -F "prompt=Keep the character and pose in <image1> unchanged. Change the background to a seaside beach at golden hour." `
  -F "image=@ref.png" -F "steps=25" -F "resolution=1024"
```

First boot: torch 2.7 + Comfy + ~17 GB download (often **20–45 min**).
