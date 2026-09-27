# Qwen-Image-2.1 (ComfyUI)

[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) pack — T2I + edit. License: [Qwen Research](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE).

| | |
| --- | --- |
| Comfy UI | **8191** |
| API (host) | **8192** — `api.py` (`/generate.png`, `/edit.png`) |
| Mode | `--novram` |

```powershell
python scripts/download_qwen_image_21_comfy.py
docker compose --profile gpu up -d --build qwen_image_21_comfy_gpu

# API (separate process; talks to Comfy)
pip install fastapi uvicorn httpx python-multipart
uvicorn qwen-image-2.1-comfy.api:app --app-dir . --host 127.0.0.1 --port 8192
# or from this folder:
#   cd qwen-image-2.1-comfy; uvicorn api:app --host 127.0.0.1 --port 8192

curl.exe -s -o out.png http://127.0.0.1:8192/generate.png `
  -F "prompt=a woman in a cozy cafe at night" `
  -F "width=352" -F "height=512" -F "steps=25"

curl.exe -s -o out.png http://127.0.0.1:8192/edit.png `
  -F "prompt=put this character in a cozy cafe at night" `
  -F "image=@cache/qwen-image-2.1-comfy/output/qwen21_rooftop_40step_00001_.png" `
  -F "steps=40" -F "resolution=512"
```
