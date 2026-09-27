#!/usr/bin/env python3
"""Build Vast onstart + template.json for Qwen-Image-2.1 Comfy edit API.

Target: >=12GB VRAM, >=32GB RAM, >=20GB disk. Port 8192.
Vast CLI prefers onstart under ~4048 chars — gzip+b64 self-extracting wrapper.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vast" / "qwen-image-2.1"
OUT.mkdir(parents=True, exist_ok=True)

# Compact API — written via heredoc on boot (compresses well in outer gzip).
APP_PY = """\
from __future__ import annotations
import json,os,time,uuid
from pathlib import Path
import httpx
from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from fastapi.responses import Response
U=os.environ.get("COMFY_URL","http://127.0.0.1:8188").rstrip("/")
IN=Path(os.environ.get("COMFY_INPUT_DIR","/workspace/ComfyUI/input"))
DIT=os.environ.get("DIT_NAME","qwen_image_2.1_int8_convrot.safetensors")
TE=os.environ.get("TE_NAME","qwen3vl_8b_int8_convrot.safetensors")
VAE=os.environ.get("VAE_NAME","qwen_image_2.1_vae_bf16.safetensors")
DS=int(os.environ.get("DEFAULT_STEPS","25")); DR=int(os.environ.get("DEFAULT_SIZE","1024"))
app=FastAPI(title="qwen-image-2.1")
def ready():
 try: return httpx.get(f"{U}/system_stats",timeout=5).status_code==200
 except Exception: return False
def wait(wf,t):
 r=httpx.post(f"{U}/prompt",json={"prompt":wf},timeout=60)
 if r.status_code>=400: raise HTTPException(r.status_code,r.text[:2000])
 pid=r.json().get("prompt_id")
 if not pid: raise HTTPException(500,f"queue failed: {r.text[:1000]}")
 end=time.time()+t
 while time.time()<end:
  h=httpx.get(f"{U}/history/{pid}",timeout=30).json()
  if pid in h:
   e=h[pid]; st=e.get("status") or {}
   if st.get("status_str")=="error": raise HTTPException(500,json.dumps(st)[:2000])
   for m in st.get("messages") or []:
    if m and m[0]=="execution_error": raise HTTPException(500,json.dumps(m[1])[:2000])
   if st.get("completed") or e.get("outputs"):
    for n in (e.get("outputs") or {}).values():
     for img in n.get("images") or []:
      vr=httpx.get(f"{U}/view",params={"filename":img["filename"],"subfolder":img.get("subfolder") or "","type":img.get("type") or "output"},timeout=120)
      vr.raise_for_status(); return vr.content
    raise HTTPException(500,"no image")
  time.sleep(1)
 raise HTTPException(504,"timeout")
def loaders():
 return {"1":{"class_type":"UNETLoader","inputs":{"unet_name":DIT,"weight_dtype":"default"}},"11":{"class_type":"QwenImage21Cache","inputs":{"model":["1",0],"device":"auto","dtype":"default"}},"2":{"class_type":"CLIPLoader","inputs":{"clip_name":TE,"type":"qwen_image","device":"default"}},"3":{"class_type":"VAELoader","inputs":{"vae_name":VAE}}}
def edit(p,name,steps,seed,res):
 w=loaders(); w["10"]={"class_type":"LoadImage","inputs":{"image":name}}
 w["4"]={"class_type":"TextEncodeQwenImage21","inputs":{"clip":["2",0],"prompt":p,"negative_prompt":"","resolution":res,"images.image_1":["10",0],"vae":["3",0]}}
 w["6"]={"class_type":"KSampler","inputs":{"model":["11",0],"positive":["4",0],"negative":["4",1],"latent_image":["4",2],"seed":seed,"steps":steps,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}}
 w["7"]={"class_type":"VAEDecode","inputs":{"samples":["6",0],"vae":["3",0]}}
 w["8"]={"class_type":"SaveImage","inputs":{"images":["7",0],"filename_prefix":"api_edit"}}; return w
def t2i(p,w,h,steps,seed):
 wf=loaders(); r=max(w,h)
 wf["4"]={"class_type":"TextEncodeQwenImage21","inputs":{"clip":["2",0],"prompt":p,"negative_prompt":"","resolution":r}}
 wf["5"]={"class_type":"EmptyLatentImage","inputs":{"width":w,"height":h,"batch_size":1}}
 wf["6"]={"class_type":"KSampler","inputs":{"model":["11",0],"positive":["4",0],"negative":["4",1],"latent_image":["5",0],"seed":seed,"steps":steps,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}}
 wf["7"]={"class_type":"VAEDecode","inputs":{"samples":["6",0],"vae":["3",0]}}
 wf["8"]={"class_type":"SaveImage","inputs":{"images":["7",0],"filename_prefix":"api_t2i"}}; return wf
@app.get("/health")
def health():
 if not ready(): raise HTTPException(503,"comfy not ready")
 return {"status":"ok","model":"Comfy-Org/Qwen-Image-2.1","defaults":{"steps":DS,"resolution":DR}}
@app.post("/edit.png")
async def edit_png(prompt:str=Form(...),image:UploadFile=File(...),steps:int=Form(DS),resolution:int=Form(DR),seed:int|None=Form(None),timeout_s:float=Form(1800)):
 if not ready(): raise HTTPException(503,"comfy not ready")
 data=await image.read()
 if not data: raise HTTPException(400,"empty image")
 IN.mkdir(parents=True,exist_ok=True)
 suf=Path(image.filename or "in.png").suffix.lower() or ".png"
 if suf not in (".png",".jpg",".jpeg",".webp"): suf=".png"
 name=f"api_{uuid.uuid4().hex}{suf}"; (IN/name).write_bytes(data)
 p=prompt.strip()
 if "<image1>" not in p.lower():
  p=f"Keep the character and pose in <image1> unchanged. {p}"
 sid=seed if seed is not None else int(uuid.uuid4().int%(2**31-1))
 return Response(content=wait(edit(p,name,max(1,min(80,int(steps))),sid,max(256,min(2048,int(resolution)))),float(timeout_s)),media_type="image/png")
@app.post("/generate.png")
async def gen_png(prompt:str=Form(...),width:int=Form(DR),height:int=Form(DR),steps:int=Form(DS),seed:int|None=Form(None),timeout_s:float=Form(1800)):
 if not ready(): raise HTTPException(503,"comfy not ready")
 p=prompt.strip()
 if not p: raise HTTPException(400,"empty prompt")
 sid=seed if seed is not None else int(uuid.uuid4().int%(2**31-1))
 return Response(content=wait(t2i(p,max(256,min(2048,int(width))),max(256,min(2048,int(height))),max(1,min(80,int(steps))),sid),float(timeout_s)),media_type="image/png")
"""

HEAD = r'''#!/bin/bash
set -eu
env >> /etc/environment 2>/dev/null || true
export DEBIAN_FRONTEND=noninteractive
export PORT="${PORT:-8192}" COMFY_PORT="${COMFY_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:${COMFY_PORT}}"
export COMFY_ROOT="${COMFY_ROOT:-/workspace/ComfyUI}"
export COMFY_INPUT_DIR="${COMFY_INPUT_DIR:-$COMFY_ROOT/input}"
export DEFAULT_STEPS="${DEFAULT_STEPS:-25}" DEFAULT_SIZE="${DEFAULT_SIZE:-1024}"
export REPO="${REPO:-Comfy-Org/Qwen-Image-2.1}"
mkdir -p /workspace
apt-get update -qq
apt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential
rm -rf /var/lib/apt/lists/*
python -m pip install -q --no-cache-dir -U pip
python -m pip install -q --no-cache-dir "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -q --no-cache-dir fastapi "uvicorn[standard]" httpx python-multipart
# mkdir input BEFORE clone creates an empty ComfyUI/ and breaks git clone
if [ ! -d "$COMFY_ROOT/.git" ]; then
  rm -rf "$COMFY_ROOT"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"
fi
mkdir -p "$COMFY_INPUT_DIR"
python -m pip install -q --no-cache-dir -r "$COMFY_ROOT/requirements.txt"
M="$COMFY_ROOT/models"; mkdir -p "$M/diffusion_models" "$M/text_encoders" "$M/vae"
AUTH=(); [ -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}")
dl(){ local r="$1" d="$2"; [ -f "$d" ] && [ "$(stat -c%s "$d" 2>/dev/null || echo 0)" -gt 1000000 ] && { echo "skip $r"; return; }
 curl -fL --retry 5 --retry-delay 2 "${AUTH[@]}" -o "$d.part" "https://huggingface.co/${REPO}/resolve/main/$r" && mv -f "$d.part" "$d"; }
dl "diffusion_models/qwen_image_2.1_int8_convrot.safetensors" "$M/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
dl "text_encoders/qwen3vl_8b_int8_convrot.safetensors" "$M/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
dl "vae/qwen_image_2.1_vae_bf16.safetensors" "$M/vae/qwen_image_2.1_vae_bf16.safetensors"
cat > /workspace/app.py <<'APPEOF'
'''

TAIL = r'''
APPEOF
cd "$COMFY_ROOT"
python main.py --listen 127.0.0.1 --port "$COMFY_PORT" --disable-auto-launch --novram &
COMFY_PID=$!
for i in $(seq 1 180); do
 curl -sf "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null && break
 kill -0 "$COMFY_PID" 2>/dev/null || { wait "$COMFY_PID" || true; exit 1; }
 sleep 2
done
trap 'kill $COMFY_PID 2>/dev/null || true' EXIT
cd /workspace
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
'''

if "APPEOF" in APP_PY:
    raise SystemExit("APP_PY must not contain APPEOF")

REAL_SCRIPT = HEAD + APP_PY + TAIL
(OUT / "onstart.real.sh").write_text(REAL_SCRIPT, encoding="utf-8", newline="\n")

blob = base64.b64encode(gzip.compress(REAL_SCRIPT.encode("utf-8"), compresslevel=9)).decode("ascii")
WRAPPER = (
    "#!/bin/bash\n"
    "python -c \"import base64,gzip,os;"
    f"p='/tmp/_vast_onstart.sh';open(p,'wb').write(gzip.decompress(base64.b64decode('{blob}')));"
    "os.chmod(p,0o755);os.execv('/bin/bash',['bash',p])\"\n"
)
(OUT / "onstart.sh").write_text(WRAPPER, encoding="utf-8", newline="\n")
(OUT / "onstart.gz.b64").write_text(blob + "\n", encoding="utf-8")

TEMPLATE = {
    "name": "qwen-image-2.1",
    "desc": (
        "Qwen-Image-2.1 Comfy int8 (--novram). FastAPI :8192 "
        "GET /health POST /edit.png POST /generate.png. "
        ">=12GB VRAM, >=32GB RAM, >=20GB disk. Weights ~17GB."
    ),
    "image": "pytorch/pytorch",
    "tag": "2.6.0-cuda12.4-cudnn9-runtime",
    "env": (
        "-p 8192:8192 -e PORT=8192 -e COMFY_PORT=8188 "
        "-e COMFY_URL=http://127.0.0.1:8188 -e DEFAULT_STEPS=25 "
        "-e DEFAULT_SIZE=1024 -e COMFY_ROOT=/workspace/ComfyUI"
    ),
    "runtype": "ssh",
    "ssh_direct": True,
    "use_ssh": True,
    "recommended_disk_space": 20,
    "private": True,
    "search_params": (
        "gpu_ram>=12 cpu_ram>=32 disk_space>=20 num_gpus=1 "
        "rented=False static_ip=True reliability>0.90"
    ),
    "onstart": WRAPPER,
}

(OUT / "template.json").write_text(json.dumps(TEMPLATE, indent=2) + "\n", encoding="utf-8")

(OUT / "README.md").write_text(
    """# Vast.ai template — Qwen-Image-2.1

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
$on = (Get-Content -Raw vast\\qwen-image-2.1\\onstart.gz.b64).Trim()
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
""",
    encoding="utf-8",
    newline="\n",
)

print("wrote", OUT / "onstart.real.sh", "chars", len(REAL_SCRIPT))
print("wrote", OUT / "onstart.sh", "chars", len(WRAPPER))
print("wrote", OUT / "template.json")
print("wrote", OUT / "README.md")
print("wrapper size", len(WRAPPER), "(target <=4048)")
if len(WRAPPER) > 4048:
    print("WARNING: wrapper over ~4048 CLI limit — use onstart.gz.b64 + README PowerShell create")
