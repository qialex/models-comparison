#!/usr/bin/env python3
"""Build Vast template onstart.

Vast writes onstart to /root/onstart.sh and `bash`es it — it does NOT auto-gunzip
base64 blobs. Keep a tiny self-extracting wrapper (plaintext) that decompresses
the real script, so the stored onstart stays under the ~4048 CLI-friendly size.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAST = ROOT / "vast" / "flux2-klein-9b-kv-int8"

# Real boot script (runs after decompress on the instance).
REAL = r'''#!/bin/bash
set -eu
env >> /etc/environment 2>/dev/null || true
PORT="${PORT:-8127}"; APP_DIR="${APP_DIR:-/workspace/app}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}" MODEL_ID="${MODEL_ID:-albex123/flux2-klein-kv-qint8-offload}"
export HEIGHT="${HEIGHT:-512}" WIDTH="${WIDTH:-512}" STEPS="${STEPS:-4}"
mkdir -p "$HF_HOME" "$APP_DIR"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
python -m pip install --no-cache-dir -U pip accelerate transformers safetensors sentencepiece protobuf Pillow fastapi "uvicorn[standard]" "pydantic>=2" huggingface_hub optimum-quanto
python -m pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git"
echo "Downloading $MODEL_ID ..."
MODEL_DIR="$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ["MODEL_ID"], token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None))
PY
)"
export MODEL_DIR APP_DIR
cp -f "$MODEL_DIR/pipeline_flux2_klein_kv_offload.py" "$APP_DIR/"
python - <<'PY'
from pathlib import Path
import os
p=Path(os.environ["APP_DIR"])/"pipeline_flux2_klein_kv_offload.py"; t=p.read_text(encoding="utf-8")
a="device = self._default_device  # \u2190 \u043f\u0440\u043e\u0441\u0442\u043e \u0431\u0435\u0440\u0451\u043c, \u043d\u0435 \u0441\u0435\u0442\u0438\u043c _execution_device\n        \n        # \u0412\u0430\u0436\u043d\u043e: \u043e\u043a\u0440\u0443\u0433\u043b\u044f\u0435\u043c \u0434\u043e \u043a\u0440\u0430\u0442\u043d\u044b\u0445 16\n        height = 2 * (int(height) // self.vae_scale_factor)\n        width = 2 * (int(width) // self.vae_scale_factor)"
b="device = self._default_device\n        if image is not None:\n            pix_h=(int(image.height)//self.vae_scale_factor)*self.vae_scale_factor; pix_w=(int(image.width)//self.vae_scale_factor)*self.vae_scale_factor\n        else:\n            pix_h=(int(height)//self.vae_scale_factor)*self.vae_scale_factor; pix_w=(int(width)//self.vae_scale_factor)*self.vae_scale_factor"
if a not in t: raise SystemExit("patch1 missing")
t=t.replace(a,b,1)
t=t.replace("target_w = (image.width // 16) * 16\n            target_h = (image.height // 16) * 16\n            img_t = self.image_processor.preprocess(image, height=target_h, width=target_w, resize_mode=\"crop\")","img_t = self.image_processor.preprocess(image, height=pix_h, width=pix_w, resize_mode=\"crop\")",1)
t=t.replace("torch.zeros((1, 3, height, width),","torch.zeros((1, 3, pix_h, pix_w),",1)
t=t.replace("il_packed = _pack_latents(il).squeeze(0)  # (256, C_patch)\n        C_patch = il_packed.shape[1]\n\n        # Noise latents\n        latents = torch.randn((1, C_patch, height//2, width//2),","_, C_patch, pack_h, pack_w = il.shape\n        il_packed = _pack_latents(il).squeeze(0)\n        latents = torch.randn((1, C_patch, pack_h, pack_w),",1)
t=t.replace("torch.arange(height//2, device=device), torch.arange(width//2, device=device)","torch.arange(pack_h, device=device), torch.arange(pack_w, device=device)",1)
t=t.replace("_unpack_latents_with_ids(lat_p, lat_ids, height//2, width//2)","_unpack_latents_with_ids(lat_p, lat_ids, pack_h, pack_w)",1)
p.write_text(t, encoding="utf-8"); print("patched")
Path(os.environ["APP_DIR"],"app.py").write_text("""from __future__ import annotations
import base64,io,os,time,threading
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI,HTTPException
from fastapi.responses import Response
from pydantic import BaseModel,Field
MID=os.environ.get("MODEL_ID","albex123/flux2-klein-kv-qint8-offload"); MD=os.environ.get("MODEL_DIR")
H=int(os.environ.get("HEIGHT","512")); W=int(os.environ.get("WIDTH","512")); S=int(os.environ.get("STEPS","4"))
pipe=None; ready=False; load_s=None; _lock=threading.Lock()
def load():
 global load_s
 from pipeline_flux2_klein_kv_offload import Flux2KleinKVOffloadPipeline
 from huggingface_hub import snapshot_download
 t0=time.perf_counter(); d=MD or snapshot_download(MID,token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)
 p=Flux2KleinKVOffloadPipeline.from_quanto(d,device="cuda"); load_s=round(time.perf_counter()-t0,2); return p
@asynccontextmanager
async def life(_):
 global pipe,ready
 if not torch.cuda.is_available(): raise RuntimeError("no cuda")
 pipe=load(); ready=True; yield; ready=False
app=FastAPI(lifespan=life)
class Req(BaseModel):
 prompt:str=Field(...,min_length=1); image_base64:str|list[str]|None=None
 height:int|None=None; width:int|None=None; steps:int|None=None; seed:int|None=None; format:str="png"
def dec(raw):
 if raw is None: return None
 from PIL import Image
 xs=raw if isinstance(raw,list) else [raw]; o=[]
 for x in xs:
  s=x.strip()
  if "," in s and s.lower().startswith("data:"): s=s.split(",",1)[1]
  o.append(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))
 return o[0] if len(o)==1 else o
def vr():
 if not torch.cuda.is_available(): return {}
 return {"vram_allocated_mb":round(torch.cuda.memory_allocated()/1048576,1),"vram_reserved_mb":round(torch.cuda.memory_reserved()/1048576,1),"vram_max_allocated_mb":round(torch.cuda.max_memory_allocated()/1048576,1)}
@app.get("/health")
def health():
 if not ready or pipe is None: raise HTTPException(503,"not ready")
 return {"status":"ok","model":MID,"device_mode":"te_dit_swap","defaults":{"height":H,"width":W,"steps":S},"load_s":load_s,**vr()}
@app.post("/generate")
def generate(b:Req):
 if not ready or pipe is None: raise HTTPException(503,"not ready")
 prompt=b.prompt.strip()
 if not prompt: raise HTTPException(400,"empty")
 h,w,st=b.height or H,b.width or W,b.steps or S; fmt=b.format.lower().strip()
 if fmt=="jpg": fmt="jpeg"
 if fmt not in ("png","jpeg"): raise HTTPException(400,"fmt")
 if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
 gen=torch.Generator(device="cuda").manual_seed(int(b.seed)) if b.seed is not None else None
 try:
  ref=dec(b.image_base64); ref=ref[0] if isinstance(ref,list) else ref
 except Exception as e: raise HTTPException(400,str(e)) from e
 t0=time.perf_counter()
 try:
  kw={"prompt":prompt,"height":h,"width":w,"num_inference_steps":st,"generator":gen}
  if ref is not None: kw["image"]=ref
  with _lock:
   img=pipe(**kw); img=img.images[0] if hasattr(img,"images") else img
 except Exception as e: raise HTTPException(500,f"{type(e).__name__}: {e}") from e
 buf=io.BytesIO(); img.save(buf,format=fmt.upper()); raw=buf.getvalue(); dt=time.perf_counter()-t0
 return {"image_base64":base64.b64encode(raw).decode(),"mime":f"image/{fmt}","prompt":prompt,"height":h,"width":w,"steps":st,"seed":b.seed,"used_reference_image":ref is not None,"latency_s":round(dt,3),"images_per_sec":round(1/dt,3) if dt>0 else None,**vr()}
@app.post("/generate.png")
def gp(b:Req):
 o=generate(b)
 return Response(base64.b64decode(o["image_base64"]),media_type="image/png",headers={"X-Latency-Seconds":str(o["latency_s"])})
""", encoding="utf-8")
print("app ok")
PY
cd "$APP_DIR"; exec python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
'''


def wrap(real: str) -> str:
    blob = base64.b64encode(gzip.compress(real.encode("utf-8"), compresslevel=9)).decode("ascii")
    # Plaintext bash Vast can execute; python is on the pytorch image.
    return (
        "#!/bin/bash\n"
        "python -c \"import base64,gzip,os;p='/tmp/_vast_onstart.sh';"
        f"open(p,'wb').write(gzip.decompress(base64.b64decode('{blob}')));"
        "os.chmod(p,0o755);os.execv('/bin/bash',['bash',p])\"\n"
    )


def main() -> int:
    real = REAL.replace("\r\n", "\n").lstrip("\n")
    onstart = wrap(real)
    VAST.mkdir(parents=True, exist_ok=True)
    (VAST / "onstart.real.sh").write_text(real, encoding="utf-8", newline="\n")
    (VAST / "onstart.sh").write_text(onstart, encoding="utf-8", newline="\n")
    print(f"real={len(real)} wrapped_onstart={len(onstart)}")
    if len(onstart) > 4048:
        raise SystemExit(f"wrapped onstart too large for CLI: {len(onstart)} > 4048")

    tpl = {
        "id": 706179,
        "name": "flux2-klein-9b-kv-int8",
        "desc": "FLUX.2-klein 9B-KV INT8 TE/DiT swap (:8127). Single-flight generate lock. Self-contained surgical 512 patch. ~19GB, >=12GB VRAM, >=32GB RAM, >=20GB disk.",
        "image": "pytorch/pytorch",
        "tag": "2.6.0-cuda12.4-cudnn9-runtime",
        "env": "-p 8127:8127 -e PORT=8127 -e HEIGHT=512 -e WIDTH=512 -e STEPS=4 -e MODEL_ID=albex123/flux2-klein-kv-qint8-offload -e HF_HOME=/workspace/hf-cache",
        "runtype": "ssh",
        "ssh_direct": True,
        "use_ssh": True,
        "recommended_disk_space": 20,
        "private": True,
        # MUST be executable bash (self-extracting). Do not store raw gzip alone.
        "onstart": onstart,
        "search_params": "gpu_ram>=12 cpu_ram>=32 disk_space>=20 num_gpus=1 rented=False static_ip=True reliability>0.90",
    }
    (VAST / "template.json").write_text(json.dumps(tpl, indent=2) + "\n", encoding="utf-8")
    print("ok", VAST / "template.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
