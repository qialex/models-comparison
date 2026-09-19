# Vast.ai template — FLUX.2-klein 9B-KV INT8

Same recipe as local **`:8127`**: public
[albex123/flux2-klein-kv-qint8-offload](https://huggingface.co/albex123/flux2-klein-kv-qint8-offload)
weights + TE↔DiT swap. Onstart is **self-contained**: downloads that pack, surgically patches the stock pipeline for 512², writes a small FastAPI wrapper. No GitHub/HF hosting of serve code.

**Client API docs (shareable):** [`../../flux2-klein-9b-kv-int8/API.md`](../../flux2-klein-9b-kv-int8/API.md)

Onstart is a **self-extracting bash wrapper** (Vast does not auto-gunzip blobs; raw gzip+b64 was why disk stayed at 0).

| | |
| --- | --- |
| **Vast template ID** | **706179** |
| **hash_id** | `1ec7dac83a206e2101e0fa6ec06d2cdf` |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| API | mapped **8127** — `GET /health`, `POST /generate` |
| Weights | ~**19 GB** on first boot |
| Filter | `gpu_ram>=12 cpu_ram>=32 disk_space>=20 num_gpus=1 static_ip=True reliability>0.90` (+ client: `gpu_frac*gpu_ram>=11.5`, `direct_port_count>=2`) |

## Rebuild / push template

```powershell
python scripts/build_vast_flux_template.py
# then (one shot — partial updates wipe other fields):
$on = (Get-Content -Raw vast\flux2-klein-9b-kv-int8\onstart.gz.b64).Trim()
vastai update template <hash_id> --name flux2-klein-9b-kv-int8 --image pytorch/pytorch --image_tag 2.6.0-cuda12.4-cudnn9-runtime --env "-p 8127:8127 -e PORT=8127 -e HEIGHT=512 -e WIDTH=512 -e STEPS=4 -e MODEL_ID=albex123/flux2-klein-kv-qint8-offload -e HF_HOME=/workspace/hf-cache" --onstart-cmd $on --desc "..." --ssh --direct --disk_space 20 --search_params "gpu_ram>=12 cpu_ram>=32 disk_space>=20 num_gpus=1 rented=False static_ip=True reliability>0.90"
vastai update instance <id> --template_hash_id <new_hash>
vastai recycle instance <id>
```

First boot: pip + weight download + load (often **15–40 min**).
