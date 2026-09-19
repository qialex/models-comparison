# Vast.ai template — Kokoro-82M TTS

Same API as local **`:8131`** (`kokoro_82m_gpu`): `hexgrad/Kokoro-82M` + FastAPI.

| | |
| --- | --- |
| Image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| API | `http://<ip>:<mapped-8080>` |
| Endpoints | `GET /health` · `GET /voices` · `POST /tts` |
| VRAM | **≥4 GB** (serial / 1–2 concurrent; avoid heavy parallel) |
| Disk | **8 GB** (minimal; weights ~300 MB + voices) |
| Search | `gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1` |
| Template ID | **726173** |
| Hash | `e2d0e011d6945f72bdea985c0c2e9778` |

Rent example:

```bash
vastai create instance <offer_id> --template_hash e2d0e011d6945f72bdea985c0c2e9778 --disk 8 --cancel-unavail
```

## Console

1. [Create template](https://cloud.vast.ai/templates/) → **New Template**
2. **Image:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
3. **Launch mode:** SSH (direct)
4. **Disk:** 8 GB
5. **Docker options:**
   ```text
   -p 8080:8080 -e PORT=8080 -e MODEL_ID=hexgrad/Kokoro-82M -e LANG_CODE=a -e DEVICE=cuda -e DEFAULT_VOICE=af_heart -e HF_HOME=/workspace/hf-cache
   ```
6. **On-start:** paste [`onstart.sh`](onstart.sh) (or rebuild via `python scripts/build_vast_kokoro_template.py`)
7. **Search filters:** `gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 rented=False`

## CLI

```bash
python scripts/build_vast_kokoro_template.py

vastai create template \
  --name "kokoro-82m" \
  --image "pytorch/pytorch" \
  --image_tag "2.6.0-cuda12.4-cudnn9-runtime" \
  --ssh --direct \
  --disk_space 8 \
  --search_params "gpu_ram>=4 cpu_ram>=4 disk_space>=8 num_gpus=1 rented=False" \
  --env "-p 8080:8080 -e PORT=8080 -e MODEL_ID=hexgrad/Kokoro-82M -e LANG_CODE=a -e DEVICE=cuda -e DEFAULT_VOICE=af_heart -e HF_HOME=/workspace/hf-cache" \
  --onstart-cmd "$(cat vast/kokoro-82m/onstart.sh)" \
  --desc "Kokoro-82M TTS FastAPI :8080 (>=4GB VRAM, 8GB disk)"
```

## Smoke

```bash
curl -s http://127.0.0.1:<port>/health
curl -s http://127.0.0.1:<port>/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from Kokoro.","voice":"af_heart","return_audio":false}'
```

## Notes

- First boot: `pip install` + HF weight/voice download (a few minutes).
- Prefer **frac such that effective VRAM ≥ 4 GB** (e.g. 8 GB @ 0.5 is OK).
- If the host must pull the full PyTorch image into the 8 GB volume and fails, bump disk to **16 GB**.
