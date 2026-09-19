# FLUX.2-klein 9B-KV INT8 — HTTP API

Image generation API (text-to-image and optional image-conditioned edit).  
Default base URL shape: `http://<host>:<port>` (container listens on **8127**; reverse proxies / Vast map that to a public port).

No auth is built into the service. Put a gateway/API key in front if you expose it beyond a private network.

---

## Quick start

```bash
# Ready?
curl -sS "http://HOST:PORT/health"

# Text → image (JSON + base64)
curl -sS -X POST "http://HOST:PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red bicycle on a rainy street, cinematic","height":512,"width":512,"steps":4,"seed":42}'

# Same request, raw PNG body
curl -sS -X POST "http://HOST:PORT/generate.png" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red bicycle on a rainy street, cinematic","seed":42}' \
  -o out.png
```

While the model is still loading, `/health` and `/generate*` return **503**.

---

## `GET /health`

Liveness + model ready check.

**200** example:

```json
{
  "status": "ok",
  "model": "albex123/flux2-klein-kv-qint8-offload",
  "device_mode": "te_dit_swap",
  "defaults": { "height": 512, "width": 512, "steps": 4 },
  "load_s": 87.2,
  "vram_allocated_mb": 1024.0,
  "vram_reserved_mb": 2048.0,
  "vram_max_allocated_mb": 11000.0
}
```

| Field | Meaning |
| --- | --- |
| `status` | `"ok"` when the pipeline is loaded |
| `defaults` | Server defaults used when request fields are omitted |
| `load_s` | Seconds spent loading weights at startup |
| `vram_*_mb` | Current / peak CUDA memory (may be absent if no CUDA stats) |

**503** — model not ready yet (`detail`: `"model not ready"` / `"not ready"`).

---

## `POST /generate`

JSON request → JSON response with a base64-encoded image.

### Request body

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `prompt` | string | **yes** | — | Non-empty after trim |
| `height` | int | no | server default (usually 512) | 256–1024 |
| `width` | int | no | server default (usually 512) | 256–1024 |
| `steps` | int | no | server default (usually 4) | 1–8 (distilled; 4 is the usual setting) |
| `seed` | int \| null | no | random | Reproducible when set |
| `format` | string | no | `"png"` | `"png"`, `"jpeg"`, or `"jpg"` |
| `image_base64` | string \| string[] \| null | no | null | Optional reference image for conditioned generation. Raw base64 **or** `data:image/...;base64,...`. If an array is sent, **only the first** image is used. |
| `guidance_scale` | float | no | server default | Accepted for API parity; this pipeline does not meaningfully use CFG |

Recommended working size on 12 GB cards: **512×512**, **4 steps**.

### Success response (200)

```json
{
  "image_base64": "<base64 bytes>",
  "mime": "image/png",
  "prompt": "a red bicycle on a rainy street, cinematic",
  "height": 512,
  "width": 512,
  "steps": 4,
  "guidance_scale": 1.0,
  "seed": 42,
  "used_reference_image": false,
  "latency_s": 4.71,
  "images_per_sec": 0.212,
  "vram_allocated_mb": 980.0,
  "vram_reserved_mb": 11000.0,
  "vram_max_allocated_mb": 10500.0
}
```

Decode `image_base64` with a standard base64 decoder; content type is `mime`.

### Errors

| Code | When |
| --- | --- |
| **400** | Empty prompt, bad `format`, or invalid `image_base64` |
| **503** | Model still loading / not ready |
| **500** | Generation failure (message in `detail`) |

FastAPI error shape: `{ "detail": "..." }`.

---

## `POST /generate.png`

Same JSON body as `/generate`, but the response body is **raw PNG bytes** (`Content-Type: image/png`).

Useful response headers:

| Header | Meaning |
| --- | --- |
| `X-Latency-Seconds` | Generation latency |
| `X-Images-Per-Sec` | `1 / latency` when available |
| `X-VRAM-Max-Allocated-MB` | Peak allocated VRAM for the call (when reported) |

---

## Python example

```python
import base64
import json
import urllib.request

BASE = "http://HOST:PORT"

def health():
    with urllib.request.urlopen(f"{BASE}/health") as r:
        return json.load(r)

def generate(prompt: str, **kwargs) -> bytes:
    body = json.dumps({"prompt": prompt, **kwargs}).encode()
    req = urllib.request.Request(
        f"{BASE}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return base64.b64decode(out["image_base64"])

# T2I
open("out.png", "wb").write(generate("a red bicycle on a rainy street", height=512, width=512, steps=4, seed=42))

# With reference image
ref = base64.b64encode(open("ref.png", "rb").read()).decode()
open("edit.png", "wb").write(generate("same scene at night", image_base64=ref, seed=1))
```

---

## Integration notes

1. **Poll `/health`** until `status == "ok"` before sending traffic (cold start can take many minutes on first boot).
2. **Timeouts**: allow at least **60–120 s** per generate on mid-range GPUs; 512² / 4 steps is often ~4–8 s once warm.
3. **Concurrency**: server serializes `/generate` with a process-wide lock (TE↔DiT swap). Overlapping calls queue; do not run multiple uvicorn workers.
4. **License**: weights are based on FLUX.2-klein (non-commercial terms from Black Forest Labs apply to model use). Confirm your product’s license before commercial use.
