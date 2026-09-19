# Vast.ai template — Qwen3.5-4B MTP + ngram-mod

Same recipe as local **`:8112`** (`qwen35_4b_ngram_gpu`): Unsloth MTP GGUF + `draft-mtp,ngram-mod`.

| | |
| --- | --- |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| API | `http://<instance-ip>:<mapped-8080>/v1` |
| Weights | ~2.7 GB (downloads on first start) |
| Disk | **≥16 GB** recommended |

## Console (fastest)

1. [Create template](https://cloud.vast.ai/templates/) → **New Template**
2. **Image:** `ghcr.io/ggml-org/llama.cpp:server-cuda`
3. **Launch mode:** SSH (or Jupyter + SSH)
4. **Ports / Docker options:**
   ```text
   -p 8080:8080 -e N_CTX=8192 -e N_PARALLEL=1 -e N_PREDICT=1536 -e N_GPU_LAYERS=99
   ```
5. **On-start script:** paste contents of [`onstart.sh`](onstart.sh)
6. Save → rent a GPU (≥ **6 GB VRAM**; 8–12 GB more comfortable at ctx 8192)

For **RTX A2000 6 GB** / short completions, use:

```text
-p 8080:8080 -e N_CTX=1024 -e N_PARALLEL=4 -e N_PREDICT=8 -e N_GPU_LAYERS=99
```

## CLI

```bash
vastai create template \
  --name "qwen3.5-4b-mtp-ngram" \
  --image "ghcr.io/ggml-org/llama.cpp" \
  --image_tag "server-cuda" \
  --ssh --direct \
  --disk_space 16 \
  --env "-p 8080:8080 -e N_CTX=8192 -e N_PARALLEL=1 -e N_PREDICT=1536 -e N_GPU_LAYERS=99" \
  --onstart-cmd "$(cat onstart.sh)" \
  --desc "Qwen3.5-4B MTP + ngram-mod (local :8112)"
```

Or from `template.json` (API):

```bash
curl -s https://console.vast.ai/api/v0/template/ \
  -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  -d @template.json
```

Docs: [Creating templates with API](https://docs.vast.ai/api-reference/creating-and-using-templates-with-api), [Template settings](https://docs.vast.ai/guides/templates/template-settings).

## Smoke test

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi in 3 words."}],"max_tokens":16,"temperature":0.1,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Notes

- Vast **SSH/Jupyter** replaces the image entrypoint — `onstart` must start `llama-server` (this script does).
- First boot downloads the MTP GGUF; later boots reuse `/models` if the volume persists.
- Temperature is **per request** (`temperature` in the JSON body).
- Expect ~**65–90 tok/s** decode on a 3060-class card for unique prompts; ngram spikes only when output repeats.
