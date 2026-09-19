# Qwen3-VL-Reranker-2B (`:8128`)

Multimodal reranker — scores (query, document) pairs where query/docs can be text and/or images.

```powershell
docker compose --profile gpu stop flux2_klein_9b_kv_int8_gpu   # free the GPU
docker compose --profile gpu up -d --build qwen3_vl_reranker_2b_gpu
curl.exe -s http://localhost:8128/health
```

## API

- `GET /health`
- `POST /score` — `{ "query", "documents": [...], "prompt?", "sigmoid?" }` → `scores`
- `POST /rank` — same + optional `top_k` → `rankings` (`corpus_id`, `score`)

Documents may be a string (text or image URL), or `{ "text"?, "image"? }`.

```powershell
curl.exe -s http://localhost:8128/score -H "Content-Type: application/json" -d "{\"query\":\"A woman playing with her dog on a beach at sunset.\",\"documents\":[\"A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.\",\"A cat sleeping on a windowsill.\"],\"sigmoid\":true}"
```
