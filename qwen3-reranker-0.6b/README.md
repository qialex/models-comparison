# Qwen3-Reranker-0.6B (`:8129`)

Text reranker (CrossEncoder). Multilingual; instruction-aware.

```bash
docker compose --profile gpu up -d --build qwen3_reranker_0_6b_gpu
curl.exe -s http://localhost:8129/health
```

```bash
curl.exe -s http://localhost:8129/score -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of China?\",\"documents\":[\"The capital of China is Beijing.\",\"Gravity attracts bodies.\"],\"sigmoid\":true}"
```

Scene suite (one-by-one):

```bash
$env:RERANK_BASE="http://127.0.0.1:8129"; python scripts/scene_evolution_rerank.py
```
