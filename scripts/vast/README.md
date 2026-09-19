# Vast.ai hunt / rent / search scripts

Tracked copies of the helpers that lived under ignored `cache/*/\_debug/`.
Auth stays on the machine (`vastai` CLI + `~/.config/vastai/vast_api_key`) — **nothing here embeds API keys**.

## Layout

| Path | Purpose |
| --- | --- |
| `kokoro/hunt_miracle.py` | Auto-rent Kokoro: static IP, reli>0.9, score/price bands |
| `kokoro/search100_collect.py` | Market snapshots (no rent) |
| `kokoro/rank_score.py` | Rank collected offers by BW×frac / ¢ |
| `flux_int8/hunt_*.py`, `rent_*.py` | Flux INT8 template hunts |
| `flux_int8/search10_*.py`, `montecarlo_vast.py` | Search-only market analysis |
| `qwen_mtp/*` | Qwen3.5-4B MTP ngram offer pick / rent |
| `_paths.py` | Repo root + `cache/` output dirs; redacts `instance_api_key` |

## Outputs

Runtime logs/JSON go under **`cache/<model>/_debug/`** (gitignored), not next to these scripts.

## Run (from repo root)

```powershell
python scripts/vast/kokoro/hunt_miracle.py
python scripts/vast/kokoro/search100_collect.py
python scripts/vast/flux_int8/hunt_flux.py
python scripts/vast/qwen_mtp/search20_static.py
```

Templates / onstart: `vast/kokoro-82m/`, `vast/flux2-klein-9b-kv-int8/`, `vast/qwen35-4b-mtp-ngram/`.
Build helpers: `scripts/build_vast_kokoro_template.py`, `scripts/build_vast_flux_template.py`.
