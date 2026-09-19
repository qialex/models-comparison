# Video generation bench (LTX-Video)

Prompts + refs are tracked; outputs under `results/video-bench/` are gitignored via `results/`.

## Goal

Animate the **4 style/identity stills** used in the FLUX identity lock check, as **image-to-video** suites.

Judge each clip on:

1. **Same person** as the first-frame ref
2. **Same visual style** (photo stays photo; anime stays anime)
3. **Motion quality** (smooth, matches the case intent)

## Refs

| ID | File | Source still |
| --- | --- | --- |
| `01_clinic_vet` | `refs/01_clinic_vet.png` | Clinic / dog photo |
| `02_anime_alley` | `refs/02_anime_alley.png` | Anime alley / spray cans |
| `03_anime_armor` | `refs/03_anime_armor.png` | Anime convention / armor |
| `04_station_man` | `refs/04_station_man.png` | Station / map photo |

Shared manifest: `refs_manifest.json`

## Suites

| Key | File | Cases |
| --- | --- | --- |
| `style_lock` | `cases_i2v_style_lock.json` | 10 mixed ambient/camera/action |
| `a_identity` | `cases_a_identity_stress.json` | 10 micro-motion identity stress |
| `b_camera` | `cases_b_camera_grammar.json` | 10 camera grammar |
| `c_action` | `cases_c_action_ladder.json` | 10 action ladder |
| `d_style_trap` | `cases_d_style_trap.json` | 10 style-trap prompts |
| `e_multi` | `cases_e_multi_condition.json` | 10 first+last frame |

Defaults: **320×576** (9:16, LTX ÷32), **97 frames** (~4 s @ 24 fps), seed **42**.

## Model / download

Repo: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video/tree/main) (~254 GB total — **do not** snapshot all).

Selective pull into `cache/ltx-video/` (~27 GB + 505 MB upscaler):

- `ltxv-2b-0.9.8-distilled.safetensors` (6.34 GB)
- `ltxv-spatial-upscaler-0.9.8.safetensors` (505 MB)
- `text_encoder/`, `tokenizer/`, `scheduler/`, `vae/`, `model_index.json`

Service `:8122` uses the **official** [`Lightricks/LTX-Video`](https://github.com/Lightricks/LTX-Video) Python package (`RectifiedFlowScheduler` + multi-scale YAML), not Diffusers’ `LTXImageToVideoPipeline` alone — the Diffusers-only path produced first-frame-then-noise on this checkpoint.

```powershell
$env:HF_HOME = "$PWD\cache\ltx-video"
python scripts/download_ltx_video.py
```

Service: `ltx_video_gpu` on **:8122**. Stop other GPU image/video services first.

```powershell
docker compose --profile gpu stop flux2_klein_gpu
docker compose --profile gpu up -d --build ltx_video_gpu
curl.exe -s http://localhost:8122/health
python tests/video-bench/run.py --suite all --base http://127.0.0.1:8122 --skip-existing
```

Results: `results/video-bench/<model>/<suite>/<case_id>/{video.mp4,prompt.txt,meta.json}`
