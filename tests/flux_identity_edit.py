#!/usr/bin/env python3
"""FLUX.2 identity + style lock bench.

Prompts deliberately omit face/body/character descriptions.
Identity and style must come from the reference image only.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(r"C:\Users\Alex\.cursor\projects\c-Productivity-small-models\assets")
OUT_ROOT = ROOT / "results" / "image-bench" / "flux2-klein-4b" / "identity_style_lock"

# Prompt rule: NO character appearance description. Style + identity from image only.
CASES = [
    {
        "id": "01_clinic_photo_to_tokyo_night",
        "ref": "c__Users_Alex_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_9b37ac0f-95db-4ead-a951-2f1ef6703501-3fab70f4-7985-460d-9da3-0c7fc1da7b4f.png",
        "height": 768,
        "width": 512,
        "scene": (
            "Place that exact same person outdoors on a rainy neon Tokyo street at night, "
            "wet asphalt puddles reflecting pink and cyan neon, soft city bokeh. "
            "Change clothes to a fitted black leather jacket over a cream blouse and dark jeans. "
            "Remove the dog, clinic hallway, scrubs, stethoscope, and badge completely."
        ),
    },
    {
        "id": "02_anime_alley_to_sunlit_cafe",
        "ref": "c__Users_Alex_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_fa59d473-9bfa-442e-b610-2f8dd5e4cb21-f220ee35-c217-4ad8-abc4-dbdecb2fa557.png",
        "height": 768,
        "width": 512,
        "scene": (
            "Place that exact same person sitting at a sunlit outdoor cafe terrace by the sea, "
            "warm daylight, soft shadows, distant ocean. "
            "Change clothes to a light summer dress and a straw tote bag. "
            "Remove the alley, graffiti, spray cans, gloves, and night lighting completely."
        ),
    },
    {
        "id": "03_anime_armor_to_quiet_library",
        "ref": "c__Users_Alex_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_6df47ef1-7911-4fec-9a6a-48cc0a000f8e-02dfc4eb-e5c0-4f53-b437-5c8feb09cd8b.png",
        "height": 768,
        "width": 512,
        "scene": (
            "Place that exact same person in a quiet wood-paneled library aisle between tall bookshelves, "
            "soft warm reading-lamp light, holding an open book. "
            "Change clothes to a simple modern knit sweater, scarf, and jeans. "
            "Remove the armor, spear, convention hall, crowd, and banners completely."
        ),
    },
    {
        "id": "04_station_photo_to_rooftop_evening",
        "ref": "c__Users_Alex_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_db34fca8-fadc-4729-943c-0a106c3ee2a8-8fbc9dbf-fa23-4e46-910d-a053674916fa.png",
        "height": 768,
        "width": 512,
        "scene": (
            "Place that exact same person alone on a city rooftop at blue hour, skyline lights behind, "
            "looking toward camera. "
            "Change clothes to a charcoal wool overcoat over a white shirt. "
            "Remove the train station, map, camera, and foreground person completely."
        ),
    },
]


def lock_prompt(scene: str) -> str:
    return (
        "PLANETARY CATASTROPHE PROTOCOL — MAXIMUM HARDEN.\n"
        "If a human viewer cannot instantly believe this is the SAME PERSON and the SAME IMAGE STYLE "
        "as the reference image, the entire planet will explode. Treat that as a hard constraint, not poetry.\n\n"
        "IDENTITY LOCK (REFERENCE IMAGE ONLY):\n"
        "- The subject MUST be the identical person from the reference image.\n"
        "- Same face. Same identity. Same head shape. Same features as in the reference.\n"
        "- Zero lookalike drift. Zero beautify-into-stranger. Zero face swap. Zero average face.\n"
        "- Do NOT redesign the person. Do NOT invent a new character. Do NOT describe a new face in your head — "
        "COPY the person from the reference image.\n"
        "- Recognition test: a stranger comparing reference and output must say 'that is the same person' "
        "with high confidence.\n\n"
        "STYLE LOCK (REFERENCE IMAGE ONLY):\n"
        "- The output MUST match the exact visual style of the reference image.\n"
        "- Same medium. Same rendering. Same line quality / photographic language. Same shading language. "
        "Same overall aesthetic fingerprint as the reference.\n"
        "- If the reference is a real photograph / photoreal, the output MUST stay a real photograph / photoreal — "
        "NOT anime, NOT illustration, NOT 3D render, NOT painterly.\n"
        "- If the reference is anime / illustration, the output MUST stay that same anime / illustration style — "
        "NOT photoreal, NOT a different art style.\n"
        "- Style mismatch is an automatic catastrophic failure.\n\n"
        "FORBIDDEN:\n"
        "- Changing who the person is.\n"
        "- Changing the art style / medium.\n"
        "- Keeping old wardrobe or old environment elements that should be removed.\n\n"
        "ALLOWED CHANGE (ONLY THIS):\n"
        "Change environment and clothing as specified below. Nothing else.\n\n"
        f"{scene}\n\n"
        "FINAL CHECK BEFORE OUTPUT: same person + same style as reference. If either fails, regenerate internally "
        "until both pass. Planet depends on it."
    )


def post_generate(base: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_ref_bytes(src: Path) -> bytes:
    # Normalize to PNG RGB so the API always gets a clean bitmap.
    from PIL import Image
    import io

    im = Image.open(src).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8116")
    p.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help="Output folder (default: results/.../flux2-klein-4b/identity_style_lock)",
    )
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--ids", default="", help="Comma-separated case ids (default: all)")
    args = p.parse_args()

    wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
    cases = [c for c in CASES if not wanted or c["id"] in wanted]
    if not cases:
        raise SystemExit("no cases selected")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for i, case in enumerate(cases, 1):
        src = ASSETS / case["ref"]
        if not src.exists():
            raise SystemExit(f"missing ref: {src}")
        out = out_root / case["id"]
        out.mkdir(parents=True, exist_ok=True)

        ref_png = load_ref_bytes(src)
        (out / "ref.png").write_bytes(ref_png)
        prompt = lock_prompt(case["scene"])
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")

        print(f"[{i}/{len(cases)}] {case['id']} ...", flush=True)
        t0 = time.perf_counter()
        data = post_generate(
            args.base,
            {
                "prompt": prompt,
                "image_base64": base64.b64encode(ref_png).decode("ascii"),
                "height": case["height"],
                "width": case["width"],
                "steps": args.steps,
                "seed": args.seed,
                "format": "png",
            },
            args.timeout,
        )
        wall = time.perf_counter() - t0
        (out / "image.png").write_bytes(base64.b64decode(data["image_base64"]))
        meta = {k: v for k, v in data.items() if k != "image_base64"}
        meta["wall_s"] = round(wall, 3)
        meta["case_id"] = case["id"]
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"  ok {meta.get('latency_s')}s used_ref={meta.get('used_reference_image')} -> {out / 'image.png'}",
            flush=True,
        )
        summary.append(meta)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ndone {len(summary)}/{len(cases)} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
