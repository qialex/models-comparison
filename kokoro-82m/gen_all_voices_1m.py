#!/usr/bin/env python3
"""Generate ~1 minute WAV per Kokoro voice into results/kokoro-82m/."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "kokoro-82m"
BASE = "http://127.0.0.1:8131"
# ~1000 chars ≈ 1 min (HF rule of thumb)
UNIT = (
    "Kokoro is an open text to speech model with many preset voices. "
    "This sample is about one minute long so you can compare tone, clarity, and pacing. "
    "The quick brown fox jumps over the lazy dog near the river at dawn. "
)
TEXT = (UNIT * 5)[:1000]

VOICES = [
    # American English
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    # British English
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    # Japanese
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    # Mandarin
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    # Spanish / French / Hindi / Italian / Portuguese
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "pf_dora", "pm_alex", "pm_santa",
]


def tts(voice: str) -> dict:
    payload = json.dumps(
        {
            "text": TEXT,
            "voice": voice,
            "speed": 1.0,
            "return_audio": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/tts",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = []
    print(f"out={OUT_DIR} voices={len(VOICES)} chars={len(TEXT)}")
    for i, voice in enumerate(VOICES, 1):
        path = OUT_DIR / f"{voice}.wav"
        print(f"[{i}/{len(VOICES)}] {voice} ...", flush=True)
        t0 = time.perf_counter()
        try:
            body = tts(voice)
            path.write_bytes(base64.b64decode(body["wav_base64"]))
            row = {
                "voice": voice,
                "file": path.name,
                "audio_s": body.get("audio_s"),
                "generate_s": body.get("generate_s"),
                "rtf": body.get("rtf"),
                "ok": True,
            }
            print(
                f"  -> {path.name} audio={body.get('audio_s')}s "
                f"gen={body.get('generate_s')}s ({time.perf_counter()-t0:.1f}s wall)"
            )
        except urllib.error.HTTPError as e:
            err = e.read()[:300].decode(errors="replace")
            row = {"voice": voice, "ok": False, "error": f"HTTP {e.code}: {err}"}
            print(f"  FAIL {row['error']}")
        except Exception as e:
            row = {"voice": voice, "ok": False, "error": str(e)}
            print(f"  FAIL {e}")
        meta.append(row)

    summary = {
        "base": BASE,
        "text_chars": len(TEXT),
        "outdir": str(OUT_DIR),
        "ok": sum(1 for m in meta if m.get("ok")),
        "fail": sum(1 for m in meta if not m.get("ok")),
        "voices": meta,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\ndone ok={summary['ok']} fail={summary['fail']} -> {OUT_DIR}")
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
