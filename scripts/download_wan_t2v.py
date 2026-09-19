#!/usr/bin/env python3
"""Download Wan2.1 T2V 1.3B Diffusers weights into cache/wan-t2v/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def main() -> int:
    hf_home = Path(os.environ.get("HF_HOME", Path.cwd() / "cache" / "wan-t2v"))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or None
    print(f"HF_HOME={hf_home}")
    print(f"downloading {REPO} ...")
    path = snapshot_download(
        repo_id=REPO,
        token=token,
        cache_dir=str(hub_cache),
        ignore_patterns=["**/*.md", "**/*.gif", "**/*.png"],
        max_workers=2,
    )
    print(f"snapshot -> {path}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
