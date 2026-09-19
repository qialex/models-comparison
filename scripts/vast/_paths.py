"""Repo + cache paths for Vast hunt scripts (no secrets)."""
from __future__ import annotations

from pathlib import Path

# scripts/vast/_paths.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def cache_debug(*parts: str) -> Path:
    """e.g. cache_debug('kokoro-82m', '_debug') → <repo>/cache/kokoro-82m/_debug"""
    p = REPO_ROOT.joinpath("cache", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def redact_create(res: object) -> object:
    """Drop instance_api_key from vastai create --raw responses before logging."""
    if isinstance(res, dict):
        out = dict(res)
        out.pop("instance_api_key", None)
        return out
    return res
