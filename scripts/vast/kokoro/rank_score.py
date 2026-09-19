#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import cache_debug  # noqa: E402

_dbg = cache_debug("kokoro-82m", "_debug")
u = json.loads((_dbg / "search100" / "unique_offers.json").read_text(encoding="utf-8"))
rows = []
for oid, rec in u.items():
    d = rec["offer"]["derived"]
    raw = rec["offer"]
    if not d.get("kokoro_core_ok"):
        continue
    score = d.get("score_bw_per_cent")
    if score is None:
        continue
    rows.append(
        {
            "id": int(oid) if str(oid).isdigit() else oid,
            "score": round(float(score), 2),
            "price": round(d["price"], 4),
            "gpu": d.get("gpu") or "?",
            "frac": d.get("frac"),
            "eff_g": d.get("eff_vram_g"),
            "bw": raw.get("gpu_mem_bw"),
            "static": bool(d.get("static_ip")),
            "reli": round(float(d.get("reli") or 0), 3),
            "verified": bool(d.get("verified")),
            "loc": d.get("loc") or "",
            "seen": rec.get("seen_rounds"),
        }
    )
rows.sort(key=lambda r: (-r["score"], r["price"]))
out = _dbg / "search100" / "ranked_by_score.json"
out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print("n", len(rows), "wrote", out)
for r in rows[:20]:
    print(
        f"{r['score']:7.2f} ${r['price']:.4f} {r['gpu']:16} "
        f"frac={r['frac']} static={r['static']} reli={r['reli']:.3f} {r['loc']} id={r['id']}"
    )
