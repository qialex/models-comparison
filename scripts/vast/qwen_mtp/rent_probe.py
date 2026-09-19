#!/usr/bin/env python3
import json
import subprocess
import sys

# templates
raw = subprocess.check_output(["vastai", "search", "templates", "--raw"], text=True)
items = json.loads(raw)
print("templates", len(items))
hits = []
for t in items:
    name = (t.get("name") or t.get("template_name") or "")
    if any(x in name.lower() for x in ("qwen", "ngram", "4b-mtp", "3.5-4b")):
        hits.append(t)
        print(
            "T",
            name,
            "id=",
            t.get("id"),
            "hash=",
            t.get("hash_id") or t.get("hash") or t.get("template_hash"),
            "img=",
            t.get("image") or t.get("image_path"),
        )
if not hits:
    # dump keys of first
    print("sample keys", list(items[0].keys())[:40] if items else None)

# offer
try:
    o = json.loads(
        subprocess.check_output(
            ["vastai", "search", "offers", "id=49061607", "--raw", "-n"], text=True
        )
    )
    print("offer49061607", o[:1])
except Exception as e:
    print("offer err", e)

# alternatives
q = "gpu_ram>=11.5 cpu_ram>=15 disk_space>=8 num_gpus=1 rentable=True static_ip=True"
offers = json.loads(
    subprocess.check_output(
        ["vastai", "search", "offers", q, "--order", "dph_total", "--limit", "50", "--raw", "-n"],
        text=True,
    )
)
rows = []
for o in offers:
    frac = float(o["gpu_frac"] if o.get("gpu_frac") is not None else 1)
    v = float(o.get("gpu_ram") or 0)
    vg = v / 1024 if v > 64 else v
    if frac * vg < 11.5:
        continue
    p = float(o["dph_total"])
    bw = float(o.get("gpu_mem_bw") or 0)
    score = (bw * frac) / (p * 100) if p else 0
    rows.append((score, p, o))
rows.sort(key=lambda x: -x[0])
print("\ntop score alternatives:")
for score, p, o in rows[:12]:
    print(
        f"{o['id']} score={score:.1f} ${p:.4f} {o.get('gpu_name')} "
        f"frac={o.get('gpu_frac')} {o.get('geolocation')}"
    )
