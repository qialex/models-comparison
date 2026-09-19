#!/usr/bin/env python3
import json
import subprocess

q = (
    "gpu_ram>11.5 cpu_ram>30 disk_space>=20 num_gpus=1 rentable=True "
    "direct_port_count>=2 static_ip=True reliability>0.90"
)
offers = json.loads(
    subprocess.check_output(
        ["vastai", "search", "offers", q, "--order", "dph_total", "--limit", "20", "--raw", "-n"],
        text=True,
    )
)
print("n=", len(offers))
for o in offers[:15]:
    vram = float(o.get("gpu_ram") or 0)
    vram_g = vram / 1024 if vram > 64 else vram
    frac = float(o.get("gpu_frac") if o.get("gpu_frac") is not None else 1)
    ram = float(o.get("cpu_ram") or 0)
    ram_g = ram / 1024 if ram > 512 else ram
    eff = frac * vram_g
    reli = float(o.get("reliability2") or o.get("reliability") or 0)
    ok = eff >= 11.5 and ram_g > 30 and float(o["dph_total"]) < 0.055
    print(
        f"{o['id']} ${float(o['dph_total']):.4f} {o.get('gpu_name')} "
        f"frac={frac:.3f} eff={eff:.1f}G ram={ram_g:.0f} reli={reli:.2f} "
        f"under055={ok} {o.get('geolocation')}"
    )
