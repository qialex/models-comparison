"""Fix q8_kernels Ampere path: attention never set use_default for sm_80-86."""

from pathlib import Path

p = Path("/opt/conda/lib/python3.11/site-packages/q8_kernels/integration/utils.py")
src = p.read_text(encoding="utf-8")
old = """    elif device_arch == "ada" or device_arch == "blackwell":
"""
new = """    elif device_arch == "ampere":
        # Official code omitted Ampere here → UnboundLocalError on RTX 30xx.
        use_default = True
    elif device_arch == "ada" or device_arch == "blackwell":
"""
if "elif device_arch == \"ampere\":" in src:
    print("ampere attention patch already present")
elif old not in src:
    raise SystemExit("attention branch marker not found")
else:
    p.write_text(src.replace(old, new, 1), encoding="utf-8")
    print("patched ampere attention fallback")
