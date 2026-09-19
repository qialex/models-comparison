from pathlib import Path

# Patch 1: setup.py — pin sm_86 for docker build without GPU
setup = Path("/tmp/q8/setup.py")
src = setup.read_text(encoding="utf-8")
needle = "def get_device_arch():\n"
insert = (
    "def get_device_arch():\n"
    "    # Patched for docker build without GPU (RTX 3060 / sm_86).\n"
    "    return 8, 6\n"
    "    # original follows (unreachable)\n"
)
if needle not in src:
    raise SystemExit("get_device_arch not found in setup.py")
setup.write_text(src.replace(needle, insert, 1), encoding="utf-8")
print("patched setup.py")
