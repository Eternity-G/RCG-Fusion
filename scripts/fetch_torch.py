"""Verified, resumable download of a CUDA 12.4 wheel for the inspected driver."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcg.download import ranged_download
from rcg.io import sha256

root = Path(__file__).resolve().parents[1] / ".wheels"
root.mkdir(exist_ok=True)
cpu = "--cpu" in sys.argv
tag = "cpu" if cpu else "cu124"
target = root / f"torch-2.6.0+{tag}-cp311-cp311-win_amd64.whl"
expected = ("24c9d3d13b9ea769dd7bd5c11cfa1fc463fd7391397156565484565ca685d908" if cpu else
            "6a1fb2714e9323f11edb6e8abf7aad5f79e45ad25c081cde87681a18d99c29eb")
if not target.exists():
    partial = target.with_suffix(".part")
    ranged_download("torch-"+tag, f"https://download.pytorch.org/whl/{tag}/torch-2.6.0%2B{tag}-cp311-cp311-win_amd64.whl",
                    partial, total_bytes=206540444 if cpu else 2532350702)
    if sha256(partial) != expected:
        raise ValueError("PyTorch wheel does not match the official index SHA-256")
    partial.replace(target)
if sha256(target) != expected:
    raise ValueError("Existing wheel hash mismatch")
print(f"Verified {target}", flush=True)
