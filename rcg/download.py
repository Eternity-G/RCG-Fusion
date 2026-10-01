"""Download public feature mirrors, verify the MMSA-published SHA before deserialization."""
from __future__ import annotations

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .io import sha256, write_json

HASHES = {
    "mosi": "d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9",
    "mosei": "45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791",
}
URLS = {
    "mosi": "https://huggingface.co/datasets/tamb2203579/CMU-MOSI/resolve/main/Processed/aligned_50.pkl?download=true",
    "mosei": "https://huggingface.co/datasets/tamb2203579/CMU-MOSEI/resolve/main/Processed/aligned_50.pkl?download=true",
}
SIZES = {"mosi": 367257274, "mosei": 4683341880}


def ranged_download(dataset, url, partial, total_bytes=None):
    """Bounded parallel ranges avoid silently truncated long proxy transfers."""
    size, chunk = total_bytes or SIZES[dataset], 8 * 1024**2
    parts = partial.parent / (partial.name + ".chunks")
    parts.mkdir(exist_ok=True)
    # Reuse only complete chunks of an earlier interrupted full transfer.
    if partial.exists() and partial.stat().st_size < size:
        with partial.open("rb") as src:
            for offset in range(0, partial.stat().st_size // chunk * chunk, chunk):
                path = parts / str(offset)
                block = src.read(chunk)
                if not path.exists():
                    path.write_bytes(block)
    offsets = list(range(0, size, chunk))
    def fetch(offset):
        end = min(size, offset+chunk)-1
        part = parts / str(offset)
        if part.exists() and part.stat().st_size == end-offset+1:
            return part
        for attempt in range(6):
            try:
                separator = "&" if "?" in url else "?"
                request = urllib.request.Request(f"{url}{separator}rcgpart={offset}&attempt={attempt}",
                    headers={"User-Agent": "RCG-Research/0.1", "Range": f"bytes={offset}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected = f"bytes {offset}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise IOError("Server did not honor requested byte range")
                    content = response.read(end-offset+2)
                if len(content) != end-offset+1:
                    raise IOError("Incomplete byte range")
                part.write_bytes(content)
                return part
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(attempt+1, 5))
    workers = min(16, max(1, int(os.environ.get("RCG_DOWNLOAD_WORKERS", "4"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(fetch, offsets)):
            if (i+1) % 16 == 0:
                print(f"{dataset}: verified-length chunks {(i+1)*8} MiB / {size/1024**2:.0f} MiB", flush=True)
    with partial.open("wb") as dst:
        for offset in offsets:
            with (parts / str(offset)).open("rb") as src:
                dst.write(src.read())


def download(dataset, root, url=None):
    target = Path(root) / dataset / "aligned_50.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) == HASHES[dataset]:
            print(f"Verified existing {target}", flush=True)
            return target
        raise ValueError(f"Existing file hash mismatch: {target}; preserve it and inspect before replacing")
    url = url or URLS.get(dataset)
    if not url:
        raise ValueError("No verified mirror configured; use --url or obtain the MMSA aligned_50.pkl and run prepare")
    partial = target.with_suffix(".pkl.part")
    ranged_download(dataset, url, partial)
    digest = sha256(partial)
    if digest != HASHES[dataset]:
        raise ValueError(f"Downloaded hash {digest} does not match MMSA; retained .part without loading")
    os.replace(partial, target)
    write_json(target.with_suffix(".source.json"), {
        "url": url, "sha256": digest,
        "reference": "https://github.com/thuiar/MMSA#2-datasets",
        "note": "Mirror content matches the hash published by MMSA. No raw videos downloaded."
    })
    print(f"Verified {target}", flush=True)
    return target
