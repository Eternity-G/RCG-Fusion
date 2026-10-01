"""Download the public 24k-example AV-MNIST Hugging Face mirror.

This mirror is used for a cross-task pilot because the official MultiBench Google
Drive download is not reachable reliably in the current environment. It must not
be described as the official 70k MultiBench split.
"""
from __future__ import annotations

import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FILES = {
    "train-00000-of-00002.parquet": "98efccc3c2bba2517a6121c1daaecfc4947bd559814c96b205ceb667694db3c7",
    "train-00001-of-00002.parquet": "948dfbcbd9df5638e5d6ee415298aac9cd38a05651c2404161f027a9be843122",
}
BASE = "https://huggingface.co/datasets/pranavmr/AV-MNIST/resolve/main/data/"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    root = Path("data/avmnist_hf/raw")
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = root / name
        if target.exists() and digest(target) == expected:
            print(f"verified {name}", flush=True)
            continue
        partial = target.with_suffix(target.suffix + ".part")
        url = BASE + name
        probe = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "rcg-study/0.1"})
        with urllib.request.urlopen(probe, timeout=120) as response:
            total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        chunk_size = 8 << 20
        chunk_root = target.with_suffix(target.suffix + ".chunks")
        chunk_root.mkdir(exist_ok=True)

        def fetch(start):
            end = min(total-1, start+chunk_size-1)
            chunk = chunk_root / str(start)
            expected_size = end-start+1
            if chunk.exists() and chunk.stat().st_size == expected_size:
                return start, expected_size
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "rcg-study/0.1"}
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        data = response.read()
                    if len(data) != expected_size:
                        raise IOError(f"short range {start}: {len(data)} != {expected_size}")
                    chunk.write_bytes(data)
                    return start, expected_size
                except Exception:
                    if attempt == 3:
                        raise

        starts = list(range(0, total, chunk_size))
        copied = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch, start) for start in starts]
            for future in as_completed(futures):
                _, size = future.result()
                copied += size
                print(f"{name}: {copied / (1 << 20):.1f}/{total / (1 << 20):.1f} MiB", flush=True)
        with open(partial, "wb") as output:
            for start in starts:
                with open(chunk_root / str(start), "rb") as source:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        output.write(block)
        if digest(partial) != expected:
            raise ValueError(f"SHA-256 mismatch for {name}")
        partial.replace(target)
    print("AV-MNIST mirror downloaded and verified", flush=True)


if __name__ == "__main__":
    main()
