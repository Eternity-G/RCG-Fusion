"""Cross-task AV-MNIST mirror pilot for reliability versus coalition contribution.

The public Hugging Face mirror contains 24k examples in one split. This script
creates a fixed stratified 70/15/15 split and uses transparent frozen features:
28x28 grayscale pixels and 130 waveform/spectrum summary values. Results are a
pilot only and are not directly comparable with the official MultiBench split.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import wave
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


MASKS = ((1, 0), (0, 1), (1, 1))
NAMES = {(0, 0): "EMPTY", (1, 0): "I", (0, 1): "A", (1, 1): "IA"}


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def payload(value):
    if isinstance(value, dict):
        return value.get("bytes")
    if hasattr(value, "get"):
        return value.get("bytes")
    raise TypeError(f"Unsupported encoded media value: {type(value)}")


def image_feature(value):
    with Image.open(io.BytesIO(payload(value))) as image:
        array = np.asarray(image.convert("L").resize((28, 28)), dtype=np.float32) / 255
    return array.reshape(-1)


def audio_feature(value):
    raw = payload(value)
    with wave.open(io.BytesIO(raw), "rb") as stream:
        channels, width, rate, frames = (stream.getnchannels(), stream.getsampwidth(),
                                          stream.getframerate(), stream.getnframes())
        samples = stream.readframes(frames)
    if width == 2:
        waveform = np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768
    elif width == 1:
        waveform = (np.frombuffer(samples, dtype=np.uint8).astype(np.float32)-128) / 128
    else:
        raise ValueError(f"Unsupported WAV sample width {width}")
    waveform = waveform.reshape(-1, channels).mean(1)
    if len(waveform) == 0:
        return np.zeros(130, dtype=np.float32)
    windowed = waveform * np.hanning(len(waveform))
    spectrum = np.log1p(np.abs(np.fft.rfft(windowed)))
    source = np.linspace(0, 1, len(spectrum))
    fixed = np.interp(np.linspace(0, 1, 128), source, spectrum)
    rms = np.sqrt(np.mean(np.square(waveform)))
    zcr = np.mean(waveform[1:] * waveform[:-1] < 0) if len(waveform) > 1 else 0
    return np.r_[fixed, rms, zcr].astype(np.float32)


def prepare(root, seed=20260927):
    root = Path(root)
    files = sorted((root / "raw").glob("*.parquet"))
    if len(files) != 2:
        raise FileNotFoundError("Run scripts/download_avmnist.py first")
    table = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if set(table.columns) != {"image", "audio", "label"}:
        raise ValueError(f"Unexpected columns: {table.columns.tolist()}")
    images, audios = [], []
    for index, row in table.iterrows():
        images.append(image_feature(row.image))
        audios.append(audio_feature(row.audio))
        if index % 1000 == 0:
            print(f"decoded {index}/{len(table)}", flush=True)
    image = np.stack(images)
    audio = np.stack(audios)
    label = table.label.to_numpy(dtype=np.int64)
    indices = np.arange(len(label))
    train, remainder = train_test_split(indices, test_size=.30, random_state=seed, stratify=label)
    valid, test = train_test_split(remainder, test_size=.50, random_state=seed, stratify=label[remainder])
    selection, calibration = train_test_split(valid, test_size=.50, random_state=seed,
                                                stratify=label[valid])
    mean_i, std_i = image[train].mean(0), image[train].std(0)
    mean_a, std_a = audio[train].mean(0), audio[train].std(0)
    std_i[std_i < 1e-6] = 1
    std_a[std_a < 1e-6] = 1
    image = (image-mean_i)/std_i
    audio = (audio-mean_a)/std_a
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    for name, idx in (("train", train), ("selection", selection), ("calibration", calibration), ("test", test)):
        np.savez_compressed(out / f"{name}.npz", image=image[idx].astype(np.float32),
                            audio=audio[idx].astype(np.float32), y=label[idx], id=idx)
    audit = {"source": "pranavmr/AV-MNIST Hugging Face community mirror", "n": len(label),
             "split_seed": seed, "splits": {"train": len(train), "selection": len(selection),
                                              "calibration": len(calibration), "test": len(test)},
             "class_counts": np.bincount(label, minlength=10).tolist(),
             "raw_sha256": {path.name: sha256(path) for path in files},
             "feature_protocol": "28x28 pixels + 128 interpolated log-spectrum bins + RMS + ZCR"}
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class Coalition(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.head = Head(sum(dims)+2)

    def forward(self, xs, mask):
        return self.head(torch.cat([xs[i]*mask[:, i:i+1] for i in range(2)] + [mask], 1))


def load_split(root, name, device):
    with np.load(Path(root) / "prepared" / f"{name}.npz") as data:
        return ([torch.tensor(data["image"], device=device), torch.tensor(data["audio"], device=device)],
                torch.tensor(data["y"], dtype=torch.long, device=device), data["id"].copy())


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def fit(model, kind, train, selection, seed, epochs=100):
    seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    best, state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(train[1]), device=train[1].device).split(128):
            xs, y = [x[idx] for x in train[0]], train[1][idx]
            optimizer.zero_grad(set_to_none=True)
            if kind < 2:
                logits = model(xs[kind])
            else:
                options = torch.tensor(MASKS, dtype=torch.float32, device=y.device)
                mask = options[torch.randint(0, 3, (len(idx),), device=y.device)]
                logits = model(xs, mask)
            loss = F.cross_entropy(logits, y)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            if kind < 2:
                logits = model(selection[0][kind])
            else:
                mask = torch.ones((len(selection[1]), 2), device=selection[1].device)
                logits = model(selection[0], mask)
            value = float(F.cross_entropy(logits, selection[1]))
        if value < best-1e-6:
            best, state, stale = value, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(state)


def temperature(logits, labels):
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    x, y = logits.astype(np.float64), labels
    def objective(log_t):
        scaled = x/np.exp(log_t)
        return (logsumexp(scaled, axis=1)-scaled[np.arange(len(y)), y]).mean()
    return float(np.exp(minimize_scalar(objective, bounds=(-4, 4), method="bounded").x))


def bootstrap_mean(values, draws=10000, seed=90210):
    rng = np.random.default_rng(seed)
    output = []
    for start in range(0, draws, 250):
        size = min(250, draws-start)
        idx = rng.integers(0, len(values), (size, len(values)))
        output.extend(values[idx].mean(1))
    return np.quantile(output, [.025, .975])


def run(root, output, seeds=(11, 22, 33, 44, 55), device="cuda"):
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    splits = {name: load_split(root, name, device) for name in ("train", "selection", "calibration", "test")}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for seed in seeds:
        seed_all(seed)
        uni = [Head(784).to(device), Head(130).to(device)]
        coalition = Coalition((784, 130)).to(device)
        for i, model in enumerate(uni): fit(model, i, splits["train"], splits["selection"], seed*10+i)
        fit(coalition, 2, splits["train"], splits["selection"], seed*10+2)
        cx, cy, _ = splits["calibration"]
        temps, thresholds = [], []
        for i, model in enumerate(uni):
            with torch.no_grad(): logits = model(cx[i]).cpu().numpy()
            t = temperature(logits, cy.cpu().numpy()); temps.append(t)
            thresholds.append(np.quantile(torch.softmax(torch.tensor(logits/t), -1).max(1).values.numpy(), .9))
        tx, ty, ids = splits["test"]
        with torch.no_grad():
            mono = torch.stack([torch.softmax(model(tx[i])/temps[i], -1) for i, model in enumerate(uni)], 1)
            losses, probs = {}, {}
            counts = np.bincount(splits["train"][1].cpu().numpy(), minlength=10)
            prior = counts/counts.sum()
            losses[(0, 0)] = -np.log(np.maximum(prior[ty.cpu().numpy()], 1e-12))
            for mask in MASKS:
                m = torch.tensor(mask, dtype=torch.float32, device=device)[None].repeat(len(ty), 1)
                p = coalition(tx, m).softmax(-1)
                probs[mask] = p.cpu().numpy()
                losses[mask] = -p[torch.arange(len(ty), device=device), ty].clamp_min(1e-12).log().cpu().numpy()
        full = losses[(1, 1)]
        stack = np.stack([losses[m] for m in MASKS], 1)
        regret = full-stack.min(1)
        frame = pd.DataFrame({"sample_id": ids, "label": ty.cpu().numpy(), "seed": seed,
                              "full_loss": full, "fusion_regret": regret,
                              "full_prediction": probs[(1, 1)].argmax(1),
                              "oracle_coalition": [NAMES[MASKS[i]] for i in stack.argmin(1)]})
        for i, name in enumerate(("image", "audio")):
            utility = losses[tuple(1 if j != i else 0 for j in range(2))]-full
            own = (1, 0) if i == 0 else (0, 1)
            other = (0, 1) if i == 0 else (1, 0)
            shapley = .5 * ((losses[(0, 0)]-losses[own]) + (losses[other]-full))
            score = mono[:, i].max(1).values.cpu().numpy()
            high = score >= thresholds[i]
            frame[f"reliability_{name}"] = score
            frame[f"conditional_contribution_{name}"] = utility
            frame[f"shapley_contribution_{name}"] = shapley
            for contribution_name, values in (("deletion", utility), ("shapley", shapley)):
                harmful = values < -.01
                hcr = harmful[high].mean()
                interval = bootstrap_mean(harmful[high].astype(float), seed=90210+i+seed)
                metrics.append({"seed": seed, "metric": f"hcr_{contribution_name}", "modality": name,
                                "estimate": hcr, "ci_low_within_seed": interval[0],
                                "ci_high_within_seed": interval[1], "n_high": int(high.sum())})
        metrics.append({"seed": seed, "metric": "mean_fusion_regret", "modality": "all",
                        "estimate": regret.mean(), "ci_low_within_seed": bootstrap_mean(regret, seed=seed)[0],
                        "ci_high_within_seed": bootstrap_mean(regret, seed=seed)[1], "n_high": np.nan})
        correct = (frame.full_prediction == frame.label).to_numpy(dtype=float)
        metrics.append({"seed": seed, "metric": "full_accuracy", "modality": "all",
                        "estimate": correct.mean(), "ci_low_within_seed": bootstrap_mean(correct, seed=seed)[0],
                        "ci_high_within_seed": bootstrap_mean(correct, seed=seed)[1], "n_high": np.nan})
        metrics.append({"seed": seed, "metric": "full_nll", "modality": "all",
                        "estimate": full.mean(), "ci_low_within_seed": bootstrap_mean(full, seed=seed)[0],
                        "ci_high_within_seed": bootstrap_mean(full, seed=seed)[1], "n_high": np.nan})
        oracle_full = (frame.oracle_coalition == "IA").to_numpy(dtype=float)
        metrics.append({"seed": seed, "metric": "oracle_full_rate", "modality": "all",
                        "estimate": oracle_full.mean(),
                        "ci_low_within_seed": bootstrap_mean(oracle_full, seed=seed)[0],
                        "ci_high_within_seed": bootstrap_mean(oracle_full, seed=seed)[1], "n_high": np.nan})
        frame.to_parquet(output / f"seed_{seed}.parquet", index=False)
        print(f"seed {seed}: regret={regret.mean():.4f}", flush=True)
    result = pd.DataFrame(metrics)
    result.to_csv(output / "metrics.csv", index=False)
    summary = result.groupby(["metric", "modality"]).estimate.agg(["mean", "std"]).reset_index()
    lines = ["# AV-MNIST 社区镜像跨任务先导", "",
             "24k 社区镜像、固定分层随机划分和透明冻结特征；不与官方 MultiBench 数值直接比较。", "",
             "| metric | modality | mean | std |", "|---|---|---:|---:|"]
    lines.extend(f"| {r.metric} | {r.modality} | {r['mean']:.4f} | {r['std']:.4f} |" for _, r in summary.iterrows())
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "all"])
    parser.add_argument("--root", default="data/avmnist_hf")
    parser.add_argument("--output", default="runs/avmnist-hf-pilot")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    if args.command in ("prepare", "all"): prepare(args.root)
    if args.command in ("run", "all"): run(args.root, args.output, args.seeds, args.device)
