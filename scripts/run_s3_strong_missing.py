"""Evaluate coalition-valid strong baselines for every available modality set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import load_dataset
from rcg.strong_models import build_model

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("tmc", "qmf", "pdf", "i2moe")
SEEDS = (11, 22, 33, 44, 55)


def mask_name(mask, names):
    return "+".join(name for name, keep in zip(names, mask) if keep)


@torch.inference_mode()
def predict(model, temperature, split, mask, device, batch_size=2048):
    output = []
    for start in range(0, len(split["y"]), batch_size):
        stop = min(len(split["y"]), start+batch_size)
        xs = [torch.as_tensor(value[start:stop], dtype=torch.float32, device=device) for value in split["x"]]
        present = torch.as_tensor(mask, dtype=torch.float32, device=device)[None].repeat(stop-start, 1)
        logits = model(xs, present)["logits"] / temperature
        output.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(output)


def run_dataset(dataset, device):
    folds, names = load_dataset(dataset, ROOT / "data")
    output = ROOT / "runs/formal-s3-strong-missing" / dataset
    output.mkdir(parents=True, exist_ok=True)
    frames, metrics = [], []
    for fold, splits in enumerate(folds):
        dims = [value.shape[1] for value in splits["train"]["x"]]
        classes = int(max(np.max(part["y"]) for part in splits.values())+1)
        for method in METHODS:
            members = []
            for seed in SEEDS:
                root = ROOT / "runs/strong-observation" / dataset / f"fold_{fold}" / method / f"seed_{seed}"
                metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
                model = build_model(method, dims, classes).to(device)
                model.load_state_dict(torch.load(root / "model.pt", map_location=device, weights_only=True))
                model.eval(); members.append((model, metadata["temperatures"]))
            for mask in nonempty_coalitions(len(names)):
                alliance = mask_name(mask, names)
                probability = np.mean([predict(model, temperatures[alliance], splits["test"], mask, device)
                                       for model, temperatures in members], axis=0)
                labels = splits["test"]["y"]; row = np.arange(len(labels))
                metrics.append({
                    "dataset": dataset, "fold": fold, "method": method,
                    "available_mask": "".join(map(str, mask)), "available_modalities": alliance,
                    "n": len(labels), "accuracy": float((probability.argmax(1) == labels).mean()),
                    "nll": float(-np.log(np.clip(probability[row, labels], 1e-12, 1)).mean()),
                })
                frame = pd.DataFrame({
                    "dataset": dataset, "fold": fold, "method": method,
                    "available_mask": "".join(map(str, mask)), "available_modalities": alliance,
                    "sample_id": splits["test"]["id"], "group_or_video_id": splits["test"]["group"],
                    "label": labels,
                })
                for cls in range(classes): frame[f"p{cls}"] = probability[:, cls]
                frames.append(frame)
            del members
            if device == "cuda": torch.cuda.empty_cache()
        print(f"{dataset} fold={fold}", flush=True)
    pd.DataFrame(metrics).to_csv(output / "metrics_by_fold_alliance.csv", index=False)
    pd.concat(frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "experiment": "S3 strong baseline arbitrary available coalition", "dataset": dataset,
        "methods": list(METHODS), "seeds": list(SEEDS), "coalition_valid_training": True,
        "test_labels_role": "evaluation only",
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(); device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    for dataset in (("mosi", "mosei", "cremad", "avmnist") if args.dataset == "all" else (args.dataset,)):
        run_dataset(dataset, device)


if __name__ == "__main__": main()
