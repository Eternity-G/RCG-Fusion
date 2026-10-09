"""Evaluate compute-matched strong fusion baselines under shared corruptions.

The corruption generator is exactly the one used by the frozen RCG chain.
Every method and seed therefore receives the same changed feature arrays.
Only calibrated full-coalition predictions are evaluated here; missing-modal
conditions belong to S3 and are deliberately excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.rcg_fusion_pipeline import load_dataset, stress_conditions
from rcg.strong_models import build_model


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
METHODS = ("tmc", "qmf", "pdf", "i2moe")


def condition_metadata(condition: str):
    if condition == "clean":
        return "clean", "none", 0., -1
    kind, modality, level, seed = condition.split(":")
    return kind, modality, float(level), int(seed.removeprefix("seed"))


def load_members(dataset: str, fold: int, dims, classes: int, names, device: str):
    loaded = {}
    for method in METHODS:
        members = []
        for seed in SEEDS:
            root = ROOT / "runs" / "strong-observation" / dataset / f"fold_{fold}" / method / f"seed_{seed}"
            metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
            model = build_model(method, dims, classes).to(device)
            model.load_state_dict(torch.load(root / "model.pt", map_location=device, weights_only=True))
            model.eval()
            full_name = "+".join(names)
            members.append((model, float(metadata["temperatures"][full_name])))
        loaded[method] = members
    return loaded


@torch.inference_mode()
def predict_member(model, temperature: float, split: dict, device: str, batch_size: int = 2048):
    probabilities = []
    n = len(split["y"])
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        xs = [torch.as_tensor(value[start:stop], dtype=torch.float32, device=device)
              for value in split["x"]]
        present = torch.ones((stop-start, len(xs)), dtype=torch.float32, device=device)
        logits = model(xs, present)["logits"] / temperature
        probabilities.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(probabilities)


def metrics(probability: np.ndarray, labels: np.ndarray):
    rows = np.arange(len(labels))
    loss = -np.log(np.clip(probability[rows, labels], 1e-12, 1))
    prediction = probability.argmax(1)
    return float((prediction == labels).mean()), float(loss.mean())


def run_dataset(dataset: str, device: str):
    folds, names = load_dataset(dataset, ROOT / "data")
    output = ROOT / "runs" / "formal-s2-strong-clean" / dataset
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_frames = [], []
    for fold, splits in enumerate(folds):
        dims = [value.shape[1] for value in splits["train"]["x"]]
        classes = int(max(np.max(part["y"]) for part in splits.values()) + 1)
        members = load_members(dataset, fold, dims, classes, names, device)
        for condition, changed, _ in stress_conditions(splits["test"], names):
            if condition.startswith("missing:"):
                continue
            kind, modality, level, corruption_seed = condition_metadata(condition)
            labels = changed["y"]
            for method, method_members in members.items():
                member_probability = [predict_member(model, temperature, changed, device)
                                      for model, temperature in method_members]
                probability = np.mean(member_probability, axis=0)
                accuracy, nll = metrics(probability, labels)
                metric_rows.append({
                    "dataset": dataset, "fold": fold, "method": method,
                    "condition": condition, "corruption_type": kind,
                    "affected_modality": modality, "corruption_level": level,
                    "corruption_seed": corruption_seed, "n_test": len(labels),
                    "accuracy": accuracy, "nll": nll,
                })
                frame = pd.DataFrame({
                    "dataset": dataset, "fold": fold, "method": method,
                    "condition": condition, "corruption_type": kind,
                    "affected_modality": modality, "corruption_level": level,
                    "corruption_seed": corruption_seed, "sample_id": changed["id"],
                    "group_or_video_id": changed["group"], "label": labels,
                })
                for cls in range(classes):
                    frame[f"p{cls}"] = probability[:, cls]
                prediction_frames.append(frame)
            print(f"{dataset} fold={fold} {condition}", flush=True)
        del members
        if device == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(metric_rows).to_csv(output / "metrics_by_condition_fold.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "experiment": "S2 compute-matched strong baseline clean-only stress",
        "dataset": dataset, "methods": list(METHODS), "seeds": list(SEEDS),
        "training_protocol": "clean-only", "shared_corruption_generator":
        "rcg.rcg_fusion_pipeline.stress_conditions", "test_labels_role": "evaluation only",
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    for dataset in (("mosi", "mosei", "cremad", "avmnist") if args.dataset == "all" else (args.dataset,)):
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
