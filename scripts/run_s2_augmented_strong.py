"""Train strong baselines with the shared dynamic degradation augmentation.

All methods use the same per-batch augmentation distribution as the RCG
coalition backbone. Selection remains clean, and test corruptions are shared.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from rcg.rcg_fusion_pipeline import dynamic_feature_augmentation, load_dataset, stress_conditions
from rcg.strong_models import build_model
from rcg.strong_observation import draw_masks, fit_temperature

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_s2_strong_stress import (METHODS, condition_metadata, metrics,
                                  predict_member)  # noqa: E402


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_augmented(model, train_split, selection_split, seed: int, batch_size: int):
    seed_all(seed)
    device = next(model.parameters()).device
    train_x = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in train_split["x"]]
    train_y = torch.as_tensor(train_split["y"], dtype=torch.long, device=device)
    select_x = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in selection_split["x"]]
    select_y = torch.as_tensor(selection_split["y"], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(100):
        model.train(); losses = []
        for index in torch.randperm(len(train_y), device=device).split(batch_size):
            features = dynamic_feature_augmentation([x[index] for x in train_x], probability=.5)
            labels = train_y[index]; present = draw_masks(len(index), len(features), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features, present)
            loss = F.cross_entropy(output["logits"], labels)
            if hasattr(model, "auxiliary"):
                loss = loss + model.auxiliary(output, labels, present)
            if hasattr(model, "interaction_regularizer"):
                full_rows = present.bool().all(1)
                if full_rows.any():
                    loss = loss + .01 * model.interaction_regularizer(
                        [x[full_rows] for x in features], present[full_rows])
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            full = torch.ones((len(select_y), len(select_x)), device=device)
            selection_nll = float(F.cross_entropy(model(select_x, full)["logits"], select_y))
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)),
                        "selection_nll": selection_nll})
        if selection_nll < best-1e-6:
            best, state, stale = selection_nll, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(state); model.eval()
    return history


@torch.inference_mode()
def full_temperature(model, selection_split, device: str):
    xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in selection_split["x"]]
    labels = selection_split["y"]
    present = torch.ones((len(labels), len(xs)), dtype=torch.float32, device=device)
    logits = model(xs, present)["logits"].cpu().numpy()
    return fit_temperature(logits, labels)


def load_or_train(dataset, fold, method, seed, dims, classes, splits, device):
    root = ROOT / "runs/formal-s2-strong-augmented" / dataset / f"fold_{fold}" / method / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    model = build_model(method, dims, classes).to(device)
    if (root / "model.pt").exists():
        model.load_state_dict(torch.load(root / "model.pt", map_location=device, weights_only=True))
        metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
        return model.eval(), float(metadata["temperature"])
    batch_size = 512 if dataset == "avmnist" else CONFIGS[dataset]["batch_size"]
    history = train_augmented(model, splits["train"], splits["selection"],
                              seed + fold*10000, batch_size)
    temperature = full_temperature(model, splits["selection"], device)
    torch.save(model.state_dict(), root / "model.pt")
    (root / "run.json").write_text(json.dumps({
        "dataset": dataset, "fold": fold, "method": method, "seed": seed,
        "temperature": temperature, "augmentation_probability": .5,
        "clean_selection": True, "history": history,
    }, indent=2), encoding="utf-8")
    return model.eval(), temperature


def run_dataset(dataset: str, device: str):
    folds, names = load_dataset(dataset, ROOT / "data")
    output = ROOT / "runs/formal-s2-strong-augmented" / dataset
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_frames = [], []
    for fold, splits in enumerate(folds):
        dims = [value.shape[1] for value in splits["train"]["x"]]
        classes = int(max(np.max(part["y"]) for part in splits.values()) + 1)
        members = {method: [load_or_train(dataset, fold, method, seed, dims, classes,
                                           splits, device) for seed in SEEDS]
                   for method in METHODS}
        for condition, changed, _ in stress_conditions(splits["test"], names):
            if condition.startswith("missing:"):
                continue
            kind, modality, level, corruption_seed = condition_metadata(condition)
            labels = changed["y"]
            for method, ensemble_members in members.items():
                probability = np.mean([predict_member(model, temperature, changed, device)
                                       for model, temperature in ensemble_members], axis=0)
                accuracy, nll = metrics(probability, labels)
                metric_rows.append({
                    "dataset": dataset, "fold": fold, "method": method, "condition": condition,
                    "corruption_type": kind, "affected_modality": modality,
                    "corruption_level": level, "corruption_seed": corruption_seed,
                    "n_test": len(labels), "accuracy": accuracy, "nll": nll,
                })
                frame = pd.DataFrame({
                    "dataset": dataset, "fold": fold, "method": method, "condition": condition,
                    "corruption_type": kind, "affected_modality": modality,
                    "corruption_level": level, "corruption_seed": corruption_seed,
                    "sample_id": changed["id"], "group_or_video_id": changed["group"],
                    "label": labels,
                })
                for cls in range(classes):
                    frame[f"p{cls}"] = probability[:, cls]
                prediction_frames.append(frame)
            print(f"{dataset} fold={fold} {condition}", flush=True)
        del members
        if device == "cuda": torch.cuda.empty_cache()
    pd.DataFrame(metric_rows).to_csv(output / "metrics_by_condition_fold.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "experiment": "S2 shared-augmentation strong baseline stress", "dataset": dataset,
        "methods": list(METHODS), "seeds": list(SEEDS), "augmentation_probability": .5,
        "clean_selection": True, "shared_test_corruptions": True,
        "test_labels_role": "evaluation only",
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    for dataset in (CONFIGS if args.dataset == "all" else (args.dataset,)):
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
