"""I2-3: hold one modality fixed and intervene on the remaining context.

The RCG track compares invariant unimodal reliability with analytic conditional
contribution and the anchored listwise score on one shared coalition backbone.
The native-score track evaluates TMC/QMF/PDF/I2MoE inside each method's own
model and contribution event.  Different event definitions are never pooled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.io import load_json
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import apply_residual_blend, predict_router
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import load_dataset, predict_outputs
from rcg.strong_models import build_model
from rcg.strong_observation import (all_masks, cremadsplits, mask_name,
                                    mmsasplits, split_tensor)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_e9_clean_stress import load_context, posterior  # noqa: E402


LEVELS = (0.5, 1.0, 2.0)
CORRUPTION_SEEDS = (101, 202, 303)
NATIVE_METHODS = ("concat", "tmc", "qmf", "pdf", "i2moe")
TAU = 0.01


def context_split(split: dict, target: int, level: float, corruption_seed: int,
                  fold: int) -> dict:
    """Keep the target exactly fixed and add nested Gaussian context noise."""
    xs = []
    for modality, value in enumerate(split["x"]):
        if modality == target or level == 0:
            xs.append(value.copy())
        else:
            rng = np.random.default_rng(
                corruption_seed * 100003 + fold * 1009 + modality * 97)
            noise = rng.standard_normal(value.shape, dtype=np.float32)
            xs.append((value + level * noise).astype(np.float32))
    return {**split, "x": xs}


def conditions():
    yield 0.0, -1
    for seed in CORRUPTION_SEEDS:
        for level in LEVELS:
            yield level, seed


def mask_index(masks, wanted):
    return next(i for i, mask in enumerate(masks) if tuple(mask) == tuple(wanted))


def rcg_records(dataset: str, device: str) -> None:
    folds, names = load_dataset(dataset, ROOT / "data")
    masks = nonempty_coalitions(len(names)); full = len(masks) - 1
    output = ROOT / "runs" / f"formal-i2-3-{dataset}" / "rcg"
    for fold, splits in enumerate(folds):
        for task_seed in SEEDS:
            context = load_context(dataset, fold, task_seed, masks, device)
            router_path = (ROOT / "runs" / f"formal-i2-2-{dataset}" /
                           f"fold_{fold}" / f"seed_{task_seed}" /
                           "listwise_pairwise_anchored" / "router.pt")
            saved = torch.load(router_path, map_location=device, weights_only=True)
            dims = [value.shape[1] for value in splits["train"]["x"]]
            router = AnalyticResidualListwiseRouter(
                dims, context["classes"], len(masks), residual_scale=.5).to(device)
            router.load_state_dict(saved["state_dict"]); router.eval()
            context["router"] = router; context["router_blend"] = float(saved["blend"])
            for target, target_name in enumerate(names):
                path = output / f"fold_{fold}" / f"seed_{task_seed}" / f"target_{target_name}.parquet"
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                singleton = tuple(int(i == target) for i in range(len(names)))
                without = tuple(int(i != target) for i in range(len(names)))
                singleton_index = mask_index(masks, singleton)
                without_index = mask_index(masks, without)
                rows = []
                for level, corruption_seed in conditions():
                    changed = context_split(
                        splits["test"], target, level, corruption_seed, fold)
                    output_value = predict_outputs(
                        context["backbone"], changed["x"], device,
                        context["temperatures"])
                    probabilities = output_value["probabilities"]
                    q = posterior(context, changed, probabilities, masks, device)
                    route = predict_router(router, changed, probabilities, masks, device, q)
                    route = apply_residual_blend(route, context["router_blend"])
                    p_full = np.clip(probabilities[:, full], 1e-6, 1.0)
                    p_without = np.clip(probabilities[:, without_index], 1e-6, 1.0)
                    p_single = np.clip(probabilities[:, singleton_index], 1e-8, 1.0)
                    labels = changed["y"]; index = np.arange(len(labels))
                    contribution = np.log(p_full[index, labels] / p_without[index, labels])
                    delta = np.clip(np.log(p_full / p_without), -1.0, 1.0)
                    analytic_expected = (q * delta).sum(1)
                    analytic_probability = (q * (delta > TAU)).sum(1)
                    confidence = p_single.max(1)
                    negative_entropy = 1.0 + (p_single * np.log(p_single)).sum(1) / np.log(p_single.shape[1])
                    scores = {
                        "max_probability": confidence,
                        "negative_entropy": negative_entropy,
                        "analytic_expected": analytic_expected,
                        "analytic_probability": analytic_probability,
                        # The router scores the sub-coalition relative to full;
                        # negation aligns it with the target-addition utility.
                        "anchored_listwise": -route["score"][:, without_index],
                    }
                    for method, score in scores.items():
                        rows.append(pd.DataFrame({
                            "dataset": dataset, "track": "same_backbone",
                            "method": method, "fold": fold,
                            "train_seed": task_seed, "target_modality": target_name,
                            "corruption_level": level,
                            "corruption_seed": corruption_seed,
                            "sample_id": changed["id"],
                            "group_or_video_id": changed["group"],
                            "score": score, "true_contribution": contribution,
                        }))
                pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
                print(f"{dataset} rcg fold={fold} seed={task_seed} target={target_name}", flush=True)
            del context, router
            if device == "cuda":
                torch.cuda.empty_cache()


def strong_splits(dataset: str):
    if dataset == "cremad":
        return cremadsplits(
            ROOT / "data" / "cremad" / "pretrained" /
            "cremad_wav2vec2_r3d18.npz")
    return mmsasplits(ROOT / "data", dataset)


@torch.inference_mode()
def strong_values(model, split, target: int, names, temperatures: dict,
                  native_temperatures, device: str):
    xs, labels_t = split_tensor(split, device); labels = labels_t.cpu().numpy()
    n = len(names); full_mask = (1,) * n
    without_mask = tuple(int(i != target) for i in range(n))

    def forward(mask):
        present = torch.tensor(mask, dtype=torch.float32, device=device)[None].repeat(len(labels), 1)
        value = model(xs, present)
        temperature = float(temperatures[mask_name(mask, names)])
        probability = torch.softmax(value["logits"] / temperature, -1).cpu().numpy()
        return value, probability

    full_output, p_full = forward(full_mask)
    _, p_without = forward(without_mask)
    if model.method == "coalition_concat" and native_temperatures is not None:
        mono = full_output["mono_logits"][:, target] / float(native_temperatures[target])
        native = mono.softmax(-1).max(1).values.cpu().numpy()
    else:
        native = full_output["native_score"][:, target].cpu().numpy()
    index = np.arange(len(labels))
    contribution = np.log(np.clip(p_full[index, labels], 1e-12, 1.0) /
                          np.clip(p_without[index, labels], 1e-12, 1.0))
    return native, contribution


def strong_records(dataset: str, device: str) -> None:
    if dataset == "avmnist":
        return
    folds, names = strong_splits(dataset)
    output = ROOT / "runs" / f"formal-i2-3-{dataset}" / "native"
    for fold, splits in enumerate(folds):
        dims = [value.shape[1] for value in splits["train"]["x"]]
        classes = int(max(split["y"].max() for split in splits.values()) + 1)
        for method in NATIVE_METHODS:
            for task_seed in SEEDS:
                base = (ROOT / "runs" / "strong-observation" / dataset /
                        f"fold_{fold}" / method / f"seed_{task_seed}")
                run = load_json(base / "run.json")
                model = build_model(method, dims, classes).to(device)
                model.load_state_dict(torch.load(
                    base / "model.pt", map_location=device, weights_only=True))
                model.eval()
                for target, target_name in enumerate(names):
                    path = (output / method / f"fold_{fold}" / f"seed_{task_seed}" /
                            f"target_{target_name}.parquet")
                    if path.exists():
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rows = []
                    for level, corruption_seed in conditions():
                        changed = context_split(
                            splits["test"], target, level, corruption_seed, fold)
                        score, contribution = strong_values(
                            model, changed, target, names, run["temperatures"],
                            run.get("native_temperatures"), device)
                        rows.append(pd.DataFrame({
                            "dataset": dataset, "track": "native_within_model",
                            "method": method, "fold": fold,
                            "train_seed": task_seed, "target_modality": target_name,
                            "corruption_level": level,
                            "corruption_seed": corruption_seed,
                            "sample_id": changed["id"],
                            "group_or_video_id": changed["group"],
                            "score": score, "true_contribution": contribution,
                        }))
                    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
                    print(f"{dataset} native={method} fold={fold} seed={task_seed} "
                          f"target={target_name}", flush=True)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="mosi")
    parser.add_argument("--track", choices=("all", "rcg", "native"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = tuple(CONFIGS) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        if args.track in ("all", "rcg"):
            rcg_records(dataset, device)
        if args.track in ("all", "native"):
            strong_records(dataset, device)
        manifest = {
            "experiment": "I2-3 fixed-target context intervention",
            "dataset": dataset, "levels": LEVELS,
            "corruption_seeds": CORRUPTION_SEEDS,
            "target_rule": "target feature is byte-identical; Gaussian noise is added only to other modalities",
            "tracks": {"same_backbone": "shared RCG event",
                       "native_within_model": "each method's own contribution event"},
            "test_labels_role": "true conditional contribution and evaluation only",
        }
        root = ROOT / "runs" / f"formal-i2-3-{dataset}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
