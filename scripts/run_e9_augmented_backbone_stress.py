"""E9 shared dynamic-degradation backbone augmentation control.

All compared policies share a coalition backbone retrained from scratch with
the same dynamic feature augmentation. Existing posterior/router/mixer models
are frozen to isolate whether task-backbone augmentation explains the E9.1
gain. Policy strengths, convex weights, and fallback rho are refit on the clean
selection split; test labels remain evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.rcg_fusion import CoalitionAwareBackbone, nonempty_coalitions
from rcg.rcg_fusion_pipeline import (calibrate_temperatures, fit_backbone,
                                     load_dataset, stress_conditions)
from rcg.stable_ensemble import fit_simplex_weights, mix_actions, select_safe_shrinkage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_e6_adaptive_shrinkage import choose_strength, gate_values, shrink  # noqa: E402
from run_e9_clean_stress import (load_context, member_predict, metadata, metrics)  # noqa: E402


def train_augmented_backbone(dataset: str, fold: int, seed: int, splits: dict,
                             device: str) -> Path:
    output = ROOT / f"runs/formal-e9-augmented-backbones/{dataset}/fold_{fold}/seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "backbone.pt"
    if checkpoint.exists():
        return checkpoint
    dims = [part.shape[1] for part in splits["train"]["x"]]
    classes = int(max(np.max(part["y"]) for part in splits.values()) + 1)
    model = CoalitionAwareBackbone(dims, classes).to(device)
    history = fit_backbone(
        model, splits["train"], splits["selection"], seed=seed + fold * 10000,
        epochs=100, batch_size=CONFIGS[dataset]["batch_size"], patience=10,
        augmentation_probability=.5,
    )
    temperatures = calibrate_temperatures(model, splits["selection"], device)
    torch.save({"state_dict": model.state_dict(), "dims": dims, "classes": classes,
                "temperatures": temperatures}, checkpoint)
    (output / "training.json").write_text(json.dumps({
        "dataset": dataset, "fold": fold, "seed": seed,
        "augmentation_probability": .5,
        "gaussian_levels": [.25, .5, 1., 2.], "mask_levels": [.25, .5, .75],
        "sampling": "one uniformly sampled modality and corruption family per augmented row",
        "clean_selection_early_stopping": True, "epochs_ran": len(history),
        "best_selection_nll": min(row["selection_full_nll"] for row in history),
        "history": history,
    }, indent=2), encoding="utf-8")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return checkpoint


def raw_member(context: dict, split: dict, masks, top_k: int, device: str) -> dict:
    """Run member with temporary zero strengths, retaining raw state."""
    old = context["strengths"]
    context["strengths"] = {"max_confidence": 0., "negative_entropy": 0., "analytic_pi_g2": 0.}
    result = member_predict(context, split, masks, top_k, device)
    context["strengths"] = old
    return result


def select_member_policies(context: dict, split: dict, masks, top_k: int, device: str) -> dict:
    raw = raw_member(context, split, masks, top_k, device)
    labels = split["y"]
    strengths = {}
    for controller in ("max_confidence", "negative_entropy", "analytic_pi_g2"):
        chosen, _ = choose_strength(raw["full_ensemble"], raw["raw_mixer_ensemble"],
                                    raw["posterior_ensemble"], labels, controller)
        strengths[controller] = float(chosen["strength"])
    context["strengths"] = strengths
    return member_predict(context, split, masks, top_k, device)


def fit_a8(selection_outputs: list[dict], labels: np.ndarray):
    actions = []
    for output in selection_outputs:
        actions.extend((output["full_ensemble"], output["A7_equal_ensemble"]))
    actions = np.asarray(actions)
    weights = fit_simplex_weights(actions, labels, l2=1e-3)
    full = np.mean([output["full_ensemble"] for output in selection_outputs], axis=0)
    convex = mix_actions(actions, weights)
    rho, grid = select_safe_shrinkage(full, convex, labels)
    return weights, float(rho), grid


def run_dataset(dataset: str, device: str) -> None:
    folds, names = load_dataset(dataset, ROOT / "data")
    masks = nonempty_coalitions(len(names)); top_k = 3 if len(names) >= 3 else 2
    root = ROOT / f"runs/formal-e9-augmented-{dataset}"
    root.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_frames, configuration = [], [], []
    for fold, splits in enumerate(folds):
        contexts = []
        for seed in SEEDS:
            checkpoint = train_augmented_backbone(dataset, fold, seed, splits, device)
            context = load_context(dataset, fold, seed, masks, device, backbone_path=checkpoint)
            contexts.append(context)
        selection_outputs = [select_member_policies(
            context, splits["selection"], masks, top_k, device) for context in contexts]
        weights, rho, grid = fit_a8(selection_outputs, splits["selection"]["y"])
        for context, output in zip(contexts, selection_outputs):
            configuration.append({
                "dataset": dataset, "fold": fold, "train_seed": context["seed"],
                **{f"{name}_strength": value for name, value in context["strengths"].items()},
                "selection_mean_a7_alpha": float(output["a7_alpha"].mean()), "rho": rho,
            })
        for condition, changed, _ in stress_conditions(splits["test"], names):
            if condition.startswith("missing:"):
                continue
            outputs = [member_predict(context, changed, masks, top_k, device) for context in contexts]
            method_names = ["full_ensemble", "posterior_ensemble", "raw_mixer_ensemble",
                            "confidence_shrink_ensemble", "entropy_shrink_ensemble", "A7_equal_ensemble"]
            ensemble = {method: np.mean([value[method] for value in outputs], axis=0)
                        for method in method_names}
            joint = []
            for output in outputs:
                joint.extend((output["full_ensemble"], output["A7_equal_ensemble"]))
            convex = mix_actions(np.asarray(joint), weights)
            ensemble["A8_safe_fallback"] = (1-rho)*ensemble["full_ensemble"] + rho*convex
            kind, modality, level, corruption_seed = metadata(condition); labels = changed["y"]
            alpha = float(np.mean([value["a7_alpha"].mean() for value in outputs]))
            mixer_weights = np.concatenate([value["mixer_weight"] for value in outputs])
            pi = np.mean([value["analytic_pi"] for value in outputs], axis=0)
            mixture = ensemble["raw_mixer_ensemble"]; rows = np.arange(len(labels))
            realized = np.log(np.clip(mixture[rows, labels], 1e-12, 1) /
                              np.clip(ensemble["full_ensemble"][rows, labels], 1e-12, 1)) > .01
            auroc = float(roc_auc_score(realized, pi)) if np.unique(realized).size == 2 else np.nan
            for method, probability in ensemble.items():
                metric_rows.append({
                    "dataset": dataset, "fold": fold, "method": method, "condition": condition,
                    "corruption_type": kind, "affected_modality": modality,
                    "corruption_level": level, "corruption_seed": corruption_seed,
                    "n_test": len(labels), "mean_a7_alpha": alpha,
                    "mean_full_weight": float(mixer_weights[:, 0].mean()),
                    "mean_posterior_weight": float(mixer_weights[:, 1].mean()),
                    "contribution_auroc": auroc, **metrics(probability, labels, ensemble["full_ensemble"]),
                })
                if method in ("full_ensemble", "A7_equal_ensemble", "A8_safe_fallback"):
                    frame = pd.DataFrame({
                        "dataset": dataset, "fold": fold, "method": method, "condition": condition,
                        "corruption_type": kind, "affected_modality": modality,
                        "corruption_level": level, "corruption_seed": corruption_seed,
                        "sample_id": changed["id"], "group_or_video_id": changed["group"], "label": labels,
                    })
                    for class_index in range(probability.shape[1]):
                        frame[f"p{class_index}"] = probability[:, class_index]
                    prediction_frames.append(frame)
            print(f"{dataset} fold={fold} {condition}: A0={metric_rows[-7]['nll']:.3f} "
                  f"A7={metric_rows[-2]['nll']:.3f} A8={metric_rows[-1]['nll']:.3f}", flush=True)
        pd.DataFrame(metric_rows).to_csv(root / "metrics_by_condition_fold.partial.csv", index=False)
        pd.DataFrame(configuration).to_csv(root / "member_configuration.partial.csv", index=False)
        del contexts
        if device == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(metric_rows).to_csv(root / "metrics_by_condition_fold.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(root / "predictions.parquet", index=False)
    pd.DataFrame(configuration).to_csv(root / "member_configuration.csv", index=False)
    (root / "manifest.json").write_text(json.dumps({
        "experiment": "E9 shared dynamic-degradation backbone augmentation control",
        "dataset": dataset, "seeds": list(SEEDS), "augmentation_probability": .5,
        "noise_levels": [.25, .5, 1., 2.], "mask_levels": [.25, .5, .75],
        "corruption_seeds": [101, 202, 303], "shared_corruption_across_methods_and_members": True,
        "retrained": "coalition task backbone from scratch",
        "frozen": "posterior, listwise router, and candidate mixer from the clean protocol",
        "refit_on_clean_selection": "policy strengths, convex weights, and fallback rho",
        "test_labels_role": "evaluation only",
    }, indent=2), encoding="utf-8")


def main() -> None:
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
