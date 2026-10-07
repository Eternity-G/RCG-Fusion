"""I3-1 post-hoc audit of how learned mixers use coalition candidates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import predict_mixer
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import apply_residual_blend, predict_router
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import load_dataset, predict_outputs


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_e5_candidate_mixer_ablation import LEARNED, make_actions  # noqa: E402


def run_dataset(dataset: str, device: str):
    config = CONFIGS[dataset]
    folds, names = load_dataset(dataset, ROOT / "data")
    masks = nonempty_coalitions(len(names)); top_k = 3 if len(names) >= 3 else 2
    rows = []
    for fold, splits in enumerate(folds):
        for seed in SEEDS:
            base = ROOT / "runs" / f"formal-e5-{dataset}" / f"fold_{fold}" / f"seed_{seed}"
            backbone, temperatures, dims, classes = _load_backbone(
                config["base"] / f"fold_{fold}/seed_{seed}/backbone.pt", device)
            probability = predict_outputs(
                backbone, splits["test"]["x"], device, temperatures)["probabilities"]
            _, q = load_posteriors(
                config["posterior"], seed, dims, classes, masks, splits["test"],
                probability, device, fold_index=fold)
            saved_router = torch.load(base / "router.pt", map_location=device, weights_only=True)
            router = AnalyticResidualListwiseRouter(
                dims, classes, len(masks), residual_scale=.5).to(device)
            router.load_state_dict(saved_router["state_dict"]); router.eval()
            route = predict_router(router, splits["test"], probability, masks, device, q)
            route = apply_residual_blend(route, float(saved_router["blend"]))
            actions, _ = make_actions(probability, q, route, top_k)
            labels = splits["test"]["y"]; index = np.arange(len(labels))
            action_loss = -np.log(np.clip(actions[index, :, labels], 1e-12, 1.0))
            full_loss = action_loss[:, 0]
            for variant, variant_config in LEARNED.items():
                saved = torch.load(base / f"{variant}.pt", map_location=device, weights_only=True)
                model = AnchoredCandidateMixer(
                    classes, anchor_full=variant_config["anchor_full"],
                    normalizer=variant_config.get("normalizer", "softmax")).to(device)
                model.load_state_dict(saved["state_dict"]); model.eval()
                output = predict_mixer(model, actions, q, device)
                weight = output["weight"]; candidate_mass = weight[:, 2:].sum(1)
                used = candidate_mass > .1
                base_weight = weight[:, :2] / np.maximum(
                    weight[:, :2].sum(1, keepdims=True), 1e-12)
                without_candidates = (base_weight[:, :, None] * actions[:, :2]).sum(1)
                mixed_probability = output["probability"]
                without_loss = -np.log(np.clip(
                    without_candidates[index, labels], 1e-12, 1.0))
                mixed_loss = -np.log(np.clip(
                    mixed_probability[index, labels], 1e-12, 1.0))
                without_prediction = without_candidates.argmax(1)
                mixed_prediction = mixed_probability.argmax(1)
                normalized = weight[:, 2:] / np.maximum(candidate_mass[:, None], 1e-12)
                candidate_loss = (normalized * action_loss[:, 2:]).sum(1)
                candidate_benefit = full_loss - candidate_loss
                posterior_benefit = full_loss - action_loss[:, 1]
                rows.append({
                    "dataset": dataset, "variant": variant, "fold": fold,
                    "train_seed": seed, "n_test": len(labels),
                    "candidate_usage_rate": float(used.mean()),
                    "mean_candidate_mass": float(candidate_mass.mean()),
                    "candidate_benefit_when_used": float(candidate_benefit[used].mean())
                    if used.any() else np.nan,
                    "candidate_positive_rate_when_used": float((candidate_benefit[used] > 0).mean())
                    if used.any() else np.nan,
                    "mass_weighted_candidate_benefit": float(
                        (candidate_mass * candidate_benefit).mean()),
                    "candidate_marginal_nll_gain": float(
                        (without_loss - mixed_loss).mean()),
                    "candidate_marginal_accuracy_gain": float(
                        ((mixed_prediction == labels).mean()
                         - (without_prediction == labels).mean())),
                    "posterior_mean_benefit": float(posterior_benefit.mean()),
                    "mean_zero_weights": float((weight < 1e-8).sum(1).mean()),
                })
            del backbone, router
            if device == "cuda":
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    out = ROOT / "runs" / f"formal-i3-1-{dataset}"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "candidate_value_by_fold_seed.csv", index=False)
    weighted = []
    metrics = [column for column in frame.columns if column not in {
        "dataset", "variant", "fold", "train_seed", "n_test"}]
    for (variant, seed), part in frame.groupby(["variant", "train_seed"]):
        weights = part.n_test.to_numpy(float)
        row = {"dataset": dataset, "variant": variant, "train_seed": seed,
               "n_test": int(weights.sum())}
        row.update({metric: float(np.average(part[metric], weights=weights))
                    for metric in metrics})
        weighted.append(row)
    pd.DataFrame(weighted).to_csv(out / "candidate_value_by_seed.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = tuple(CONFIGS) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset, device)
        print(f"completed I3-1 candidate audit: {dataset}", flush=True)


if __name__ == "__main__":
    main()
