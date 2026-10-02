"""E6: compare fixed, reliability-based, and contribution-probability shrinkage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import analytic_action_probability, predict_mixer
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import apply_residual_blend, predict_router
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.projected_fusion_pipeline import load_posteriors, observed_loss
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs

from run_e4_anchored_routing import CONFIGS, SEEDS
from run_e5_candidate_mixer_ablation import make_actions


ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = (0., .1, .25, .5, .75, 1.)
CONTROLLERS = ("fixed", "max_confidence", "negative_entropy",
               "analytic_pi_g1", "analytic_pi_g2", "analytic_pi_g3")


def gate_values(controller: str, full: np.ndarray, mixture: np.ndarray,
                posterior: np.ndarray) -> np.ndarray:
    if controller == "fixed":
        return np.ones(len(full), dtype=np.float64)
    if controller == "max_confidence":
        return mixture.max(1).astype(np.float64)
    if controller == "negative_entropy":
        p = np.clip(mixture, 1e-8, 1)
        entropy = -(p*np.log(p)).sum(1)/np.log(p.shape[1])
        return np.clip(1-entropy, 0, 1)
    pi = analytic_action_probability(full, mixture, posterior, tolerance=.01)
    gamma = {"analytic_pi_g1": 1., "analytic_pi_g2": 2.,
             "analytic_pi_g3": 3.}[controller]
    return pi**gamma


def shrink(full: np.ndarray, mixture: np.ndarray, gate: np.ndarray,
           strength: float):
    alpha = np.clip(float(strength)*gate, 0, 1)
    probability = (1-alpha[:, None])*full+alpha[:, None]*mixture
    return probability, alpha


def evaluate(probability, full, labels, coalition_losses, alpha):
    metric, prediction = probability_metrics(probability, labels)
    full_metric, full_prediction = probability_metrics(full, labels)
    loss, full_loss = observed_loss(probability, labels), observed_loss(full, labels)
    oracle_loss = coalition_losses.min(1)
    denominator = float(np.mean(full_loss-oracle_loss))
    negative = (full_prediction == labels) & (prediction != labels)
    correction = (full_prediction != labels) & (prediction == labels)
    return {
        **metric, "accuracy_gain": metric["accuracy"]-full_metric["accuracy"],
        "macro_f1_gain": metric["macro_f1"]-full_metric["macro_f1"],
        "nll_gain": full_metric["nll"]-metric["nll"],
        "relative_nll_reduction": (full_metric["nll"]-metric["nll"])/full_metric["nll"],
        "negative_flip_rate": float(negative.mean()),
        "correction_rate": float(correction.mean()),
        "clipped_harm": float(np.minimum(np.maximum(loss-full_loss, 0), 1).mean()),
        "harmful_action_rate": float((loss > full_loss).mean()),
        "regret_reduction": float(1-np.mean(loss-oracle_loss)/max(denominator, 1e-12)),
        "mean_alpha": float(alpha.mean()), "activation_rate": float((alpha > 1e-8).mean()),
        "full_nll": full_metric["nll"], "full_accuracy": full_metric["accuracy"],
    }, loss, prediction


def choose_strength(full, mixture, posterior, labels, controller):
    gate = gate_values(controller, full, mixture, posterior)
    full_loss = observed_loss(full, labels); records = []
    for strength in STRENGTHS:
        probability, alpha = shrink(full, mixture, gate, strength)
        loss = observed_loss(probability, labels)
        records.append({"strength": strength, "selection_nll": float(loss.mean()),
                        "selection_clipped_harm": float(np.minimum(
                            np.maximum(loss-full_loss, 0), 1).mean()),
                        "selection_mean_alpha": float(alpha.mean())})
    feasible = [row for row in records if row["selection_clipped_harm"] <= .02]
    best = min(feasible, key=lambda row: (row["selection_nll"], row["selection_mean_alpha"]))
    return best, records


def run_dataset(dataset: str, device: str):
    cfg = CONFIGS[dataset]; folds, names = load_dataset(dataset, ROOT/"data")
    masks = nonempty_coalitions(len(names)); top_k = 3 if len(names) >= 3 else 2
    e5_root = ROOT/"runs"/f"formal-e5-{dataset}"
    root = ROOT/"runs"/f"formal-e6-{dataset}"; root.mkdir(parents=True, exist_ok=True)
    deployment_rows, curve_rows = [], []
    for fold_index, splits in enumerate(folds):
        for task_seed in SEEDS:
            out = root/f"fold_{fold_index}"/f"seed_{task_seed}"; out.mkdir(parents=True, exist_ok=True)
            backbone, temperatures, dims, classes = _load_backbone(
                cfg["base"]/f"fold_{fold_index}/seed_{task_seed}/backbone.pt", device)
            bundles, posterior, route = {}, {}, {}
            e5_out = e5_root/f"fold_{fold_index}"/f"seed_{task_seed}"
            router_saved = torch.load(e5_out/"router.pt", map_location=device, weights_only=True)
            router = AnalyticResidualListwiseRouter(
                dims, classes, len(masks), residual_scale=.5).to(device)
            router.load_state_dict(router_saved["state_dict"]); router.eval()
            for split_name in ("selection", "test"):
                raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])
                _, posterior[split_name] = load_posteriors(
                    cfg["posterior"], task_seed, dims, classes, masks, splits[split_name],
                    bundles[split_name]["probabilities"], device, fold_index=fold_index)
                value = predict_router(
                    router, splits[split_name], bundles[split_name]["probabilities"],
                    masks, device, posterior[split_name])
                route[split_name] = apply_residual_blend(value, float(router_saved["blend"]))
            actions = {}
            for split_name in ("selection", "test"):
                actions[split_name], _ = make_actions(
                    bundles[split_name]["probabilities"], posterior[split_name],
                    route[split_name], top_k)
            mixer_saved = torch.load(
                e5_out/"anchored_harm_oracle.pt", map_location=device, weights_only=True)
            mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
            mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
            mixtures = {split_name: predict_mixer(
                mixer, actions[split_name], posterior[split_name], device)["probability"]
                for split_name in ("selection", "test")}

            chosen = {}
            for controller in CONTROLLERS:
                chosen[controller], selection_curve = choose_strength(
                    actions["selection"][:, 0], mixtures["selection"],
                    posterior["selection"], splits["selection"]["y"], controller)
                test_gate = gate_values(
                    controller, actions["test"][:, 0], mixtures["test"], posterior["test"])
                for selection_row in selection_curve:
                    probability, alpha = shrink(
                        actions["test"][:, 0], mixtures["test"], test_gate,
                        selection_row["strength"])
                    metric, _, _ = evaluate(
                        probability, actions["test"][:, 0], splits["test"]["y"],
                        bundles["test"]["losses"], alpha)
                    curve_rows.append({"dataset": dataset, "fold": fold_index,
                                       "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                                       "controller": controller, **selection_row, **metric})

            policies = {"full_coalition": ("fixed", 0.),
                        "raw_mixer": ("fixed", 1.),
                        "global_fixed": ("fixed", chosen["fixed"]["strength"]),
                        "max_confidence": ("max_confidence", chosen["max_confidence"]["strength"]),
                        "negative_entropy": ("negative_entropy", chosen["negative_entropy"]["strength"]),
                        "analytic_pi_g1": ("analytic_pi_g1", chosen["analytic_pi_g1"]["strength"]),
                        "analytic_pi_g2": ("analytic_pi_g2", chosen["analytic_pi_g2"]["strength"]),
                        "analytic_pi_g3": ("analytic_pi_g3", chosen["analytic_pi_g3"]["strength"]),
                        "pi2_fixed075": ("analytic_pi_g2", .75)}
            prediction_frames = []
            for variant, (controller, strength) in policies.items():
                gate = gate_values(
                    controller, actions["test"][:, 0], mixtures["test"], posterior["test"])
                probability, alpha = shrink(
                    actions["test"][:, 0], mixtures["test"], gate, strength)
                metric, losses, predictions = evaluate(
                    probability, actions["test"][:, 0], splits["test"]["y"],
                    bundles["test"]["losses"], alpha)
                selection = ({"selection_nll": np.nan, "selection_clipped_harm": np.nan}
                             if variant in ("full_coalition", "raw_mixer", "pi2_fixed075")
                             else chosen[controller])
                deployment_rows.append({"dataset": dataset, "fold": fold_index,
                                        "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                                        "variant": variant, "controller": controller,
                                        "strength": float(strength), **selection, **metric})
                prediction_frames.append(pd.DataFrame({
                    "sample_id": splits["test"]["id"],
                    "group_or_video_id": splits["test"]["group"],
                    "label": splits["test"]["y"], "variant": variant,
                    "loss": losses, "prediction": predictions, "alpha": alpha}))
            pd.concat(prediction_frames, ignore_index=True).to_parquet(
                out/"predictions.parquet", index=False)
            print(f"{dataset} fold={fold_index} seed={task_seed}: "
                  f"pi2={chosen['analytic_pi_g2']['strength']:.2f}", flush=True)

    deployment = pd.DataFrame(deployment_rows)
    curve = pd.DataFrame(curve_rows)
    deployment.to_csv(root/"metrics_by_fold_seed.csv", index=False)
    curve.to_csv(root/"curves_by_fold_seed.csv", index=False)
    id_columns = {"dataset", "variant", "controller", "fold", "train_seed", "n_test"}
    numeric = [c for c in deployment.select_dtypes(include=[np.number]).columns if c not in id_columns]
    seed_rows = []
    for (variant, seed), values in deployment.groupby(["variant", "train_seed"]):
        weights = values.n_test.to_numpy(float)
        row = {"dataset": dataset, "variant": variant, "train_seed": int(seed),
               "n_test": int(weights.sum())}
        row.update({metric: float(np.average(values[metric], weights=weights))
                    for metric in numeric})
        seed_rows.append(row)
    by_seed = pd.DataFrame(seed_rows); by_seed.to_csv(root/"metrics_by_seed.csv", index=False)
    summary = by_seed.groupby("variant")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(root/"metrics_summary.csv", index=False)
    (root/"manifest.json").write_text(json.dumps({
        "experiment": "E6 adaptive shrinkage", "dataset": dataset,
        "source_mixer": "anchored_harm_oracle", "seeds": list(SEEDS),
        "strength_grid": list(STRENGTHS), "selection_harm_budget": .02,
        "main_prespecified_policy": "0.75 * analytic_pi ** 2",
        "selection_role": "choose strength under empirical 2% selection harm budget",
        "test_labels_role": "evaluation only", "model_ensemble": False,
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = CONFIGS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
