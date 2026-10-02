"""E8 bridge experiment: aggregate the actual A7 member predictions into A8."""
from __future__ import annotations

import argparse
import json
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
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs
from rcg.stable_ensemble import fit_simplex_weights, mix_actions, select_safe_shrinkage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_e5_candidate_mixer_ablation import make_actions  # noqa: E402
from run_e6_adaptive_shrinkage import gate_values, shrink  # noqa: E402


def evaluate(probability, labels, full):
    metric, prediction = probability_metrics(probability, labels)
    full_metric, full_prediction = probability_metrics(full, labels)
    rows = np.arange(len(labels))
    loss = -np.log(np.clip(probability[rows, labels], 1e-12, 1))
    full_loss = -np.log(np.clip(full[rows, labels], 1e-12, 1))
    return {**metric,
            "accuracy_gain": metric["accuracy"]-full_metric["accuracy"],
            "nll_gain": full_metric["nll"]-metric["nll"],
            "negative_flip_rate": float(((full_prediction == labels) & (prediction != labels)).mean()),
            "correction_rate": float(((full_prediction != labels) & (prediction == labels)).mean()),
            "clipped_harm": float(np.minimum(np.maximum(loss-full_loss, 0), 1).mean())}


def selected_strength(dataset, fold, seed):
    frame = pd.read_csv(ROOT/f"runs/formal-e6-{dataset}/metrics_by_fold_seed.csv")
    row = frame[(frame.variant == "analytic_pi_g2") &
                (frame.fold == fold) & (frame.train_seed == seed)]
    if len(row) != 1: raise ValueError(f"missing E6 strength: {dataset} fold={fold} seed={seed}")
    return float(row.iloc[0].strength)


def member_outputs(dataset, fold_index, splits, names, seed, device):
    cfg = CONFIGS[dataset]; masks = nonempty_coalitions(len(names))
    top_k = 3 if len(names) >= 3 else 2
    backbone, temperatures, dims, classes = _load_backbone(
        cfg["base"]/f"fold_{fold_index}/seed_{seed}/backbone.pt", device)
    checkpoint = ROOT/f"runs/formal-e5-{dataset}/fold_{fold_index}/seed_{seed}"
    router_saved = torch.load(checkpoint/"router.pt", map_location=device, weights_only=True)
    router = AnalyticResidualListwiseRouter(dims, classes, len(masks), residual_scale=.5).to(device)
    router.load_state_dict(router_saved["state_dict"]); router.eval()
    mixer_saved = torch.load(checkpoint/"anchored_harm_oracle.pt", map_location=device, weights_only=True)
    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
    mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
    strength = selected_strength(dataset, fold_index, seed)
    result = {}
    for split_name in ("selection", "test"):
        raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
        bundle = attach_observed_losses(raw, splits[split_name]["y"])
        _, posterior = load_posteriors(cfg["posterior"], seed, dims, classes, masks,
                                       splits[split_name], bundle["probabilities"], device,
                                       fold_index=fold_index)
        route = predict_router(router, splits[split_name], bundle["probabilities"],
                               masks, device, posterior)
        route = apply_residual_blend(route, float(router_saved["blend"]))
        actions, _ = make_actions(bundle["probabilities"], posterior, route, top_k)
        mixture = predict_mixer(mixer, actions, posterior, device)["probability"]
        gate = gate_values("analytic_pi_g2", actions[:, 0], mixture, posterior)
        a7, alpha = shrink(actions[:, 0], mixture, gate, strength)
        result[split_name] = {"full": actions[:, 0], "a7": a7, "alpha": alpha}
    return result, strength


def run_dataset(dataset, device):
    folds, names = load_dataset(dataset, ROOT/"data")
    root = ROOT/f"runs/formal-e8-{dataset}"; root.mkdir(parents=True, exist_ok=True)
    metrics, predictions, weights, configs = [], [], [], []
    for fold_index, splits in enumerate(folds):
        members = {"selection": {"full": [], "a7": []}, "test": {"full": [], "a7": []}}
        for seed in SEEDS:
            output, strength = member_outputs(dataset, fold_index, splits, names, seed, device)
            configs.append({"dataset": dataset, "fold": fold_index, "seed": seed,
                            "a7_strength": strength, "selection_mean_alpha": output["selection"]["alpha"].mean(),
                            "test_mean_alpha": output["test"]["alpha"].mean()})
            for split in members:
                for action in members[split]: members[split][action].append(output[split][action])
        for split in members:
            for action in members[split]: members[split][action] = np.asarray(members[split][action])
        sel_full = members["selection"]["full"].mean(0); test_full = members["test"]["full"].mean(0)
        sel_a7 = members["selection"]["a7"].mean(0); test_a7 = members["test"]["a7"].mean(0)
        # Joint action set implements A8: each task member contributes its full and A7 action.
        sel_actions = np.empty((2*len(SEEDS), *sel_full.shape), dtype=np.float64)
        test_actions = np.empty((2*len(SEEDS), *test_full.shape), dtype=np.float64)
        names_actions = []
        for i, seed in enumerate(SEEDS):
            sel_actions[2*i:2*i+2] = [members["selection"]["full"][i], members["selection"]["a7"][i]]
            test_actions[2*i:2*i+2] = [members["test"]["full"][i], members["test"]["a7"][i]]
            names_actions.extend((f"seed_{seed}_full", f"seed_{seed}_a7"))
        weight = fit_simplex_weights(sel_actions, splits["selection"]["y"], l2=1e-3)
        sel_convex = mix_actions(sel_actions, weight); test_convex = mix_actions(test_actions, weight)
        rho, grid = select_safe_shrinkage(sel_full, sel_convex, splits["selection"]["y"])
        sel_safe = (1-rho)*sel_full + rho*sel_convex
        test_safe = (1-rho)*test_full + rho*test_convex
        methods = {"A0_full_ensemble": (sel_full, test_full),
                   "A7_equal_ensemble": (sel_a7, test_a7),
                   "A8_joint_convex": (sel_convex, test_convex),
                   "A8_safe_fallback": (sel_safe, test_safe)}
        for method, (selection_probability, test_probability) in methods.items():
            labels = splits["test"]["y"]; selection_labels = splits["selection"]["y"]
            metrics.append({"dataset": dataset, "fold": fold_index, "method": method,
                            "n_test": len(labels), "rho": rho,
                            "selection_accuracy": float((selection_probability.argmax(1) == selection_labels).mean()),
                            "selection_nll": float(-np.log(np.clip(selection_probability[
                                np.arange(len(selection_labels)), selection_labels], 1e-12, 1)).mean()),
                            **evaluate(test_probability, labels, test_full)})
            frame = pd.DataFrame({"dataset": dataset, "fold": fold_index, "method": method,
                                  "sample_id": splits["test"]["id"],
                                  "group_or_video_id": splits["test"]["group"], "label": labels})
            for k in range(test_probability.shape[1]): frame[f"p{k}"] = test_probability[:, k]
            predictions.append(frame)
        for action, value in zip(names_actions, weight):
            weights.append({"dataset": dataset, "fold": fold_index, "action": action,
                            "weight": float(value), "rho": rho})
        print(f"{dataset} fold={fold_index}: rho={rho:.1f} "
              f"A0={evaluate(test_full, splits['test']['y'], test_full)['nll']:.4f} "
              f"A7={evaluate(test_a7, splits['test']['y'], test_full)['nll']:.4f} "
              f"A8={evaluate(test_safe, splits['test']['y'], test_full)['nll']:.4f}", flush=True)
    metrics = pd.DataFrame(metrics); metrics.to_csv(root/"metrics_by_fold.csv", index=False)
    rows = []
    for method, values in metrics.groupby("method"):
        w = values.n_test.to_numpy(float); row = {"dataset": dataset, "method": method, "n_test": int(w.sum())}
        for col in values.select_dtypes(include=[np.number]).columns:
            if col not in {"fold", "n_test"}: row[col] = float(np.average(values[col], weights=w))
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/"metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(root/"predictions.parquet", index=False)
    pd.DataFrame(weights).to_csv(root/"weights.csv", index=False)
    pd.DataFrame(configs).to_csv(root/"member_configuration.csv", index=False)
    (root/"manifest.json").write_text(json.dumps({"experiment": "E8 actual A7-to-A8 bridge",
        "dataset": dataset, "seeds": list(SEEDS), "joint_actions": "five full + five A7",
        "simplex_l2": 1e-3, "selection_role": "A7 strength inherited from E6; fit weights and rho",
        "test_labels_role": "evaluation only"}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    for dataset in (CONFIGS if args.dataset == "all" else (args.dataset,)): run_dataset(dataset, device)


if __name__ == "__main__": main()
