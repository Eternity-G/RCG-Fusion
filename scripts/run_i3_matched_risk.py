"""I3-2: evaluate shrinkage controllers at matched empirical harm budgets.

The primary controller family shares one frozen candidate-mixer action.  Native
TMC/QMF/PDF scores are converted to an uncertainty percentile using only the
selection split, so method-specific score scales do not alter the action or the
strength grid.  A posterior convex-projection action is retained as a distinct
architectural baseline and is never pooled with the matched-action comparison.
"""
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
from rcg.projected_fusion import build_projection
from rcg.projected_fusion_pipeline import load_posteriors, observed_loss
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs
from rcg.strong_models import build_model

from run_e4_anchored_routing import CONFIGS, SEEDS
from run_e5_candidate_mixer_ablation import make_actions


ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = tuple(float(value) for value in np.linspace(0, 1, 41))
BUDGETS = (.005, .01, .02, .05)
CORE_CONTROLLERS = ("fixed", "max_confidence", "negative_entropy",
                    "analytic_pi_g1", "analytic_pi_g2", "analytic_pi_g3")
NATIVE_METHODS = ("tmc", "qmf", "pdf")


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values to [0,1] using a selection-only empirical CDF."""
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, values, side="right") / max(len(ordered), 1)


def native_gate(dataset: str, method: str, fold: int, seed: int, splits: dict,
                dims: list[int], classes: int, device: str):
    checkpoint = (ROOT / "runs" / "strong-observation" / dataset /
                  f"fold_{fold}" / method / f"seed_{seed}" / "model.pt")
    if not checkpoint.exists():
        return None
    model = build_model(method, dims, classes).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval(); raw = {}
    with torch.inference_mode():
        for split_name in ("selection", "test"):
            xs = [torch.as_tensor(value, dtype=torch.float32, device=device)
                  for value in splits[split_name]["x"]]
            mask = torch.ones((len(xs[0]), len(xs)), dtype=torch.float32, device=device)
            score = model(xs, mask)["native_score"].detach().cpu().numpy()
            # Lower aggregate native reliability means a larger correction gate.
            raw[split_name] = -np.asarray(score, dtype=np.float64).mean(1)
    return {
        "selection": empirical_percentile(raw["selection"], raw["selection"]),
        "test": empirical_percentile(raw["selection"], raw["test"]),
    }


def gate_values(controller: str, full: np.ndarray, action: np.ndarray,
                posterior: np.ndarray, native: dict[str, np.ndarray] | None = None,
                split: str | None = None) -> np.ndarray:
    if controller == "fixed":
        return np.ones(len(full), dtype=np.float64)
    if controller == "max_confidence":
        classes = action.shape[1]
        return np.clip((action.max(1)-1/classes)/(1-1/classes), 0, 1)
    if controller == "negative_entropy":
        probability = np.clip(action, 1e-8, 1)
        entropy = -(probability*np.log(probability)).sum(1)/np.log(classes := probability.shape[1])
        return np.clip(1-entropy, 0, 1)
    if controller.startswith("native_"):
        if native is None or split is None:
            raise ValueError("native controller requires selection-derived gate values")
        return native[split]
    pi = analytic_action_probability(full, action, posterior, tolerance=.01)
    gamma = {"analytic_pi_g1": 1., "analytic_pi_g2": 2.,
             "analytic_pi_g3": 3.}[controller]
    return pi**gamma


def shrink(full: np.ndarray, action: np.ndarray, gate: np.ndarray, strength: float):
    alpha = np.clip(float(strength)*gate, 0, 1)
    return (1-alpha[:, None])*full+alpha[:, None]*action, alpha


def metrics(probability: np.ndarray, full: np.ndarray, labels: np.ndarray,
            coalition_losses: np.ndarray, alpha: np.ndarray) -> dict[str, float]:
    value, prediction = probability_metrics(probability, labels)
    baseline, full_prediction = probability_metrics(full, labels)
    loss = observed_loss(probability, labels); full_loss = observed_loss(full, labels)
    positive_harm = np.maximum(loss-full_loss, 0)
    clipped = np.minimum(positive_harm, 1)
    oracle_loss = coalition_losses.min(1)
    denominator = max(float(np.mean(full_loss-oracle_loss)), 1e-12)
    return {
        **value,
        "accuracy_gain": value["accuracy"]-baseline["accuracy"],
        "macro_f1_gain": value["macro_f1"]-baseline["macro_f1"],
        "nll_gain": baseline["nll"]-value["nll"],
        "negative_flip_rate": float(((full_prediction == labels) &
                                      (prediction != labels)).mean()),
        "correction_rate": float(((full_prediction != labels) &
                                   (prediction == labels)).mean()),
        "clipped_harm": float(clipped.mean()),
        "harm_p95": float(np.quantile(positive_harm, .95)),
        "harmful_action_rate": float((positive_harm > 0).mean()),
        "regret_reduction": float(1-np.mean(loss-oracle_loss)/denominator),
        "mean_alpha": float(alpha.mean()),
        "activation_rate": float((alpha > 1e-8).mean()),
    }


def analytic_projection(posterior: np.ndarray, coalition_probability: np.ndarray,
                        top_k: int) -> np.ndarray:
    full = np.clip(coalition_probability[:, -1], 1e-6, 1)
    delta = np.clip(np.log(np.clip(coalition_probability, 1e-6, 1))-
                    np.log(full[:, None, :]), -1, 1)
    gain = (posterior[:, None, :]*np.maximum(delta, 0)).sum(2)
    harm = (posterior[:, None, :]*np.maximum(-delta, 0)).sum(2)
    return build_projection(posterior, coalition_probability, gain, harm,
                            top_k=top_k, alpha=1.)["projected_probability"]


def run_dataset(dataset: str, device: str):
    cfg = CONFIGS[dataset]; folds, names = load_dataset(dataset, ROOT/"data")
    masks = nonempty_coalitions(len(names)); top_k = 3 if len(names) >= 3 else 2
    e5_root = ROOT/"runs"/f"formal-e5-{dataset}"
    root = ROOT/"runs"/f"formal-i3-2-{dataset}"; root.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_index, splits in enumerate(folds):
        for task_seed in SEEDS:
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
                value = predict_router(router, splits[split_name],
                                       bundles[split_name]["probabilities"], masks,
                                       device, posterior[split_name])
                route[split_name] = apply_residual_blend(value, float(router_saved["blend"]))
            actions = {split_name: make_actions(
                bundles[split_name]["probabilities"], posterior[split_name],
                route[split_name], top_k)[0] for split_name in ("selection", "test")}
            mixer_saved = torch.load(e5_out/"anchored_harm_oracle.pt",
                                     map_location=device, weights_only=True)
            mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
            mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
            mixtures = {split_name: predict_mixer(
                mixer, actions[split_name], posterior[split_name], device)["probability"]
                for split_name in ("selection", "test")}
            projections = {split_name: analytic_projection(
                posterior[split_name], bundles[split_name]["probabilities"], top_k)
                for split_name in ("selection", "test")}

            controllers = {name: None for name in CORE_CONTROLLERS}
            if dataset != "avmnist":
                for method in NATIVE_METHODS:
                    native = native_gate(dataset, method, fold_index, task_seed, splits,
                                         dims, classes, device)
                    if native is not None:
                        controllers[f"native_{method}"] = native

            action_families = {"candidate_mixer": mixtures,
                               "convex_projection": projections}
            for action_name, action in action_families.items():
                allowed = controllers if action_name == "candidate_mixer" else {
                    "fixed": None, "analytic_pi_g2": None}
                for controller, native in allowed.items():
                    gates = {split_name: gate_values(
                        controller, actions[split_name][:, 0], action[split_name],
                        posterior[split_name], native=native, split=split_name)
                        for split_name in ("selection", "test")}
                    for strength in STRENGTHS:
                        split_metrics = {}
                        for split_name in ("selection", "test"):
                            probability, alpha = shrink(actions[split_name][:, 0],
                                                        action[split_name],
                                                        gates[split_name], strength)
                            split_metrics[split_name] = metrics(
                                probability, actions[split_name][:, 0],
                                splits[split_name]["y"], bundles[split_name]["losses"], alpha)
                        row = {"dataset": dataset, "fold": fold_index,
                               "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                               "action": action_name, "controller": controller,
                               "strength": strength}
                        row.update({f"selection_{key}": value for key, value in
                                    split_metrics["selection"].items()})
                        row.update({f"test_{key}": value for key, value in
                                    split_metrics["test"].items()})
                        rows.append(row)
            print(f"{dataset} fold={fold_index} seed={task_seed}: I3-2 curves done", flush=True)

    frame = pd.DataFrame(rows); frame.to_csv(root/"curves_by_fold_seed.csv", index=False)
    (root/"manifest.json").write_text(json.dumps({
        "experiment": "I3-2 matched harm-budget risk-benefit",
        "dataset": dataset, "method_version": "rcg-fusion-a8-v1",
        "seeds": list(SEEDS), "strength_grid": list(STRENGTHS),
        "budgets": list(BUDGETS), "shared_action": "candidate_mixer",
        "native_controller_adapter": "selection empirical CDF of negative mean native reliability",
        "selection_role": "choose strength independently at each harm budget",
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
