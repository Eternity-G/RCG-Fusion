"""E7: isolate model-level aggregation and full-coalition fallback.

All meta-models are fitted on the selection split. Test labels are only used
after the aggregation rules have been frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import load_dataset, predict_outputs
from rcg.stable_ensemble import fit_simplex_weights, mix_actions, select_safe_shrinkage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402

GROUPS = {
    "mosi": {
        "G1": (CONFIGS["mosi"]["base"], CONFIGS["mosi"]["posterior"], SEEDS),
        "G2": (ROOT / "runs/formal-e1-mosi/g2/base", ROOT / "runs/formal-e1-mosi/g2/posterior", (66, 77, 88, 99, 110)),
        "G3": (ROOT / "runs/formal-e1-mosi/g3/base", ROOT / "runs/formal-e1-mosi/g3/posterior", (121, 132, 143, 154, 165)),
        "G4": (ROOT / "runs/formal-e1-mosi/g4/base", ROOT / "runs/formal-e1-mosi/g4/posterior", (176, 187, 198, 209, 220)),
        "G5": (ROOT / "runs/formal-e1-mosi/g5/base", ROOT / "runs/formal-e1-mosi/g5/posterior", (231, 242, 253, 264, 275)),
    },
    "mosei": {
        "G1": (CONFIGS["mosei"]["base"], CONFIGS["mosei"]["posterior"], SEEDS),
        "G2": (ROOT / "runs/formal-e1-mosei/g2/base", ROOT / "runs/formal-e1-mosei/g2/posterior", (66, 77, 88, 99, 110)),
        "G3": (ROOT / "runs/formal-e1-mosei/g3/base", ROOT / "runs/formal-e1-mosei/g3/posterior", (121, 132, 143, 154, 165)),
        "G4": (ROOT / "runs/formal-e1-mosei/g4/base", ROOT / "runs/formal-e1-mosei/g4/posterior", (176, 187, 198, 209, 220)),
        "G5": (ROOT / "runs/formal-e1-mosei/g5/base", ROOT / "runs/formal-e1-mosei/g5/posterior", (231, 242, 253, 264, 275)),
    },
    "cremad": {"G1": (CONFIGS["cremad"]["base"], CONFIGS["cremad"]["posterior"], SEEDS)},
    "avmnist": {"G1": (CONFIGS["avmnist"]["base"], CONFIGS["avmnist"]["posterior"], SEEDS)},
}


def normalize(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, None)
    return probability / probability.sum(axis=1, keepdims=True)


def logit_features(actions: np.ndarray) -> np.ndarray:
    """Concatenate member log probabilities for standard logistic stacking."""
    return np.concatenate([np.log(np.clip(action, 1e-8, 1.0)) for action in actions], axis=1)


def fit_logit_stacking(selection_actions: np.ndarray, labels: np.ndarray):
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    model.fit(logit_features(selection_actions), labels)
    return model


def predict_logit_stacking(model, actions: np.ndarray) -> np.ndarray:
    return normalize(model.predict_proba(logit_features(actions)))


def evaluate(probability: np.ndarray, labels: np.ndarray, full: np.ndarray) -> dict:
    metric, prediction = probability_metrics(probability, labels)
    full_metric, full_prediction = probability_metrics(full, labels)
    rows = np.arange(len(labels))
    loss = -np.log(np.clip(probability[rows, labels], 1e-12, 1))
    full_loss = -np.log(np.clip(full[rows, labels], 1e-12, 1))
    return {
        **metric,
        "accuracy_gain": metric["accuracy"] - full_metric["accuracy"],
        "nll_gain": full_metric["nll"] - metric["nll"],
        "relative_error_reduction": (
            metric["accuracy"] - full_metric["accuracy"]
        ) / max(1 - full_metric["accuracy"], 1e-12),
        "relative_nll_reduction": (full_metric["nll"] - metric["nll"]) / full_metric["nll"],
        "negative_flip_rate": float(((full_prediction == labels) & (prediction != labels)).mean()),
        "correction_rate": float(((full_prediction != labels) & (prediction == labels)).mean()),
        "clipped_harm": float(np.minimum(np.maximum(loss - full_loss, 0), 1).mean()),
    }


def member_actions(dataset, fold_index, splits, base, posterior, seeds, masks, device):
    result = {"selection": [], "test": []}
    names = []
    for seed in seeds:
        model, temperatures, dims, classes = _load_backbone(
            base / f"fold_{fold_index}" / f"seed_{seed}" / "backbone.pt", device)
        for split_name in result:
            probability = predict_outputs(
                model, splits[split_name]["x"], device, temperatures)["probabilities"]
            _, q = load_posteriors(
                posterior, seed, dims, classes, masks, splits[split_name], probability,
                device, fold_index=fold_index)
            result[split_name].extend((probability[:, -1], q))
        names.extend((f"seed_{seed}_full", f"seed_{seed}_posterior"))
    return {key: np.asarray(value) for key, value in result.items()}, names


def aggregate_folds(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns
               if column not in {"fold", "n_test"}]
    for (dataset, group, method), values in frame.groupby(["dataset", "group", "method"]):
        weights = values.n_test.to_numpy(float)
        row = {"dataset": dataset, "group": group, "method": method,
               "n_test": int(weights.sum())}
        row.update({metric: float(np.average(values[metric], weights=weights))
                    for metric in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset(dataset: str, device: str) -> None:
    folds, modality_names = load_dataset(dataset, ROOT / "data")
    masks = nonempty_coalitions(len(modality_names))
    output = ROOT / "runs" / f"formal-e7-{dataset}"
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows, weight_rows, rho_rows = [], [], [], []
    for group, (base, posterior, seeds) in GROUPS[dataset].items():
        for fold_index, splits in enumerate(folds):
            actions, action_names = member_actions(
                dataset, fold_index, splits, Path(base), Path(posterior), seeds, masks, device)
            selection_y, test_y = splits["selection"]["y"], splits["test"]["y"]
            selection_full = actions["selection"][::2].mean(0)
            test_full = actions["test"][::2].mean(0)
            selection_posterior = actions["selection"][1::2].mean(0)
            test_posterior = actions["test"][1::2].mean(0)

            logit_model = fit_logit_stacking(actions["selection"], selection_y)
            test_logit = predict_logit_stacking(logit_model, actions["test"])
            selection_logit = predict_logit_stacking(logit_model, actions["selection"])

            simplex = fit_simplex_weights(actions["selection"], selection_y, l2=1e-3)
            selection_convex = mix_actions(actions["selection"], simplex)
            test_convex = mix_actions(actions["test"], simplex)
            rho, grid = select_safe_shrinkage(selection_full, selection_convex, selection_y)
            selection_safe = normalize((1-rho)*selection_full + rho*selection_convex)
            test_safe = normalize((1-rho)*test_full + rho*test_convex)

            methods = {
                "full_ensemble": (selection_full, test_full),
                "posterior_ensemble": (selection_posterior, test_posterior),
                "equal_full_posterior": (actions["selection"].mean(0), actions["test"].mean(0)),
                "logit_stacking": (selection_logit, test_logit),
                "simplex_convex": (selection_convex, test_convex),
                "simplex_safe_fallback": (selection_safe, test_safe),
            }
            for method, (selection_probability, test_probability) in methods.items():
                row = {"dataset": dataset, "group": group, "fold": fold_index,
                       "method": method, "n_test": len(test_y),
                       "selection_accuracy": float((selection_probability.argmax(1) == selection_y).mean()),
                       "selection_nll": float(-np.log(np.clip(
                           selection_probability[np.arange(len(selection_y)), selection_y], 1e-12, 1)).mean()),
                       **evaluate(test_probability, test_y, test_full)}
                metric_rows.append(row)
                frame = pd.DataFrame({"dataset": dataset, "group": group, "fold": fold_index,
                                      "method": method, "sample_id": splits["test"]["id"],
                                      "group_or_video_id": splits["test"]["group"], "label": test_y})
                for k in range(test_probability.shape[1]): frame[f"p{k}"] = test_probability[:, k]
                prediction_rows.append(frame)
            for name, value in zip(action_names, simplex):
                weight_rows.append({"dataset": dataset, "group": group, "fold": fold_index,
                                    "action": name, "weight": float(value)})
            rho_rows.append({"dataset": dataset, "group": group, "fold": fold_index,
                             "rho": float(rho), "selection_grid": json.dumps(grid)})
            print(f"{dataset} {group} fold={fold_index}: rho={rho:.1f} "
                  f"full_nll={evaluate(test_full, test_y, test_full)['nll']:.4f} "
                  f"safe_nll={evaluate(test_safe, test_y, test_full)['nll']:.4f}", flush=True)

    fold_metrics = pd.DataFrame(metric_rows)
    fold_metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    aggregate_folds(fold_metrics).to_csv(output / "metrics_by_group.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "simplex_weights.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(output / "safe_fallback.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "experiment": "E7 model-level stable aggregation",
        "dataset": dataset, "groups": {g: list(map(int, v[2])) for g, v in GROUPS[dataset].items()},
        "methods": ["full_ensemble", "posterior_ensemble", "equal_full_posterior",
                    "logit_stacking", "simplex_convex", "simplex_safe_fallback"],
        "simplex_l2": 1e-3, "fallback_grid": [round(x, 1) for x in np.linspace(0, 1, 11)],
        "selection_role": "fit stacking, simplex weights, and safe fallback rho",
        "test_labels_role": "evaluation only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*GROUPS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = GROUPS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets: run_dataset(dataset, device)


if __name__ == "__main__":
    main()
