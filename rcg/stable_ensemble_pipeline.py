"""Evaluate the stable full/posterior convex ensemble."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .io import write_json
from .posterior_analytic_pipeline import _load_backbone
from .posterior_shrinkage_pipeline import probability_metrics
from .projected_fusion_pipeline import load_posteriors
from .rcg_fusion import nonempty_coalitions
from .rcg_fusion_pipeline import load_dataset, predict_outputs
from .stable_ensemble import (fit_simplex_weights, mix_actions,
                              select_safe_shrinkage)


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data); masks = nonempty_coalitions(len(names))
    base, posterior, root = Path(args.base_run), Path(args.posterior_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=True); all_rows = []
    for fold_index, splits in enumerate(folds):
        actions = {"selection": [], "test": []}; action_names = []
        per_seed_full = []
        for seed in args.seeds:
            model, temperatures, dims, classes = _load_backbone(
                base/f"fold_{fold_index}"/f"seed_{seed}"/"backbone.pt", device)
            for split_name in actions:
                probability = predict_outputs(
                    model, splits[split_name]["x"], device, temperatures)["probabilities"]
                _, q = load_posteriors(
                    posterior, seed, dims, classes, masks, splits[split_name], probability,
                    device, fold_index=fold_index)
                actions[split_name].extend((probability[:, -1], q))
                if split_name == "test":
                    per_seed_full.append(probability[:, -1])
            action_names.extend((f"seed_{seed}_full", f"seed_{seed}_posterior"))
        selection_actions = np.asarray(actions["selection"])
        test_actions = np.asarray(actions["test"])
        weight = fit_simplex_weights(selection_actions, splits["selection"]["y"], args.l2)
        selection_candidate = mix_actions(selection_actions, weight)
        selection_full = np.mean(selection_actions[::2], axis=0)
        rho, rho_grid = select_safe_shrinkage(
            selection_full, selection_candidate, splits["selection"]["y"])
        candidate = mix_actions(test_actions, weight)
        full_ensemble = np.mean(per_seed_full, axis=0)
        final = (1-rho)*full_ensemble+rho*candidate
        labels = splits["test"]["y"]
        final_metrics, _ = probability_metrics(final, labels)
        ensemble_metrics, _ = probability_metrics(full_ensemble, labels)
        single_metrics = [probability_metrics(value, labels)[0] for value in per_seed_full]
        mean_single_accuracy = float(np.mean([x["accuracy"] for x in single_metrics]))
        mean_single_nll = float(np.mean([x["nll"] for x in single_metrics]))
        item = {"dataset": args.dataset, "fold": fold_index, "n_test": len(labels),
                "members": len(args.seeds),
                "safe_shrinkage_rho": rho,
                **{f"final_{k}": v for k, v in final_metrics.items()},
                **{f"full_ensemble_{k}": v for k, v in ensemble_metrics.items()},
                "mean_single_accuracy": mean_single_accuracy,
                "mean_single_nll": mean_single_nll,
                "relative_error_reduction_vs_mean_single": float(
                    (final_metrics["accuracy"]-mean_single_accuracy)/max(1-mean_single_accuracy, 1e-12)),
                "relative_nll_reduction_vs_mean_single": float(
                    (mean_single_nll-final_metrics["nll"])/mean_single_nll),
                "relative_error_reduction_vs_full_ensemble": float(
                    (final_metrics["accuracy"]-ensemble_metrics["accuracy"])/max(1-ensemble_metrics["accuracy"], 1e-12)),
                "relative_nll_reduction_vs_full_ensemble": float(
                    (ensemble_metrics["nll"]-final_metrics["nll"])/ensemble_metrics["nll"])}
        all_rows.append(item)
        out = root/f"fold_{fold_index}"; out.mkdir(exist_ok=True)
        write_json(out/"metrics.json", item)
        write_json(out/"weights.json", {name: float(value) for name, value in zip(action_names, weight)})
        write_json(out/"safe_shrinkage.json", {"selected_rho": rho,
                                                 "selection_grid": rho_grid})
        frame = pd.DataFrame({"sample_id": splits["test"]["id"],
                              "group_or_video_id": splits["test"]["group"],
                              "label": labels})
        for k in range(final.shape[1]):
            frame[f"full_ensemble_p{k}"] = full_ensemble[:, k]
            frame[f"final_p{k}"] = final[:, k]
        frame.to_parquet(out/"predictions.parquet", index=False)
        print(f"{args.dataset} fold={fold_index}: final_acc={final_metrics['accuracy']:.4f} "
              f"final_nll={final_metrics['nll']:.4f} error_reduction="
              f"{item['relative_error_reduction_vs_mean_single']:.3f}", flush=True)
    pd.DataFrame(all_rows).to_csv(root/"metrics.csv", index=False)
    mean = pd.DataFrame(all_rows).mean(numeric_only=True)
    checks = {"system_error_reduction_10pct": mean.relative_error_reduction_vs_mean_single >= .10,
              "beats_compute_matched_full_ensemble_nll":
                  mean.relative_nll_reduction_vs_full_ensemble > 0,
              "beats_compute_matched_full_ensemble_accuracy":
                  mean.relative_error_reduction_vs_full_ensemble > 0}
    write_json(root/"complete.json", {"checks": checks, "passed": all(checks.values()),
                                      "comparison_note": "10% is versus mean single model; full ensemble is the compute-matched baseline"})


def main():
    parser = argparse.ArgumentParser(description="Stable full/posterior ensemble")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), required=True)
    parser.add_argument("--data", default="data"); parser.add_argument("--base-run", required=True)
    parser.add_argument("--posterior-run", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--l2", type=float, default=0.); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
