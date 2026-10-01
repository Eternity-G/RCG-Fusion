"""Run the MOSI E2 supervision ablation with fixed backbone and posterior models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.listwise_router import (AnalyticResidualListwiseRouter,
                                 anchor_preserving_candidates,
                                 build_soft_coalition_targets)
from rcg.listwise_router_pipeline import (apply_residual_blend, evaluate,
                                          predict_router, select_residual_blend,
                                          train_router)
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                     predict_outputs, seed_all)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/rcg-fusion-mosi-v4"
POSTERIOR = ROOT / "runs/rcg-posterior-analytic-mosi-v2"
OUTPUT = ROOT / "runs/formal-e2-mosi"
SEEDS = (11, 22, 33, 44, 55)

VARIANTS = {
    "in_sample_hard": {"source": "in_sample", "teachers": 1, "hard": True,
                       "agreement": False, "pair_weight": 0.0, "stable_weight": 0.0},
    "oof_single_hard": {"source": "oof", "teachers": 1, "hard": True,
                        "agreement": False, "pair_weight": 0.0, "stable_weight": 0.0},
    "oof_multi_hard": {"source": "oof", "teachers": 5, "hard": True,
                       "agreement": False, "pair_weight": 0.0, "stable_weight": 0.0},
    "oof_multi_soft": {"source": "oof", "teachers": 5, "hard": False,
                       "agreement": False, "pair_weight": 0.0, "stable_weight": 0.0},
    "oof_soft_pair": {"source": "oof", "teachers": 5, "hard": False,
                      "agreement": False, "pair_weight": 0.5, "stable_weight": 0.0},
    "oof_soft_full": {"source": "oof", "teachers": 5, "hard": False,
                      "agreement": True, "pair_weight": 0.5, "stable_weight": 0.2},
}


def make_target(losses: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    target = build_soft_coalition_targets(
        losses, oracle_temperature=0.1, epsilon=0.01, stable_fraction=0.8
    )
    if config["hard"]:
        oracle = losses.mean(0).argmin(1)
        target["soft_oracle"] = np.eye(losses.shape[2], dtype=np.float32)[oracle]
    if not config["agreement"]:
        target["agreement_weight"] = np.ones(losses.shape[1], dtype=np.float32)
    return target


def top3_metrics(output: dict, losses: np.ndarray) -> tuple[float, float]:
    candidates = anchor_preserving_candidates(
        output["analytic_score"], output["score"], top_k=3
    )
    oracle = losses.argmin(1)
    coverage = np.mean([
        oracle[row] in candidates[row] for row in range(len(oracle))
    ])
    candidate_loss = np.take_along_axis(losses, candidates, axis=1).min(1).mean()
    return float(coverage), float(candidate_loss)


def run(device: str = "auto") -> None:
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset("mosi", ROOT / "data")
    if len(folds) != 1:
        raise ValueError("E2 MOSI expects one standard outer split")
    splits = folds[0]
    masks = nonempty_coalitions(len(names))
    with np.load(BASE / "fold_0/oof_targets.npz") as saved:
        oof = {key: saved[key] for key in saved.files}
    teacher_seed_to_index = {
        int(seed): index for index, seed in enumerate(oof["teacher_seeds"])
    }
    rows = []
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for task_seed in SEEDS:
        backbone, temperatures, dims, classes = _load_backbone(
            BASE / f"fold_0/seed_{task_seed}/backbone.pt", device
        )
        bundles = {}
        for split_name in ("train", "selection", "test"):
            raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
            bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])

        posterior = {"train_oof": []}
        for teacher_probability in oof["probabilities"]:
            _, value = load_posteriors(
                POSTERIOR, task_seed, dims, classes, masks, splits["train"],
                teacher_probability, device, fold_index=0,
            )
            posterior["train_oof"].append(value)
        posterior["train_oof"] = np.stack(posterior["train_oof"])
        for split_name in ("train", "selection", "test"):
            _, posterior[split_name] = load_posteriors(
                POSTERIOR, task_seed, dims, classes, masks, splits[split_name],
                bundles[split_name]["probabilities"], device, fold_index=0,
            )

        for variant, config in VARIANTS.items():
            if config["source"] == "in_sample":
                train_probability = bundles["train"]["probabilities"][None]
                train_losses = bundles["train"]["losses"][None]
                train_posterior = posterior["train"][None]
            else:
                indices = list(range(config["teachers"]))
                if config["teachers"] == 1:
                    indices = [teacher_seed_to_index[task_seed]]
                train_probability = oof["probabilities"][indices]
                train_losses = oof["losses"][indices]
                train_posterior = posterior["train_oof"][indices]

            train_target = make_target(train_losses, config)
            train_target["posterior_override"] = train_posterior
            selection_target = make_target(
                bundles["selection"]["losses"][None], config
            )
            selection_target["posterior_override"] = posterior["selection"]

            seed_all(task_seed)
            model = AnalyticResidualListwiseRouter(
                dims, classes, len(masks), residual_scale=0.5
            ).to(device)
            history = train_router(
                model, splits["train"], train_probability, train_target,
                splits["selection"], bundles["selection"]["probabilities"],
                selection_target, masks, device, seed=task_seed,
                epochs=100, batch_size=64, patience=10,
                objective_kwargs={
                    "pair_weight": config["pair_weight"],
                    "stable_weight": config["stable_weight"],
                },
            )
            selection_output = predict_router(
                model, splits["selection"], bundles["selection"]["probabilities"],
                masks, device, posterior["selection"],
            )
            blend, _ = select_residual_blend(
                selection_output, bundles["selection"]["losses"]
            )
            output = predict_router(
                model, splits["test"], bundles["test"]["probabilities"],
                masks, device, posterior["test"],
            )
            output = apply_residual_blend(output, blend)
            result, _, _ = evaluate(
                output, bundles["test"], splits["test"]["y"]
            )
            top3, candidate_nll = top3_metrics(output, bundles["test"]["losses"])
            rows.append({
                "variant": variant,
                "train_seed": task_seed,
                "epochs": len(history),
                "residual_blend": blend,
                "router_ndcg": result["router_ndcg"],
                "router_pairwise_accuracy": result["router_pairwise_accuracy"],
                "router_top1": result["router_top1"],
                "router_top2": result["router_top2"],
                "anchored_top3": top3,
                "candidate_oracle_nll": candidate_nll,
                "selected_nll": result["selected_nll"],
                "selection_regret": result["selection_regret"],
            })
            print(
                f"{variant} seed={task_seed} ndcg={result['router_ndcg']:.4f} "
                f"top3={top3:.4f} candidate_nll={candidate_nll:.4f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "metrics_by_seed.csv", index=False)
    metrics = [column for column in frame.columns if column not in {
        "variant", "train_seed", "epochs"
    }]
    summary = frame.groupby("variant")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(OUTPUT / "metrics_summary.csv", index=False)
    manifest = {
        "dataset": "MOSI",
        "seeds": list(SEEDS),
        "variants": VARIANTS,
        "base_run": str(BASE),
        "posterior_run": str(POSTERIOR),
        "device": device,
        "selection_role": "early stopping and residual blend only",
        "test_labels_role": "evaluation only",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args().device)


if __name__ == "__main__":
    main()
