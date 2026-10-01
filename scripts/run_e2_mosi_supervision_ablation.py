"""Run the E2 supervision ablation with fixed backbone and posterior models."""
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
SEEDS = (11, 22, 33, 44, 55)
CONFIGS = {
    "mosi": {
        "base": ROOT / "runs/rcg-fusion-mosi-v4",
        "posterior": ROOT / "runs/rcg-posterior-analytic-mosi-v2",
        "oof": ROOT / "runs/rcg-fusion-mosi-v4/fold_0/oof_targets.npz",
        "output": ROOT / "runs/formal-e2-mosi",
        "batch_size": 64,
    },
    "mosei": {
        "base": ROOT / "runs/rcg-fusion-mosei-shrinkage-v1",
        "posterior": ROOT / "runs/rcg-posterior-analytic-mosei-shrinkage-v1",
        "oof": ROOT / "runs/formal-e2-mosei-oof/oof_targets.npz",
        "output": ROOT / "runs/formal-e2-mosei",
        "batch_size": 128,
    },
    "cremad": {
        "base": ROOT / "runs/rcg-fusion-cremad-v1",
        "posterior": ROOT / "runs/rcg-posterior-analytic-cremad-v1",
        "oof_template": str(
            ROOT / "runs/formal-e2-cremad-oof/fold_{fold}/oof_targets.npz"
        ),
        "output": ROOT / "runs/formal-e2-cremad",
        "batch_size": 64,
    },
}

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


def candidate_metrics(output: dict, losses: np.ndarray) -> tuple[int, float, float]:
    top_k = min(3, losses.shape[1] - 1)
    candidates = anchor_preserving_candidates(
        output["analytic_score"], output["score"], top_k=top_k
    )
    oracle = losses.argmin(1)
    coverage = np.mean([
        oracle[row] in candidates[row] for row in range(len(oracle))
    ])
    candidate_loss = np.take_along_axis(losses, candidates, axis=1).min(1).mean()
    return top_k, float(coverage), float(candidate_loss)


def run(dataset: str, device: str = "auto") -> None:
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    device = "cpu" if device == "auto" else device
    dataset_config = CONFIGS[dataset]
    base, posterior_root, output_dir = (
        dataset_config["base"], dataset_config["posterior"], dataset_config["output"]
    )
    folds, names = load_dataset(dataset, ROOT / "data")
    masks = nonempty_coalitions(len(names))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "metrics_by_seed.partial.csv"
    if checkpoint.exists():
        saved_rows = pd.read_csv(checkpoint)
        rows = (saved_rows.to_dict("records")
                if "candidate_k" in saved_rows.columns else [])
    else:
        rows = []
    completed = {
        (row["variant"], int(row["train_seed"]), int(row.get("fold", 0)))
        for row in rows
    }

    for fold_index, splits in enumerate(folds):
        if "oof_template" in dataset_config:
            oof_path = Path(dataset_config["oof_template"].format(fold=fold_index))
        else:
            oof_path = dataset_config["oof"]
        with np.load(oof_path) as saved:
            oof = {key: saved[key] for key in saved.files}
        teacher_seed_to_index = {
            int(seed): index for index, seed in enumerate(oof["teacher_seeds"])
        }

        for task_seed in SEEDS:
            backbone, temperatures, dims, classes = _load_backbone(
                base / f"fold_{fold_index}/seed_{task_seed}/backbone.pt", device
            )
            bundles = {}
            for split_name in ("train", "selection", "test"):
                raw = predict_outputs(
                    backbone, splits[split_name]["x"], device, temperatures
                )
                bundles[split_name] = attach_observed_losses(
                    raw, splits[split_name]["y"]
                )

            posterior = {"train_oof": []}
            for teacher_probability in oof["probabilities"]:
                _, value = load_posteriors(
                    posterior_root, task_seed, dims, classes, masks, splits["train"],
                    teacher_probability, device, fold_index=fold_index,
                )
                posterior["train_oof"].append(value)
            posterior["train_oof"] = np.stack(posterior["train_oof"])
            for split_name in ("train", "selection", "test"):
                _, posterior[split_name] = load_posteriors(
                    posterior_root, task_seed, dims, classes, masks,
                    splits[split_name], bundles[split_name]["probabilities"],
                    device, fold_index=fold_index,
                )

            for variant, config in VARIANTS.items():
                if (variant, task_seed, fold_index) in completed:
                    continue
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

                seed_all(task_seed + fold_index * 10000)
                model = AnalyticResidualListwiseRouter(
                    dims, classes, len(masks), residual_scale=0.5
                ).to(device)
                history = train_router(
                    model, splits["train"], train_probability, train_target,
                    splits["selection"], bundles["selection"]["probabilities"],
                    selection_target, masks, device,
                    seed=task_seed + fold_index * 10000,
                    epochs=100, batch_size=dataset_config["batch_size"], patience=10,
                    objective_kwargs={
                        "pair_weight": config["pair_weight"],
                        "stable_weight": config["stable_weight"],
                    },
                )
                selection_output = predict_router(
                    model, splits["selection"],
                    bundles["selection"]["probabilities"], masks, device,
                    posterior["selection"],
                )
                blend, _ = select_residual_blend(
                    selection_output, bundles["selection"]["losses"]
                )
                route_output = predict_router(
                    model, splits["test"], bundles["test"]["probabilities"],
                    masks, device, posterior["test"],
                )
                route_output = apply_residual_blend(route_output, blend)
                result, _, _ = evaluate(
                    route_output, bundles["test"], splits["test"]["y"]
                )
                candidate_k, candidate_coverage, candidate_nll = candidate_metrics(
                    route_output, bundles["test"]["losses"]
                )
                rows.append({
                    "variant": variant, "fold": fold_index,
                    "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                    "epochs": len(history), "residual_blend": blend,
                    "router_ndcg": result["router_ndcg"],
                    "router_pairwise_accuracy": result["router_pairwise_accuracy"],
                    "router_top1": result["router_top1"],
                    "router_top2": result["router_top2"],
                    "candidate_k": candidate_k,
                    "anchored_candidate_coverage": candidate_coverage,
                    "anchored_top3": candidate_coverage,
                    "candidate_oracle_nll": candidate_nll,
                    "selected_nll": result["selected_nll"],
                    "selection_regret": result["selection_regret"],
                })
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                print(
                    f"fold={fold_index} {variant} seed={task_seed} "
                    f"ndcg={result['router_ndcg']:.4f} top{candidate_k}="
                    f"{candidate_coverage:.4f} "
                    f"candidate_nll={candidate_nll:.4f}", flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metrics_by_fold_seed.csv", index=False)
    metrics = [column for column in frame.columns if column not in {
        "variant", "fold", "train_seed", "n_test", "epochs", "candidate_k"
    }]
    seed_rows = []
    for (variant, seed), values in frame.groupby(["variant", "train_seed"]):
        weights = values.n_test.to_numpy(dtype=float)
        row = {"variant": variant, "train_seed": int(seed),
               "n_test": int(weights.sum())}
        row.update({
            metric: float(np.average(values[metric], weights=weights))
            for metric in metrics
        })
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary = seed_frame.groupby("variant")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "metrics_summary.csv", index=False)
    manifest = {
        "dataset": dataset,
        "seeds": list(SEEDS),
        "variants": VARIANTS,
        "base_run": str(base),
        "posterior_run": str(posterior_root),
        "oof_targets": dataset_config.get(
            "oof_template", str(dataset_config.get("oof"))
        ),
        "outer_folds": len(folds),
        "device": device,
        "selection_role": "early stopping and residual blend only",
        "test_labels_role": "evaluation only",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CONFIGS), default="mosi")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(args.dataset, args.device)


if __name__ == "__main__":
    main()
