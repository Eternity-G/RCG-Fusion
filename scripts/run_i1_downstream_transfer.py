"""I1-2: propagate each coalition-supervision variant through one fixed decision chain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import predict_mixer
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import (apply_residual_blend, predict_router,
                                          select_residual_blend, train_router)
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                     predict_outputs, seed_all)

from run_e2_mosi_supervision_ablation import (CONFIGS, SEEDS, VARIANTS,
                                               candidate_metrics, make_target)
from run_e5_candidate_mixer_ablation import (make_actions, train_mixer)
from run_e6_adaptive_shrinkage import (choose_strength, evaluate, gate_values, shrink)


ROOT = Path(__file__).resolve().parents[1]
MIXER_CONFIG = {"harm_weight": 2.0, "oracle_weight": 0.05}
CONTROLLER = "analytic_pi_g2"


def weighted_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    ids = {"dataset", "variant", "fold", "train_seed", "n_test"}
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns
               if column not in ids]
    rows = []
    for (variant, seed), values in frame.groupby(["variant", "train_seed"]):
        weight = values.n_test.to_numpy(float)
        row = {"dataset": values.dataset.iloc[0], "variant": variant,
               "train_seed": int(seed), "n_test": int(weight.sum())}
        row.update({column: float(np.average(values[column], weights=weight))
                    for column in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset(dataset: str, device: str):
    cfg = CONFIGS[dataset]
    folds, modality_names = load_dataset(dataset, ROOT/"data")
    masks = nonempty_coalitions(len(modality_names))
    top_k = 3 if len(modality_names) >= 3 else 2
    output_root = ROOT/"runs"/f"formal-i1-2-{dataset}"
    output_root.mkdir(parents=True, exist_ok=True)
    partial = output_root/"metrics_by_fold_seed.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    completed = {(str(row["variant"]), int(row["fold"]), int(row["train_seed"]))
                 for row in rows}

    for fold_index, splits in enumerate(folds):
        oof_path = (Path(cfg["oof_template"].format(fold=fold_index))
                    if "oof_template" in cfg else cfg["oof"])
        with np.load(oof_path) as saved:
            oof = {key: saved[key] for key in saved.files}
        teacher_seed_to_index = {int(seed): index for index, seed in enumerate(oof["teacher_seeds"])}

        for task_seed in SEEDS:
            missing = [name for name in VARIANTS
                       if (name, fold_index, task_seed) not in completed]
            if not missing:
                continue
            backbone, temperatures, dims, classes = _load_backbone(
                cfg["base"]/f"fold_{fold_index}/seed_{task_seed}/backbone.pt", device)
            bundles = {}
            for split_name in ("train", "selection", "test"):
                raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])

            posterior = {"train_oof": []}
            for teacher_probability in oof["probabilities"]:
                _, value = load_posteriors(
                    cfg["posterior"], task_seed, dims, classes, masks, splits["train"],
                    teacher_probability, device, fold_index=fold_index)
                posterior["train_oof"].append(value)
            posterior["train_oof"] = np.stack(posterior["train_oof"])
            for split_name in ("train", "selection", "test"):
                _, posterior[split_name] = load_posteriors(
                    cfg["posterior"], task_seed, dims, classes, masks,
                    splits[split_name], bundles[split_name]["probabilities"],
                    device, fold_index=fold_index)

            decision_cache = {}
            for variant in missing:
                config = VARIANTS[variant]
                variant_root = output_root/f"fold_{fold_index}"/f"seed_{task_seed}"/variant
                variant_root.mkdir(parents=True, exist_ok=True)
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

                target = make_target(train_losses, config)
                target["posterior_override"] = train_posterior
                selection_target = make_target(bundles["selection"]["losses"][None], config)
                selection_target["posterior_override"] = posterior["selection"]
                seed = task_seed+fold_index*10000
                seed_all(seed)
                router = AnalyticResidualListwiseRouter(
                    dims, classes, len(masks), residual_scale=.5).to(device)
                history = train_router(
                    router, splits["train"], train_probability, target,
                    splits["selection"], bundles["selection"]["probabilities"],
                    selection_target, masks, device, seed=seed, epochs=100,
                    batch_size=cfg["batch_size"], patience=10,
                    objective_kwargs={"pair_weight": config["pair_weight"],
                                      "stable_weight": config["stable_weight"]})
                selection_route = predict_router(
                    router, splits["selection"], bundles["selection"]["probabilities"],
                    masks, device, posterior["selection"])
                blend, _ = select_residual_blend(selection_route, bundles["selection"]["losses"])

                routes, actions = {}, {}
                for split_name in ("selection", "test"):
                    route = predict_router(
                        router, splits[split_name], bundles[split_name]["probabilities"],
                        masks, device, posterior[split_name])
                    routes[split_name] = apply_residual_blend(route, blend)
                    actions[split_name], _ = make_actions(
                        bundles[split_name]["probabilities"], posterior[split_name],
                        routes[split_name], top_k)

                # The mixer always sees the same five OOF contexts.  Only the
                # router supervision above differs between variants.
                train_actions, train_posteriors, train_labels = [], [], []
                for teacher_index, teacher_probability in enumerate(oof["probabilities"]):
                    route = predict_router(
                        router, splits["train"], teacher_probability, masks, device,
                        posterior["train_oof"][teacher_index])
                    route = apply_residual_blend(route, blend)
                    current, _ = make_actions(
                        teacher_probability, posterior["train_oof"][teacher_index],
                        route, top_k)
                    train_actions.append(current)
                    train_posteriors.append(posterior["train_oof"][teacher_index])
                    train_labels.append(splits["train"]["y"])
                train_actions = np.concatenate(train_actions)
                train_posteriors = np.concatenate(train_posteriors)
                train_labels = np.concatenate(train_labels)

                digest = hashlib.sha256()
                for value in (train_actions, actions["selection"], actions["test"]):
                    digest.update(np.ascontiguousarray(value).view(np.uint8))
                action_key = digest.hexdigest()
                if action_key in decision_cache:
                    cached = decision_cache[action_key]
                    (mixer_epochs, mixer_state, selected, raw_metric, adaptive_metric,
                     losses, prediction, alpha) = cached
                else:
                    seed_all(seed)
                    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
                    mixer_history = train_mixer(
                        mixer, train_actions, train_posteriors, train_labels,
                        actions["selection"], posterior["selection"], splits["selection"]["y"],
                        device, seed=seed, batch_size=cfg["batch_size"],
                        **MIXER_CONFIG)
                    mixture = {split_name: predict_mixer(
                        mixer, actions[split_name], posterior[split_name], device)
                        for split_name in ("selection", "test")}
                    selected, _ = choose_strength(
                        actions["selection"][:, 0], mixture["selection"]["probability"],
                        posterior["selection"], splits["selection"]["y"], CONTROLLER)
                    gate = gate_values(
                        CONTROLLER, actions["test"][:, 0], mixture["test"]["probability"],
                        posterior["test"])
                    adaptive, alpha = shrink(
                        actions["test"][:, 0], mixture["test"]["probability"], gate,
                        selected["strength"])
                    raw_metric, _, _ = evaluate(
                        mixture["test"]["probability"], actions["test"][:, 0],
                        splits["test"]["y"], bundles["test"]["losses"], np.ones(len(alpha)))
                    adaptive_metric, losses, prediction = evaluate(
                        adaptive, actions["test"][:, 0], splits["test"]["y"],
                        bundles["test"]["losses"], alpha)
                    mixer_epochs = len(mixer_history)
                    mixer_state = {key: value.detach().cpu().clone()
                                   for key, value in mixer.state_dict().items()}
                    decision_cache[action_key] = (
                        mixer_epochs, mixer_state, selected, raw_metric, adaptive_metric,
                        losses, prediction, alpha)
                candidate_k, candidate_coverage, candidate_nll = candidate_metrics(
                    routes["test"], bundles["test"]["losses"])
                rows.append({
                    "dataset": dataset, "variant": variant, "fold": fold_index,
                    "train_seed": task_seed, "n_test": len(splits["test"]["y"]),
                    "router_epochs": len(history), "mixer_epochs": mixer_epochs,
                    "residual_blend": blend, "shrinkage_strength": selected["strength"],
                    "candidate_k": candidate_k,
                    "candidate_coverage": candidate_coverage,
                    "candidate_oracle_nll": candidate_nll,
                    **{f"raw_{key}": value for key, value in raw_metric.items()},
                    **{f"adaptive_{key}": value for key, value in adaptive_metric.items()},
                })
                torch.save({"state_dict": router.state_dict(), "blend": blend},
                           variant_root/"router.pt")
                torch.save({"state_dict": mixer_state, "classes": classes},
                           variant_root/"mixer.pt")
                prediction_frame = pd.DataFrame({
                    "sample_id": splits["test"]["id"],
                    "group_or_video_id": splits["test"]["group"],
                    "label": splits["test"]["y"], "loss": losses,
                    "prediction": prediction, "alpha": alpha})
                prediction_frame.to_parquet(variant_root/"predictions.parquet", index=False)
                pd.DataFrame(rows).to_csv(partial, index=False)
                print(f"{dataset} fold={fold_index} seed={task_seed} {variant}: "
                      f"raw={raw_metric['nll_gain']:.5f} "
                      f"adaptive={adaptive_metric['nll_gain']:.5f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root/"metrics_by_fold_seed.csv", index=False)
    by_seed = weighted_seed_metrics(frame)
    by_seed.to_csv(output_root/"metrics_by_seed.csv", index=False)
    summary = by_seed.groupby("variant").agg(
        adaptive_nll_gain_mean=("adaptive_nll_gain", "mean"),
        adaptive_nll_gain_std=("adaptive_nll_gain", "std"),
        adaptive_accuracy_gain_mean=("adaptive_accuracy_gain", "mean"),
        adaptive_correction_rate_mean=("adaptive_correction_rate", "mean"),
        adaptive_negative_flip_rate_mean=("adaptive_negative_flip_rate", "mean"),
        adaptive_clipped_harm_mean=("adaptive_clipped_harm", "mean"),
        adaptive_regret_reduction_mean=("adaptive_regret_reduction", "mean"),
    ).reset_index()
    summary.to_csv(output_root/"metrics_summary.csv", index=False)
    (output_root/"manifest.json").write_text(json.dumps({
        "experiment": "I1-2 soft supervision downstream transfer",
        "dataset": dataset, "variants": VARIANTS, "seeds": list(SEEDS),
        "fixed_mixer": MIXER_CONFIG, "fixed_controller": CONTROLLER,
        "selection_role": "early stopping, router blend, and shrinkage strength only",
        "test_labels_role": "evaluation only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="mosi")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    datasets = tuple(CONFIGS) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
