"""Regenerate P0-v2 independent ensembles from the existing E1 seed groups.

The historical E1 checkpoints contain independent coalition backbones, OOF
teachers, and posterior estimators. This runner trains only the current
listwise router/mixer, applies the current A7 policy, and fits canonical A8-v2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from rcg.anchored_mixer import AnchoredCandidateMixer  # noqa: E402
from rcg.anchored_mixer_pipeline import predict_mixer  # noqa: E402
from rcg.final_system import METHOD_VERSION, fit_final_system, prediction_hash  # noqa: E402
from rcg.listwise_router import AnalyticResidualListwiseRouter  # noqa: E402
from rcg.listwise_router_pipeline import apply_residual_blend, predict_router  # noqa: E402
from rcg.posterior_analytic_pipeline import _load_backbone  # noqa: E402
from rcg.projected_fusion_pipeline import load_posteriors  # noqa: E402
from rcg.rcg_fusion import nonempty_coalitions  # noqa: E402
from rcg.rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs  # noqa: E402
import run_e4_anchored_routing as e4  # noqa: E402
import run_e5_candidate_mixer_ablation as e5  # noqa: E402
import run_e6_adaptive_shrinkage as e6  # noqa: E402

GROUPS = {
    "g2": (66, 77, 88, 99, 110),
    "g3": (121, 132, 143, 154, 165),
    "g4": (176, 187, 198, 209, 220),
    "g5": (231, 242, 253, 264, 275),
}


def configuration(dataset: str, group: str):
    root = ROOT/f"runs/formal-e1-{dataset}/{group}"
    config = {"base": root/"base", "posterior": root/"posterior",
              "batch_size": 64 if dataset in ("mosi", "cremad") else 128}
    # CREMA-D/AV-MNIST use the protocol-frozen multi-teacher OOF library for
    # every independent task-training repeat. This isolates optimization
    # randomness without silently changing the contribution target itself.
    if dataset == "cremad":
        config["oof_template"] = str(ROOT/"runs/formal-e2-cremad-oof/fold_{fold}/oof_targets.npz")
    elif dataset == "avmnist":
        config["oof"] = ROOT/"runs/formal-e2-avmnist-oof/oof_targets.npz"
    else:
        config["oof"] = root/"base/fold_0/oof_targets.npz"
    return config


def selected_strength(e6_root: Path, fold: int, seed: int) -> float:
    frame = pd.read_csv(e6_root/"metrics_by_fold_seed.csv")
    row = frame[(frame.variant == "analytic_pi_g2") &
                (frame.fold == fold) & (frame.train_seed == seed)]
    if len(row) != 1: raise ValueError(f"missing A7 strength fold={fold} seed={seed}")
    return float(row.iloc[0].strength)


def member_outputs(cfg, e5_root, e6_root, splits, masks, fold, seed, device):
    backbone, temperatures, dims, classes = _load_backbone(
        cfg["base"]/f"fold_{fold}/seed_{seed}/backbone.pt", device)
    checkpoint = e5_root/f"fold_{fold}/seed_{seed}"
    saved = torch.load(checkpoint/"router.pt", map_location=device, weights_only=True)
    router = AnalyticResidualListwiseRouter(dims, classes, len(masks), residual_scale=.5).to(device)
    router.load_state_dict(saved["state_dict"]); router.eval()
    mixer_saved = torch.load(checkpoint/"anchored_harm_oracle.pt", map_location=device, weights_only=True)
    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
    mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
    strength = selected_strength(e6_root, fold, seed); top_k = 3 if len(masks) == 7 else 2
    result = {}
    for split_name in ("selection", "test"):
        raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
        bundle = attach_observed_losses(raw, splits[split_name]["y"])
        _, posterior = load_posteriors(cfg["posterior"], seed, dims, classes, masks,
                                       splits[split_name], bundle["probabilities"], device,
                                       fold_index=fold)
        route = predict_router(router, splits[split_name], bundle["probabilities"],
                               masks, device, posterior)
        route = apply_residual_blend(route, float(saved["blend"]))
        actions, _ = e5.make_actions(bundle["probabilities"], posterior, route, top_k)
        mixture = predict_mixer(mixer, actions, posterior, device)["probability"]
        gate = e6.gate_values("analytic_pi_g2", actions[:, 0], mixture, posterior)
        a7, _ = e6.shrink(actions[:, 0], mixture, gate, strength)
        result[split_name] = {"full": actions[:, 0], "a7": a7}
    return result


def aggregate(dataset: str, group: str, seeds, cfg, e5_root, e6_root, device):
    folds, names = load_dataset(dataset, ROOT/"data"); masks = nonempty_coalitions(len(names))
    output = ROOT/f"runs/formal-s1-replicates/{dataset}/{group}"; output.mkdir(parents=True, exist_ok=True)
    records = []
    for fold, splits in enumerate(folds):
        members = {split: {kind: [] for kind in ("full", "a7")} for split in ("selection", "test")}
        for seed in seeds:
            values = member_outputs(cfg, e5_root, e6_root, splits, masks, fold, seed, device)
            for split in members:
                for kind in members[split]: members[split][kind].append(values[split][kind])
        for split in members:
            for kind in members[split]: members[split][kind] = np.asarray(members[split][kind])
        final = fit_final_system(members["selection"]["full"], members["selection"]["a7"],
                                 splits["selection"]["y"], members["test"]["full"],
                                 members["test"]["a7"], member_ids=seeds)
        frame = pd.DataFrame({"dataset": dataset, "group": group, "fold": fold,
                              "sample_id": splits["test"]["id"],
                              "group_or_video_id": splits["test"]["group"],
                              "label": splits["test"]["y"],
                              "fallback_rho": final.fallback_rho,
                              "aggregation_l2": final.aggregation_l2})
        for index in range(final.final_probability.shape[1]):
            frame[f"p{index}"] = final.final_probability[:, index]
            frame[f"full_p{index}"] = final.full_ensemble[:, index]
        records.append(frame)
    frame = pd.concat(records, ignore_index=True)
    pcols = sorted([column for column in frame if column.startswith("p") and column[1:].isdigit()],
                   key=lambda value: int(value[1:]))
    digest = prediction_hash(frame.sample_id, frame.fold, frame[pcols].to_numpy(float))
    frame.to_parquet(output/"predictions.parquet", index=False)
    (output/"manifest.json").write_text(json.dumps({"method_version": METHOD_VERSION,
        "dataset": dataset, "group": group, "seeds": list(seeds), "prediction_hash": digest,
        "test_labels_role": "evaluation only"}, indent=2), encoding="utf-8")
    print(f"done {dataset}/{group}: {digest[:12]}", flush=True)


def run(dataset: str, group: str, device: str):
    seeds = GROUPS[group]; cfg = configuration(dataset, group)
    e4.CONFIGS[dataset] = cfg; e5.CONFIGS[dataset] = cfg; e6.CONFIGS[dataset] = cfg
    e4.SEEDS = seeds; e5.SEEDS = seeds; e6.SEEDS = seeds
    # Independent system repeats need only the deployed mixer. E5's other
    # variants are ablations already covered by I3-1 and would multiply this
    # confirmation run without changing the final prediction.
    e5.BASELINES = ()
    e5.LEARNED = {"anchored_harm_oracle": {
        "anchor_full": True, "harm_weight": 2., "oracle_weight": .05}}
    root = ROOT/f"runs/formal-s1-replicates/{dataset}/{group}"
    e5_root, e6_root = root/"e5", root/"e6"
    e5.run_dataset(dataset, device, output_root=e5_root)
    e6.run_dataset(dataset, device, e5_root_override=e5_root, output_root=e6_root)
    aggregate(dataset, group, seeds, cfg, e5_root, e6_root, device)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument(
        "--dataset", choices=("mosi", "mosei", "cremad", "avmnist", "all"), default="all")
    parser.add_argument("--group", choices=(*GROUPS, "all"), default="all")
    parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    datasets = ("mosi", "mosei", "cremad", "avmnist") if args.dataset == "all" else (args.dataset,)
    groups = tuple(GROUPS) if args.group == "all" else (args.group,)
    for dataset in datasets:
        for group in groups: run(dataset, group, device)


if __name__ == "__main__": main()
