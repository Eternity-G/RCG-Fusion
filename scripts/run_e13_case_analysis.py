"""E13: rule-based success, failure, RCG, and full-retention case analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import predict_mixer
from rcg.final_system import METHOD_VERSION
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import apply_residual_blend, predict_router
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_e5_candidate_mixer_ablation import make_actions  # noqa: E402
from run_e6_adaptive_shrinkage import gate_values, shrink  # noqa: E402


DATASETS = ("mosi", "mosei", "cremad", "avmnist")
DISPLAY = {"mosi": "MOSI", "mosei": "MOSEI", "cremad": "CREMA-D",
           "avmnist": "AV-MNIST"}
COLORS = {"mosi": "#4C78A8", "mosei": "#72B7B2", "cremad": "#F2A65A",
          "avmnist": "#9C6ADE"}


def coalition_name(mask, names):
    return "+".join(name for name, keep in zip(names, mask) if keep)


def selected_strength(dataset, fold, seed):
    frame = pd.read_csv(ROOT/f"runs/formal-e6-{dataset}/metrics_by_fold_seed.csv")
    row = frame[(frame.variant == "analytic_pi_g2") &
                (frame.fold == fold) & (frame.train_seed == seed)]
    if len(row) != 1:
        raise ValueError(f"missing E6 strength for {dataset}/{fold}/{seed}")
    return float(row.iloc[0].strength)


def member_detail(dataset, fold, splits, names, seed, device):
    cfg = CONFIGS[dataset]; masks = nonempty_coalitions(len(names))
    backbone, temperatures, dims, classes = _load_backbone(
        cfg["base"]/f"fold_{fold}/seed_{seed}/backbone.pt", device)
    checkpoint = ROOT/f"runs/formal-e5-{dataset}/fold_{fold}/seed_{seed}"
    router_saved = torch.load(checkpoint/"router.pt", map_location=device, weights_only=True)
    router = AnalyticResidualListwiseRouter(
        dims, classes, len(masks), residual_scale=.5).to(device)
    router.load_state_dict(router_saved["state_dict"]); router.eval()
    mixer_saved = torch.load(checkpoint/"anchored_harm_oracle.pt",
                             map_location=device, weights_only=True)
    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
    mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
    result = {}
    for split_name in ("calibration", "test"):
        raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
        bundle = attach_observed_losses(raw, splits[split_name]["y"])
        _, posterior = load_posteriors(cfg["posterior"], seed, dims, classes, masks,
                                       splits[split_name], bundle["probabilities"], device,
                                       fold_index=fold)
        if split_name == "calibration":
            result[split_name] = {"probabilities": bundle["probabilities"]}
            continue
        route = predict_router(router, splits[split_name], bundle["probabilities"],
                               masks, device, posterior)
        route = apply_residual_blend(route, float(router_saved["blend"]))
        top_k = min(3 if len(names) >= 3 else 2, len(masks)-1)
        action, candidates = make_actions(bundle["probabilities"], posterior, route, top_k)
        mixer_output = predict_mixer(mixer, action, posterior, device)
        strength = selected_strength(dataset, fold, seed)
        gate = gate_values("analytic_pi_g2", action[:, 0], mixer_output["probability"], posterior)
        final, alpha = shrink(action[:, 0], mixer_output["probability"], gate, strength)
        result[split_name] = {
            "probabilities": bundle["probabilities"], "losses": bundle["losses"],
            "posterior": posterior, "analytic": route["analytic_score"],
            "route": route["score"], "candidates": candidates,
            "action_weight": mixer_output["weight"], "mixture": mixer_output["probability"],
            "a7": final, "alpha": alpha, "benefit_probability": np.sqrt(gate),
        }
    del backbone, router, mixer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def final_from_e8(dataset, fold, sample_ids):
    """Load the canonical final probabilities instead of reconstructing A8 locally."""
    path = ROOT/f"runs/formal-e8-{dataset}/final_predictions.parquet"
    table = pd.read_parquet(path)
    table = table[table.fold == fold].copy()
    table.sample_id = table.sample_id.astype(str)
    table = table.set_index("sample_id").loc[np.asarray(sample_ids, dtype=str)].reset_index()
    if not (table.method_version == METHOD_VERSION).all():
        raise ValueError(f"noncanonical method version in {path}")
    pcols = sorted((column for column in table if column.startswith("p")),
                   key=lambda value: int(value[1:]))
    fullcols = sorted((column for column in table if column.startswith("full_p")),
                      key=lambda value: int(value[6:]))
    return (table[fullcols].to_numpy(), table[pcols].to_numpy(),
            float(table.fallback_rho.iloc[0]))


def json_map(names, values, digits=4):
    return json.dumps({name: round(float(value), digits) for name, value in zip(names, values)},
                      ensure_ascii=False, separators=(",", ":"))


def build_fold_records(dataset, fold, splits, names, members):
    masks = nonempty_coalitions(len(names)); cnames = [coalition_name(mask, names) for mask in masks]
    test = [item["test"] for item in members]
    calibration = [item["calibration"] for item in members]
    probability = np.mean([item["probabilities"] for item in test], axis=0)
    calibration_probability = np.mean([item["probabilities"] for item in calibration], axis=0)
    full, final, rho = final_from_e8(dataset, fold, splits["test"]["id"])
    posterior = np.mean([item["posterior"] for item in test], axis=0)
    analytic = np.mean([item["analytic"] for item in test], axis=0)
    route = np.mean([item["route"] for item in test], axis=0)
    action_weight = np.mean([item["action_weight"] for item in test], axis=0)
    alpha = np.mean([item["alpha"] for item in test], axis=0)
    benefit_probability = np.mean([item["benefit_probability"] for item in test], axis=0)
    y = splits["test"]["y"]; rows = np.arange(len(y))
    full_loss = -np.log(np.clip(full[rows, y], 1e-12, 1))
    final_loss = -np.log(np.clip(final[rows, y], 1e-12, 1))
    coalition_loss = -np.log(np.clip(probability[rows, :, y], 1e-12, 1))
    full_prediction, final_prediction = full.argmax(1), final.argmax(1)
    route_order = np.argsort(-route, axis=1)
    analytic_order = np.argsort(-analytic, axis=1)
    singleton, without, threshold = {}, {}, {}
    for modality, name in enumerate(names):
        one = tuple(int(index == modality) for index in range(len(names)))
        removed = tuple(int(index != modality) for index in range(len(names)))
        singleton[name], without[name] = masks.index(one), masks.index(removed)
        calibration_reliability = calibration_probability[:, singleton[name]].max(1)
        threshold[name] = float(np.quantile(calibration_reliability, .9))
    records = []
    for index in range(len(y)):
        role_names = ["full", "posterior", "anchor"]
        role_names += [f"support{rank}" for rank in range(1, action_weight.shape[1]-2)]
        reliability = {name: float(probability[index, singleton[name]].max()) for name in names}
        contribution = {name: float(coalition_loss[index, without[name]]-
                                    coalition_loss[index, -1]) for name in names}
        true_probability = probability[index, :, y[index]]
        full_advantage = float(coalition_loss[index, :-1].min()-coalition_loss[index, -1])
        records.append({
            "dataset": dataset, "fold": fold, "sample_id": str(splits["test"]["id"][index]),
            "method_version": METHOD_VERSION,
            "group_or_video_id": str(splits["test"]["group"][index]), "label": int(y[index]),
            "base_prediction": int(full_prediction[index]),
            "final_prediction": int(final_prediction[index]),
            "base_correct": bool(full_prediction[index] == y[index]),
            "final_correct": bool(final_prediction[index] == y[index]),
            "base_loss": float(full_loss[index]), "final_loss": float(final_loss[index]),
            "loss_benefit": float(full_loss[index]-final_loss[index]),
            "base_true_probability": float(full[index, y[index]]),
            "final_true_probability": float(final[index, y[index]]),
            "mean_alpha": float(alpha[index]),
            "benefit_probability": float(benefit_probability[index]), "rho": rho,
            "full_advantage_over_best_incomplete": full_advantage,
            "full_is_coalition_oracle": bool(coalition_loss[index].argmin() == len(masks)-1),
            "coalition_true_probabilities": json_map(cnames, true_probability),
            "coalition_probability_vectors": json.dumps(
                {name: probability[index, j].round(6).tolist() for j, name in enumerate(cnames)},
                ensure_ascii=False, separators=(",", ":")),
            "analytic_scores": json_map(cnames, analytic[index]),
            "route_scores": json_map(cnames, route[index]),
            "analytic_ranking": ">".join(cnames[j] for j in analytic_order[index]),
            "route_ranking": ">".join(cnames[j] for j in route_order[index]),
            "action_role_weights": json_map(role_names, action_weight[index]),
            "reliability": json.dumps(reliability, ensure_ascii=False, separators=(",", ":")),
            "reliability_threshold": json.dumps(threshold, ensure_ascii=False,
                                                 separators=(",", ":")),
            "deletion_contribution": json.dumps(contribution, ensure_ascii=False,
                                                 separators=(",", ":")),
            "posterior": json.dumps(posterior[index].round(6).tolist(), separators=(",", ":")),
            "base_probability": json.dumps(full[index].round(6).tolist(), separators=(",", ":")),
            "final_probability": json.dumps(final[index].round(6).tolist(), separators=(",", ":")),
        })
    return pd.DataFrame(records)


def select_cases(records):
    cases = []
    for dataset, frame in records.groupby("dataset", sort=False):
        success = frame[(~frame.base_correct) & frame.final_correct].nlargest(5, "loss_benefit")
        for rank, (_, row) in enumerate(success.iterrows(), 1):
            cases.append({**row.to_dict(), "case_type": "successful_correction", "rank": rank,
                          "selection_value": row.loss_benefit, "modality": ""})
        failure = frame[frame.base_correct & (~frame.final_correct)].copy()
        failure["damage"] = -failure.loss_benefit
        failure = failure.nlargest(5, "damage")
        for rank, (_, row) in enumerate(failure.iterrows(), 1):
            cases.append({**row.to_dict(), "case_type": "negative_flip", "rank": rank,
                          "selection_value": row.damage, "modality": ""})
        harmful = []
        for _, row in frame.iterrows():
            reliability = json.loads(row.reliability)
            threshold = json.loads(row.reliability_threshold)
            contribution = json.loads(row.deletion_contribution)
            for modality in reliability:
                if reliability[modality] >= threshold[modality] and contribution[modality] < -.01:
                    harmful.append({**row.to_dict(), "modality": modality,
                                    "modality_reliability": reliability[modality],
                                    "modality_threshold": threshold[modality],
                                    "modality_contribution": contribution[modality],
                                    "selection_value": -contribution[modality]})
        harmful = sorted(harmful, key=lambda item: item["selection_value"], reverse=True)[:5]
        for rank, row in enumerate(harmful, 1):
            cases.append({**row, "case_type": "high_reliability_harm", "rank": rank})
        if dataset == "avmnist":
            retained = frame[(frame.full_is_coalition_oracle) & frame.base_correct &
                             frame.final_correct].sort_values(
                                 ["mean_alpha", "full_advantage_over_best_incomplete"],
                                 ascending=[True, False]).head(5)
            for rank, (_, row) in enumerate(retained.iterrows(), 1):
                cases.append({**row.to_dict(), "case_type": "full_retention", "rank": rank,
                              "selection_value": row.full_advantage_over_best_incomplete,
                              "modality": ""})
    return pd.DataFrame(cases)


def compact_alliances(text):
    values = json.loads(text)
    return "; ".join(f"{key}:{value:.3f}" for key, value in values.items())


def write_markdown(cases, output):
    lines = ["# E13 规则选取案例表", "",
             "所有案例由脚本中的冻结规则自动产生；完整逐样本内部量见Parquet源记录。", ""]
    for case_type, title in (("successful_correction", "成功纠错：按CE收益降序"),
                             ("negative_flip", "负向翻转：按CE损害降序"),
                             ("high_reliability_harm", "高可靠负贡献：按损害幅度降序"),
                             ("full_retention", "AV-MNIST完整联盟保留：先按alpha升序")):
        lines += [f"## {title}", ""]
        subset = cases[cases.case_type == case_type]
        if case_type == "high_reliability_harm":
            lines += ["| 数据集 | 排名 | 样本 | 模态 | 可靠性/阈值 | 条件贡献 | α | 路由Top-3 | 联盟真实类概率 |",
                      "|---|---:|---|---|---:|---:|---:|---|---|"]
            for row in subset.itertuples():
                lines.append(f"| {DISPLAY[row.dataset]} | {row.rank} | `{row.sample_id}` | {row.modality} | "
                             f"{row.modality_reliability:.3f}/{row.modality_threshold:.3f} | "
                             f"{row.modality_contribution:.3f} | {row.mean_alpha:.3f} | "
                             f"{'>'.join(row.route_ranking.split('>')[:3])} | "
                             f"{compact_alliances(row.coalition_true_probabilities)} |")
        else:
            lines += ["| 数据集 | 排名 | 样本 | 标签 | Base→Final | CE收益 | p(y) Base→Final | α | 路由Top-3 | 动作权重 |",
                      "|---|---:|---|---:|---|---:|---:|---:|---|---|"]
            for row in subset.itertuples():
                lines.append(f"| {DISPLAY[row.dataset]} | {row.rank} | `{row.sample_id}` | {row.label} | "
                             f"{row.base_prediction}→{row.final_prediction} | {row.loss_benefit:+.3f} | "
                             f"{row.base_true_probability:.3f}→{row.final_true_probability:.3f} | "
                             f"{row.mean_alpha:.3f} | {'>'.join(row.route_ranking.split('>')[:3])} | "
                             f"{row.action_role_weights} |")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def render(cases, output):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.linewidth": .8})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)
    for axis, case_type, title in zip(
            axes[:2], ("successful_correction", "negative_flip"),
            ("Largest rule-selected corrections", "Largest rule-selected negative flips")):
        data = cases[cases.case_type == case_type].groupby("dataset").head(1)
        y = np.arange(len(data))
        axis.hlines(y, data.base_true_probability, data.final_true_probability,
                    color="#B9B9B9", linewidth=1.5)
        axis.scatter(data.base_true_probability, y, color="#777777", s=25, label="Full ensemble")
        axis.scatter(data.final_true_probability, y,
                     c=[COLORS[item] for item in data.dataset], s=32, label="RCG final")
        axis.set_yticks(y, [DISPLAY[item] for item in data.dataset]); axis.invert_yaxis()
        axis.set_xlim(0, 1); axis.set_xlabel("True-class probability"); axis.set_title(title)
        axis.grid(axis="x", color="#E8E8E8", linewidth=.6); axis.set_axisbelow(True)
    harmful = cases[cases.case_type == "high_reliability_harm"]
    for dataset, data in harmful.groupby("dataset"):
        axes[2].scatter(data.modality_reliability, -data.modality_contribution,
                        color=COLORS[dataset], label=DISPLAY[dataset], s=28, alpha=.85)
    axes[2].set_xlabel("Unimodal reliability")
    axes[2].set_ylabel("Negative contribution magnitude (nats)")
    axes[2].set_title("High-reliability harmful modalities")
    axes[2].legend(frameon=False, fontsize=6)
    axes[2].text(.03, .94, "No AV-MNIST case passed its calibration-set q90 threshold",
                 transform=axes[2].transAxes, fontsize=6, color="#666666",
                 va="top", ha="left")
    axes[2].grid(color="#E8E8E8", linewidth=.6); axes[2].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-.13, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default=str(ROOT/"results"))
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    all_records = []
    for dataset in DATASETS:
        folds, names = load_dataset(dataset, ROOT/"data")
        for fold, splits in enumerate(folds):
            members = [member_detail(dataset, fold, splits, names, seed, device) for seed in SEEDS]
            all_records.append(build_fold_records(dataset, fold, splits, names, members))
            print(f"{dataset} fold={fold}: detailed inference complete", flush=True)
    records = pd.concat(all_records, ignore_index=True)
    cases = select_cases(records)
    records.to_parquet(output/"e13_case_records.parquet", index=False)
    cases.to_csv(output/"e13_selected_cases.csv", index=False)
    write_markdown(cases, output/"e13_case_tables.md")
    figure = ROOT/"figures/e13_case_analysis.png"; figure.parent.mkdir(exist_ok=True)
    render(cases, figure)
    counts = cases.groupby(["dataset", "case_type"]).size().rename("selected").reset_index()
    counts.to_csv(output/"e13_case_counts.csv", index=False)
    source_hashes = {
        dataset: json.loads((ROOT/f"runs/formal-e8-{dataset}/manifest.json").read_text(
            encoding="utf-8"))["prediction_hash"] for dataset in DATASETS
    }
    (output/"e13_manifest.json").write_text(json.dumps({
        "experiment": "E13 rule-based cases", "task_seeds": list(SEEDS),
        "method_version": METHOD_VERSION, "source_prediction_hashes": source_hashes,
        "success_rule": "base wrong, A8 final correct; top 5 by CE benefit",
        "failure_rule": "base correct, A8 final wrong; top 5 by CE damage",
        "harm_rule": "ensemble singleton reliability >= calibration q90 and deletion contribution < -0.01; top 5 by harm",
        "avmnist_retention_rule": "full coalition oracle and base/final correct; ascending mean alpha then descending full advantage",
        "test_labels_role": "offline rule-based selection and explanation only",
        "figure": "PNG, 300 dpi, white background"
    }, indent=2), encoding="utf-8")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
