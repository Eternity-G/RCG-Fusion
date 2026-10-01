"""Summarize and plot an E2 supervision ablation."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (11, 22, 33, 44, 55)
CONFIGS = {
    "mosi": {
        "run": ROOT / "runs/formal-e2-mosi",
        "base": ROOT / "runs/rcg-fusion-mosi-v4",
        "oof": ROOT / "runs/rcg-fusion-mosi-v4/fold_0/oof_targets.npz",
        "figure": ROOT / "figures/e2_mosi_supervision_ablation.png",
    },
    "mosei": {
        "run": ROOT / "runs/formal-e2-mosei",
        "base": ROOT / "runs/rcg-fusion-mosei-shrinkage-v1",
        "oof": ROOT / "runs/formal-e2-mosei-oof/oof_targets.npz",
        "figure": ROOT / "figures/e2_mosei_supervision_ablation.png",
    },
}
ORDER = (
    "in_sample_hard", "oof_single_hard", "oof_multi_hard",
    "oof_multi_soft", "oof_soft_pair", "oof_soft_full",
)
LABELS = {
    "in_sample_hard": "In-sample hard",
    "oof_single_hard": "OOF-1 hard",
    "oof_multi_hard": "OOF-5 hard",
    "oof_multi_soft": "OOF-5 soft",
    "oof_soft_pair": "Soft + pairwise",
    "oof_soft_full": "Full soft",
}


def teacher_statistics(oof_path: Path) -> dict[str, float]:
    with np.load(oof_path) as saved:
        losses = saved["losses"]
    teachers, _, coalitions = losses.shape
    oracle = losses.argmin(2)
    hard = np.eye(coalitions)[oracle]
    softened = softmax(-losses / 0.1, axis=2)
    hard_variance = float(((hard - hard.mean(0)) ** 2).sum(2).mean())
    soft_variance = float(((softened - softened.mean(0)) ** 2).sum(2).mean())
    vote = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=coalitions).max() / teachers,
        0, oracle,
    )
    pair_agreement = np.mean([
        np.mean(oracle[first] == oracle[second])
        for first, second in combinations(range(teachers), 2)
    ])
    aggregate = softened.mean(0)
    normalized_entropy = float((
        -(aggregate * np.log(np.clip(aggregate, 1e-12, 1))).sum(1)
        / np.log(coalitions)
    ).mean())
    return {
        "hard_target_variance": hard_variance,
        "soft_target_variance": soft_variance,
        "soft_to_hard_variance_ratio": soft_variance / hard_variance,
        "mean_max_vote_fraction": float(vote.mean()),
        "at_least_four_of_five_agreement": float(np.mean(vote >= 0.8)),
        "five_of_five_agreement": float(np.mean(vote == 1.0)),
        "mean_pairwise_teacher_oracle_agreement": float(pair_agreement),
        "normalized_soft_oracle_entropy": normalized_entropy,
    }


def audit_frame(base: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        record = json.loads((base / f"fold_0/seed_{seed}/target_audit.json").read_text())
        rows.append({
            "train_seed": seed,
            "mean_in_sample_loss": record["mean_in_sample_loss"],
            "mean_oof_loss": record["mean_oof_loss"],
            "oof_minus_in_sample_loss": record["oof_minus_in_sample_loss"],
            "oof_in_sample_loss_spearman": record["oof_in_sample_loss_spearman"],
        })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "router_ndcg", "router_pairwise_accuracy", "router_top1", "router_top2",
        "anchored_top3", "candidate_oracle_nll", "selected_nll", "selection_regret",
    )
    rows = []
    for variant in ORDER:
        values = frame[frame.variant == variant]
        row = {"variant": variant}
        for metric in metrics:
            row[f"{metric}_mean"] = values[metric].mean()
            row[f"{metric}_std"] = values[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def comparison(frame: pd.DataFrame, first: str, second: str, metric: str) -> dict:
    pivot = frame.pivot(index="train_seed", columns="variant", values=metric)
    difference = pivot[second] - pivot[first]
    pvalue = None if np.allclose(difference, 0.0) else float(
        ttest_1samp(difference, 0.0).pvalue
    )
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "mean_second_minus_first": float(difference.mean()),
        "seeds_second_better": int((
            difference > 0 if "nll" not in metric and "regret" not in metric
            else difference < 0
        ).sum()),
        "paired_t_pvalue": pvalue,
    }


def plot(audit: pd.DataFrame, stats: dict[str, float], summary: pd.DataFrame,
         figure: Path) -> None:
    figure.parent.mkdir(parents=True, exist_ok=True)
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)

    x = np.arange(len(audit))
    for index, row in audit.iterrows():
        axes[0, 0].plot(
            [x[index] - 0.08, x[index] + 0.08],
            [row.mean_in_sample_loss, row.mean_oof_loss],
            color="#999999", linewidth=1,
        )
    axes[0, 0].scatter(x - 0.08, audit.mean_in_sample_loss, color=blue,
                       label="In-sample", zorder=2)
    axes[0, 0].scatter(x + 0.08, audit.mean_oof_loss, color=orange,
                       label="OOF", zorder=2)
    axes[0, 0].set_xticks(x, [str(seed) for seed in audit.train_seed])
    axes[0, 0].set_xlabel("Task seed")
    axes[0, 0].set_ylabel("Mean coalition NLL")
    axes[0, 0].set_title("(a) Training optimism audit", loc="left", fontsize=10)
    axes[0, 0].legend(frameon=False)

    variance = [stats["hard_target_variance"], stats["soft_target_variance"]]
    axes[0, 1].bar([0, 1], variance, color=[orange, green], width=0.62)
    axes[0, 1].set_xticks([0, 1], ["Hard oracle", "Soft oracle"])
    axes[0, 1].set_ylabel("Across-teacher target variance")
    axes[0, 1].set_title("(b) Target stability", loc="left", fontsize=10)
    for index, value in enumerate(variance):
        axes[0, 1].text(index, value, f"{value:.3f}", ha="center", va="bottom")

    ordered = summary.set_index("variant").loc[list(ORDER)]
    positions = np.arange(len(ORDER))
    axes[1, 0].bar(
        positions, ordered.anchored_top3_mean,
        yerr=ordered.anchored_top3_std, color=blue, capsize=3,
    )
    axes[1, 0].set_xticks(positions, [LABELS[value] for value in ORDER],
                         rotation=22, ha="right")
    axes[1, 0].set_ylabel("Top-3 oracle coverage")
    axes[1, 0].set_title("(c) Candidate coverage (mean ± SD)", loc="left", fontsize=10)

    axes[1, 1].bar(
        positions, ordered.candidate_oracle_nll_mean,
        yerr=ordered.candidate_oracle_nll_std, color=green, capsize=3,
    )
    axes[1, 1].set_xticks(positions, [LABELS[value] for value in ORDER],
                         rotation=22, ha="right")
    axes[1, 1].set_ylabel("Candidate oracle NLL")
    axes[1, 1].set_title("(d) Best loss inside Top-3 (mean ± SD)", loc="left", fontsize=10)

    for axis in axes.flat:
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)


def run(dataset: str) -> None:
    config = CONFIGS[dataset]
    frame = pd.read_csv(config["run"] / "metrics_by_seed.csv")
    if set(frame.variant) != set(ORDER) or len(frame) != 30:
        raise ValueError("expected six variants by five seeds")
    if frame.groupby("variant").train_seed.nunique().min() != 5:
        raise ValueError("each E2 variant must contain all five seeds")
    audit = audit_frame(config["base"])
    stats = teacher_statistics(config["oof"])
    summary = summarize(frame)
    comparisons = [
        comparison(frame, "oof_single_hard", "oof_multi_hard", "anchored_top3"),
        comparison(frame, "oof_multi_hard", "oof_multi_soft", "anchored_top3"),
        comparison(frame, "oof_multi_hard", "oof_soft_pair", "anchored_top3"),
        comparison(frame, "oof_multi_soft", "oof_soft_full", "anchored_top3"),
        comparison(frame, "in_sample_hard", "oof_soft_full", "anchored_top3"),
        comparison(frame, "in_sample_hard", "oof_soft_full", "candidate_oracle_nll"),
    ]
    RESULTS.mkdir(exist_ok=True)
    prefix = f"e2_{dataset}"
    frame.to_csv(RESULTS / f"{prefix}_supervision_ablation_by_seed.csv", index=False)
    summary.to_csv(RESULTS / f"{prefix}_supervision_ablation.csv", index=False)
    audit.to_csv(RESULTS / f"{prefix}_target_audit.csv", index=False)
    output = {
        "dataset": dataset.upper(), "seeds": list(SEEDS),
        "teacher_statistics": stats,
        "audit_mean": {
            column: float(audit[column].mean())
            for column in audit.columns if column != "train_seed"
        },
        "comparisons": comparisons,
    }
    (RESULTS / f"{prefix}_supervision_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(audit, stats, summary, config["figure"])
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CONFIGS), default="mosi")
    run(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
