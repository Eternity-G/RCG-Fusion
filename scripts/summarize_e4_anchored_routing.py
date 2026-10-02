"""Aggregate E4 seed-level results and draw the publication PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "MOSI": ROOT/"runs/formal-e4-mosi/metrics_by_seed.csv",
    "MOSEI": ROOT/"runs/formal-e4-mosei/metrics_by_seed.csv",
    "CREMA-D": ROOT/"runs/formal-e4-cremad-top2/metrics_by_seed.csv",
    "AV-MNIST": ROOT/"runs/formal-e4-avmnist-top2/metrics_by_seed.csv",
}
LABELS = {
    "analytic_only": "Analytic only",
    "hard_oracle_classifier": "Hard oracle cls.",
    "soft_listwise": "Soft listwise",
    "pairwise_only": "Pairwise only",
    "listwise_pairwise_unanchored": "List + pair",
    "listwise_pairwise_anchored": "Anchored list + pair",
}
COLORS = {
    "analytic_only": "#8C8C8C", "hard_oracle_classifier": "#D6A657",
    "soft_listwise": "#77A6B6", "pairwise_only": "#A98BC3",
    "listwise_pairwise_unanchored": "#4C78A8",
    "listwise_pairwise_anchored": "#D95F5F",
}


def ci95(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(stats.t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.
    for rank, index in enumerate(order):
        candidate = min(1., (len(values)-rank)*values[index])
        running = max(running, candidate); adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    frames = []
    for dataset, path in SOURCES.items():
        frame = pd.read_csv(path); frame["dataset"] = dataset; frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    result_dir = ROOT/"results"; figure_dir = ROOT/"figures"
    result_dir.mkdir(exist_ok=True); figure_dir.mkdir(exist_ok=True)
    data.to_csv(result_dir/"e4_routing_by_seed.csv", index=False)

    metrics = ["candidate_oracle_recovery", "candidate_headroom_fraction",
               "action_oracle_nll", "ndcg", "pairwise_accuracy", "top1_recovery",
               "selected_nll", "residual_blend"]
    summary_rows = []
    for (dataset, variant), values in data.groupby(["dataset", "variant"]):
        row = {"dataset": dataset, "variant": variant, "n_seeds": len(values)}
        for metric in metrics:
            row[f"{metric}_mean"] = values[metric].mean()
            row[f"{metric}_sd"] = values[metric].std(ddof=1)
            row[f"{metric}_ci95"] = ci95(values[metric].to_numpy())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(result_dir/"e4_routing_summary.csv", index=False)

    tests = []
    comparisons = [("listwise_pairwise_anchored", "analytic_only", "anchored_vs_analytic"),
                   ("listwise_pairwise_anchored", "listwise_pairwise_unanchored",
                    "anchored_vs_unanchored")]
    for dataset in SOURCES:
        subset = data[data.dataset == dataset]
        for left, right, comparison in comparisons:
            for metric in ("candidate_oracle_recovery", "candidate_headroom_fraction"):
                a = subset[subset.variant == left].sort_values("train_seed")[metric].to_numpy()
                b = subset[subset.variant == right].sort_values("train_seed")[metric].to_numpy()
                difference = a-b
                tests.append({"dataset": dataset, "comparison": comparison, "metric": metric,
                              "mean_difference": difference.mean(), "ci95": ci95(difference),
                              "t_statistic": stats.ttest_rel(a, b).statistic,
                              "p_value": stats.ttest_rel(a, b).pvalue})
    adjusted = holm([row["p_value"] for row in tests])
    for row, value in zip(tests, adjusted): row["holm_p"] = value
    pd.DataFrame(tests).to_csv(result_dir/"e4_routing_paired_tests.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "legend.frameon": False,
    })
    variants = list(LABELS)
    datasets = list(SOURCES)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)

    def grouped(ax, metric, ylabel, scale=1.):
        x = np.arange(len(datasets)); width = .12
        for j, variant in enumerate(variants):
            means, errors = [], []
            for dataset in datasets:
                row = summary[(summary.dataset == dataset) & (summary.variant == variant)].iloc[0]
                means.append(row[f"{metric}_mean"]*scale)
                errors.append(row[f"{metric}_ci95"]*scale)
            ax.bar(x+(j-2.5)*width, means, width, yerr=errors, capsize=1.5,
                   color=COLORS[variant], label=LABELS[variant], linewidth=0)
        ax.set_xticks(x, datasets); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.18, linewidth=.5)

    grouped(axes[0, 0], "candidate_oracle_recovery", "Oracle recovery (%)", 100)
    axes[0, 0].set_title("Candidate-set oracle recovery", loc="left", fontweight="bold")
    grouped(axes[0, 1], "candidate_headroom_fraction", "Recoverable full regret (%)", 100)
    axes[0, 1].set_title("Candidate oracle headroom", loc="left", fontweight="bold")
    grouped(axes[1, 0], "ndcg", "NDCG", 1)
    axes[1, 0].set_title("Full coalition ranking", loc="left", fontweight="bold")

    ax = axes[1, 1]; x = np.arange(len(datasets)); width = .34
    for j, metric in enumerate(("candidate_oracle_recovery", "candidate_headroom_fraction")):
        values, errors = [], []
        for dataset in datasets:
            subset = data[data.dataset == dataset]
            a = subset[subset.variant == "listwise_pairwise_anchored"].sort_values("train_seed")[metric].to_numpy()
            b = subset[subset.variant == "listwise_pairwise_unanchored"].sort_values("train_seed")[metric].to_numpy()
            values.append((a-b).mean()*100); errors.append(ci95((a-b)*100))
        ax.bar(x+(j-.5)*width, values, width, yerr=errors, capsize=2,
               color=("#D95F5F", "#4C78A8")[j],
               label=("Oracle recovery", "Headroom")[j])
    ax.axhline(0, color="#444444", linewidth=.7); ax.set_xticks(x, datasets)
    ax.set_ylabel("Anchoring effect (percentage points)")
    ax.set_title("Anchoring vs. unanchored residual rank", loc="left", fontweight="bold")
    ax.legend(ncol=2, loc="lower left"); ax.grid(axis="y", alpha=.18, linewidth=.5)

    for label, ax in zip("abcd", axes.flat):
        ax.text(-.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("Listwise residual routing expands useful candidates mainly in text-dominant tasks",
                 y=1.075, fontsize=9, fontweight="bold")
    fig.savefig(figure_dir/"e4_anchored_routing.png", dpi=300, facecolor="white",
                bbox_inches="tight")


if __name__ == "__main__":
    main()
