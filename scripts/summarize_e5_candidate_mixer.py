"""Aggregate E5 candidate-mixer results and export the publication PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {name: ROOT/f"runs/formal-e5-{slug}/metrics_by_seed.csv" for name, slug in (
    ("MOSI", "mosi"), ("MOSEI", "mosei"), ("CREMA-D", "cremad"),
    ("AV-MNIST", "avmnist"))}
METHODS = ["posterior_only", "analytic_hard", "listwise_hard", "equal_action_average",
           "softmax_mixer", "anchored_mixer", "anchored_harm",
           "anchored_harm_oracle"]
LABELS = {"posterior_only": "Posterior only", "analytic_hard": "Analytic hard", "listwise_hard": "Listwise hard",
          "equal_action_average": "Equal average", "softmax_mixer": "Softmax",
          "anchored_mixer": "Full anchor", "anchored_harm": "+ harm",
          "anchored_harm_oracle": "+ harm + oracle"}
COLORS = {"posterior_only": "#3E8E7E", "analytic_hard": "#8C8C8C", "listwise_hard": "#B0B0B0",
          "equal_action_average": "#D6A657", "softmax_mixer": "#77A6B6",
          "anchored_mixer": "#4C78A8", "anchored_harm": "#A98BC3",
          "anchored_harm_oracle": "#D95F5F"}


def ci95(values):
    values = np.asarray(values, float)
    return float(stats.t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))


def holm(values):
    order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.
    for rank, index in enumerate(order):
        candidate = min(1., (len(values)-rank)*values[index])
        running = max(running, candidate); adjusted[index] = running
    return adjusted


def main():
    frames = []
    for dataset, path in SOURCES.items():
        frame = pd.read_csv(path); frame["dataset"] = dataset; frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    result_dir, figure_dir = ROOT/"results", ROOT/"figures"
    result_dir.mkdir(exist_ok=True); figure_dir.mkdir(exist_ok=True)
    data.to_csv(result_dir/"e5_candidate_mixer_by_seed.csv", index=False)

    metrics = ["accuracy", "macro_f1", "nll", "accuracy_gain", "macro_f1_gain",
               "nll_gain", "relative_nll_reduction", "negative_flip_rate",
               "correction_rate", "clipped_harm", "regret_reduction",
               "mean_full_weight", "mean_posterior_weight", "mean_candidate_weight",
               "candidate_usage_rate", "action_oracle_gap"]
    rows = []
    for (dataset, variant), values in data.groupby(["dataset", "variant"]):
        row = {"dataset": dataset, "variant": variant, "n_seeds": len(values)}
        for metric in metrics:
            row[f"{metric}_mean"] = values[metric].mean()
            row[f"{metric}_sd"] = values[metric].std(ddof=1)
            row[f"{metric}_ci95"] = ci95(values[metric].dropna()) if values[metric].notna().all() else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(result_dir/"e5_candidate_mixer_summary.csv", index=False)

    comparisons = []
    pairs = [("softmax_mixer", "full_coalition", "softmax_vs_full"),
             ("softmax_mixer", "posterior_only", "softmax_vs_posterior"),
             ("anchored_harm", "posterior_only", "harm_vs_posterior"),
             ("anchored_mixer", "softmax_mixer", "anchor_vs_softmax"),
             ("anchored_harm", "anchored_mixer", "harm_vs_anchor"),
             ("anchored_harm_oracle", "anchored_harm", "oracle_vs_harm")]
    for dataset in SOURCES:
        subset = data[data.dataset == dataset]
        for left, right, name in pairs:
            for metric in ("nll", "accuracy", "clipped_harm"):
                a = subset[subset.variant == left].sort_values("train_seed")[metric].to_numpy()
                b = subset[subset.variant == right].sort_values("train_seed")[metric].to_numpy()
                difference = a-b
                test = stats.ttest_rel(a, b)
                comparisons.append({"dataset": dataset, "comparison": name, "metric": metric,
                                    "mean_difference": difference.mean(), "ci95": ci95(difference),
                                    "t_statistic": test.statistic, "p_value": test.pvalue})
    adjusted = holm([row["p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted): row["holm_p"] = value
    pd.DataFrame(comparisons).to_csv(result_dir/"e5_candidate_mixer_paired_tests.csv", index=False)

    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7,
                         "legend.frameon": False})
    datasets = list(SOURCES); x = np.arange(len(datasets)); width = .102
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)

    def grouped(ax, metric, ylabel, scale=1., methods=METHODS):
        for index, method in enumerate(methods):
            means, errors = [], []
            for dataset in datasets:
                row = summary[(summary.dataset == dataset) & (summary.variant == method)].iloc[0]
                means.append(row[f"{metric}_mean"]*scale)
                errors.append(row[f"{metric}_ci95"]*scale)
            ax.bar(x+(index-(len(methods)-1)/2)*width, means, width,
                   yerr=errors, capsize=1.4, color=COLORS[method],
                   label=LABELS[method], linewidth=0)
        ax.axhline(0, color="#444444", linewidth=.7)
        ax.set_xticks(x, datasets); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18, linewidth=.5)

    grouped(axes[0, 0], "nll_gain", "NLL improvement", 1)
    axes[0, 0].set_title("Probability quality relative to full fusion", loc="left", fontweight="bold")
    grouped(axes[0, 1], "accuracy_gain", "Accuracy change (pp)", 100)
    axes[0, 1].set_title("Decision accuracy relative to full fusion", loc="left", fontweight="bold")

    net = data.copy(); net["net_correction"] = net.correction_rate-net.negative_flip_rate
    net_rows = []
    for (dataset, variant), values in net.groupby(["dataset", "variant"]):
        net_rows.append({"dataset": dataset, "variant": variant,
                         "mean": values.net_correction.mean(),
                         "ci": ci95(values.net_correction.to_numpy())})
    net_summary = pd.DataFrame(net_rows)
    ax = axes[1, 0]
    for index, method in enumerate(METHODS):
        values = [net_summary[(net_summary.dataset == d) &
                              (net_summary.variant == method)].iloc[0] for d in datasets]
        ax.bar(x+(index-(len(METHODS)-1)/2)*width, [v["mean"]*100 for v in values], width,
               yerr=[v["ci"]*100 for v in values], capsize=1.4,
               color=COLORS[method], linewidth=0)
    ax.axhline(0, color="#444444", linewidth=.7); ax.set_xticks(x, datasets)
    ax.set_ylabel("Corrections minus flips (pp)")
    ax.set_title("Net correction rate", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.18, linewidth=.5)

    learned = METHODS[4:]
    grouped(axes[1, 1], "clipped_harm", "Mean clipped harm (%)", 100, learned)
    axes[1, 1].axhline(2, color="#C44E52", linestyle="--", linewidth=1,
                       label="2% planned budget")
    axes[1, 1].set_title("Raw mixer harm before adaptive shrinkage", loc="left", fontweight="bold")

    for label, ax in zip("abcd", axes.flat):
        ax.text(-.13, 1.08, label, transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHODS]
    fig.legend(handles, [LABELS[m] for m in METHODS], loc="outside upper center",
               ncol=4, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("Soft candidate mixing improves predictions but does not by itself control harm",
                 y=1.075, fontsize=9, fontweight="bold")
    fig.savefig(figure_dir/"e5_candidate_mixer.png", dpi=300, facecolor="white",
                bbox_inches="tight")


if __name__ == "__main__":
    main()
