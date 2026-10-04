"""Summarize I1-2 fixed-chain downstream transfer across datasets."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
ORDER = ("in_sample_hard", "oof_single_hard", "oof_multi_hard",
         "oof_multi_soft", "oof_soft_pair", "oof_soft_full")
LABELS = ("In-sample", "OOF-1", "OOF-5 hard", "OOF-5 soft",
          "Soft + pair", "Full soft")
COLORS = ("#7F7F7F", "#D55E00", "#E69F00", "#56B4E9", "#0072B2", "#009E73")


def holm(values):
    result = np.ones(len(values)); order = np.argsort(values); running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[index]))
        result[index] = running
    return result


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    def fmt(value):
        return f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(fmt(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join((header, rule, *rows))


def main():
    frames = []
    for dataset in DATASETS:
        path = ROOT/"runs"/f"formal-i1-2-{dataset}"/"metrics_by_seed.csv"
        if not path.exists():
            raise FileNotFoundError(f"I1-2 is incomplete: {path}")
        frame = pd.read_csv(path)
        if set(frame.variant) != set(ORDER) or frame.groupby("variant").train_seed.nunique().min() != 5:
            raise ValueError(f"I1-2 does not contain six variants x five seeds: {dataset}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(ROOT/"results/i1_2_downstream_by_seed.csv", index=False)
    summary = data.groupby(["dataset", "variant"]).agg(
        nll_gain=("adaptive_nll_gain", "mean"),
        accuracy_gain=("adaptive_accuracy_gain", "mean"),
        regret_reduction=("adaptive_regret_reduction", "mean"),
        correction=("adaptive_correction_rate", "mean"),
        negative_flip=("adaptive_negative_flip_rate", "mean"),
        clipped_harm=("adaptive_clipped_harm", "mean"),
        strength=("shrinkage_strength", "mean"),
    ).reset_index()
    summary.to_csv(ROOT/"results/i1_2_downstream_summary.csv", index=False)

    comparisons = []
    for dataset in DATASETS:
        part = data[data.dataset == dataset]
        pivot = part.pivot(index="train_seed", columns="variant", values="adaptive_nll_gain")
        for first in ("in_sample_hard", "oof_single_hard", "oof_multi_hard"):
            difference = pivot["oof_soft_full"]-pivot[first]
            pvalue = 1. if np.allclose(difference, 0) else float(
                ttest_1samp(difference, 0, alternative="greater").pvalue)
            comparisons.append({"dataset": dataset, "first": first,
                                "second": "oof_soft_full",
                                "nll_gain_difference": float(difference.mean()),
                                "improved_seeds": int((difference > 0).sum()),
                                "paired_t_one_sided_p": pvalue})
    comparison = pd.DataFrame(comparisons)
    comparison["holm_p"] = holm(comparison.paired_t_one_sided_p.to_numpy())
    comparison.to_csv(ROOT/"results/i1_2_downstream_comparisons.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
    x = np.arange(len(DATASETS)); width = .12
    for index, variant in enumerate(ORDER):
        values = summary[summary.variant == variant].set_index("dataset").loc[list(DATASETS)]
        axes[0].bar(x+(index-2.5)*width, values.nll_gain, width,
                    color=COLORS[index], label=LABELS[index])
        axes[1].bar(x+(index-2.5)*width, values.regret_reduction*100, width,
                    color=COLORS[index])
    names = [value.upper() if value != "cremad" else "CREMA-D" for value in DATASETS]
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[0].set_ylabel("Adaptive-chain NLL improvement")
    axes[1].set_ylabel("Fusion regret reduction (%)")
    axes[0].set_title("(a) Supervision-to-NLL transfer", loc="left", fontsize=10)
    axes[1].set_title("(b) Supervision-to-regret transfer", loc="left", fontsize=10)
    for axis in axes:
        axis.set_xticks(x, names); axis.grid(axis="y", color="#DDDDDD", linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(ROOT/"figures/i1_2_downstream_transfer.png", dpi=300,
                facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)

    fixed = summary[summary.variant == "oof_soft_full"].copy()
    hard = summary[summary.variant == "oof_multi_hard"][
        ["dataset", "nll_gain", "regret_reduction"]]
    decision = fixed.merge(hard, on="dataset", suffixes=("_soft", "_hard"))
    decision["nll_transfer_advantage"] = decision.nll_gain_soft-decision.nll_gain_hard
    decision["regret_transfer_advantage"] = (
        decision.regret_reduction_soft-decision.regret_reduction_hard)
    decision.to_csv(ROOT/"results/i1_2_downstream_decision.csv", index=False)
    passes = int((decision.nll_transfer_advantage > 0).sum()) >= 2
    report = ["# I1-2 软监督下游传递", "",
              f"- 至少两个数据集改善下游NLL：{'通过' if passes else '未通过'}。",
              "- 所有版本使用同一骨干、后验、mixer、收缩控制器和selection协议。", "",
              markdown_table(decision), "",
              "配对比较及Holm校正见 `i1_2_downstream_comparisons.csv`。"]
    (ROOT/"results/i1_2_downstream_report.md").write_text(
        "\n".join(report), encoding="utf-8")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
