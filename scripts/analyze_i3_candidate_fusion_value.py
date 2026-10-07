"""I3-1: candidate-fusion value under one frozen action set.

Figure contract
---------------
Core conclusion: soft action mixing is more useful than hard routing, and the
coalition candidates provide measurable marginal probability-quality value on
top of the full/posterior actions in the non-saturated tasks.
Archetype: task gain, net correction, and candidate marginal-value panels.
Export: Python/Matplotlib, PNG only, 300 dpi, opaque white background.
Review risk: separate gains due to the posterior action from gains attributable
to coalition candidates; never compare a learned mixer only with full fusion.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; FIGURES = ROOT / "figures"
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
METHODS = ("posterior_only", "analytic_hard", "listwise_hard",
           "equal_action_average", "softmax_mixer", "sparsemax_mixer",
           "anchored_mixer", "anchored_harm", "anchored_harm_oracle")
LABELS = {"posterior_only": "Posterior", "analytic_hard": "Analytic hard",
          "listwise_hard": "Listwise hard", "equal_action_average": "Equal avg.",
          "softmax_mixer": "Softmax", "sparsemax_mixer": "Sparsemax",
          "anchored_mixer": "Full anchor", "anchored_harm": "+ harm",
          "anchored_harm_oracle": "Complete mixer"}
COLORS = {"posterior_only": "#8C8C8C", "analytic_hard": "#BDBDBD",
          "listwise_hard": "#D9D9D9", "equal_action_average": "#E69F00",
          "softmax_mixer": "#56B4E9", "sparsemax_mixer": "#CC79A7",
          "anchored_mixer": "#0072B2", "anchored_harm": "#009E73",
          "anchored_harm_oracle": "#D55E00"}


def holm(values):
    result = np.ones(len(values)); order = np.argsort(values); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values)-rank)*values[index]))
        result[index] = running
    return result


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    def fmt(value):
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join((header, rule, *rows))


def load_data():
    metrics, details = [], []
    for dataset in DATASETS:
        current = pd.read_csv(ROOT / "runs" / f"formal-e5-{dataset}" / "metrics_by_seed.csv")
        current["dataset"] = dataset; metrics.append(current)
        detail = pd.read_csv(ROOT / "runs" / f"formal-i3-1-{dataset}" /
                             "candidate_value_by_seed.csv")
        detail["dataset"] = dataset; details.append(detail)
    return pd.concat(metrics, ignore_index=True), pd.concat(details, ignore_index=True)


def paired_statistics(data):
    rows = []
    for dataset in DATASETS:
        part = data[data.dataset == dataset].pivot(
            index="train_seed", columns="variant", values="nll_gain")
        comparisons = {
            "complete_vs_best_hard": part.anchored_harm_oracle
                - part[["analytic_hard", "listwise_hard"]].max(axis=1),
            "complete_vs_posterior": part.anchored_harm_oracle - part.posterior_only,
            "complete_vs_softmax": part.anchored_harm_oracle - part.softmax_mixer,
            "complete_vs_sparsemax": part.anchored_harm_oracle - part.sparsemax_mixer,
        }
        for name, difference in comparisons.items():
            pvalue = 1.0 if np.allclose(difference, 0) else float(
                ttest_1samp(difference, 0, alternative="greater").pvalue)
            rows.append({"dataset": dataset, "comparison": name,
                         "metric": "nll_gain", "difference": difference.mean(),
                         "improved_seeds": int((difference > 0).sum()), "p": pvalue})
    result = pd.DataFrame(rows); result["holm_p"] = holm(result.p.to_numpy())
    return result


def plot(data, detail, path):
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    x = np.arange(4); names = ("MOSI", "MOSEI", "CREMA-D", "AV-MNIST")
    width = .095
    for axis, metric, ylabel, title, scale in (
        (axes[0, 0], "nll_gain", "NLL reduction vs full", "a  Probability quality", 1),
        (axes[0, 1], "accuracy_gain", "Accuracy change (pp)", "b  Decision accuracy", 100),
    ):
        summary = data.groupby(["dataset", "variant"])[metric].agg(["mean", "std"])
        for index, method in enumerate(METHODS):
            mean = [summary.loc[(dataset, method), "mean"]*scale for dataset in DATASETS]
            std = [summary.loc[(dataset, method), "std"]*scale for dataset in DATASETS]
            axis.bar(x+(index-4)*width, mean, width, yerr=std, capsize=1,
                     color=COLORS[method], label=LABELS[method], zorder=3)
        axis.axhline(0, color="#777777", lw=.8); axis.set_xticks(x, names, rotation=15, ha="right")
        axis.set_ylabel(ylabel); axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E0E0E0", linewidth=.6, zorder=0)
    axes[0, 0].legend(ncol=3, fontsize=6.7)

    complete = data[data.variant == "anchored_harm_oracle"].copy()
    summary = complete.groupby("dataset").agg(
        correction=("correction_rate", "mean"), correction_sd=("correction_rate", "std"),
        flip=("negative_flip_rate", "mean"), flip_sd=("negative_flip_rate", "std"))
    axes[1, 0].bar(x-.18, summary.loc[list(DATASETS), "correction"]*100, .36,
                   yerr=summary.loc[list(DATASETS), "correction_sd"]*100,
                   color="#009E73", capsize=2, label="Error → correct")
    axes[1, 0].bar(x+.18, summary.loc[list(DATASETS), "flip"]*100, .36,
                   yerr=summary.loc[list(DATASETS), "flip_sd"]*100,
                   color="#D55E00", capsize=2, label="Correct → error")
    axes[1, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 0].set_ylabel("Samples (%)"); axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("c  Corrections and negative flips", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", color="#E0E0E0", linewidth=.6, zorder=0)

    # The sparsemax removal counterfactual is ill-conditioned when it assigns
    # exactly zero mass to both fallback actions.  Keep that diagnostic in the
    # exported CSV, but focus the paper panel on the pre-registered complete
    # mixer whose fallback mass remains defined and whose candidate value is the
    # actual I3-1 attribution target.
    complete_detail = detail[detail.variant == "anchored_harm_oracle"]
    summary = complete_detail.groupby("dataset").agg(
        gain=("candidate_marginal_nll_gain", "mean"),
        gain_sd=("candidate_marginal_nll_gain", "std"),
        mass=("mean_candidate_mass", "mean"))
    means = summary.loc[list(DATASETS), "gain"].to_numpy()
    stds = summary.loc[list(DATASETS), "gain_sd"].to_numpy()
    bars = axes[1, 1].bar(x, means, .58, yerr=stds, capsize=2,
                          color=COLORS["anchored_harm_oracle"], zorder=3)
    for bar, mass in zip(bars, summary.loc[list(DATASETS), "mass"]):
        value = bar.get_height()
        offset = max(np.nanmax(np.abs(means))*0.045, 0.00015)
        axes[1, 1].text(bar.get_x()+bar.get_width()/2,
                        value + (offset if value >= 0 else -offset),
                        f"candidate mass {mass*100:.1f}%", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=6.5)
    axes[1, 1].axhline(0, color="#777777", lw=.8)
    axes[1, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 1].set_ylabel("NLL gain from retaining candidates")
    axes[1, 1].set_title("d  Candidate marginal value in complete mixer", loc="left",
                         fontweight="bold")
    axes[1, 1].grid(axis="y", color="#E0E0E0", linewidth=.6, zorder=0)
    fig.savefig(path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def main():
    data, detail = load_data()
    data.to_csv(RESULTS / "i3_1_candidate_fusion_by_seed.csv", index=False)
    detail.to_csv(RESULTS / "i3_1_candidate_value_by_seed.csv", index=False)
    summary = data[data.variant.isin(METHODS)].groupby(["dataset", "variant"]).agg(
        accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
        nll=("nll", "mean"), nll_gain=("nll_gain", "mean"),
        accuracy_gain=("accuracy_gain", "mean"),
        correction_rate=("correction_rate", "mean"),
        negative_flip_rate=("negative_flip_rate", "mean"),
        clipped_harm=("clipped_harm", "mean"),
        candidate_weight=("mean_candidate_weight", "mean")).reset_index()
    summary.to_csv(RESULTS / "i3_1_candidate_fusion_summary.csv", index=False)
    tests = paired_statistics(data); tests.to_csv(
        RESULTS / "i3_1_primary_comparisons.csv", index=False)
    complete = summary[summary.variant == "anchored_harm_oracle"].set_index("dataset")
    posterior = summary[summary.variant == "posterior_only"].set_index("dataset")
    candidate = detail[detail.variant == "anchored_harm_oracle"].groupby(
        "dataset").candidate_marginal_nll_gain.mean()
    decisions = pd.DataFrame([
        {"criterion": "complete soft mixer exceeds best hard route in NLL",
         "datasets_passed": int((tests[tests.comparison == "complete_vs_best_hard"].difference > 0).sum()),
         "required": 3},
        {"criterion": "complete mixer improves NLL over posterior alone",
         "datasets_passed": int(((complete.nll-posterior.nll) < 0).sum()), "required": 2},
        {"criterion": "coalition candidates have positive marginal NLL value",
         "datasets_passed": int((candidate > 0).sum()), "required": 2},
        {"criterion": "correction rate exceeds negative flips",
         "datasets_passed": int((complete.correction_rate > complete.negative_flip_rate).sum()),
         "required": 4},
    ])
    decisions["pass"] = decisions.datasets_passed >= decisions.required
    decisions.to_csv(RESULTS / "i3_1_decision.csv", index=False)
    plot(data, detail, FIGURES / "i3_1_candidate_fusion_value.png")
    report = ["# I3-1 候选融合价值实验", "",
              "所有版本共享骨干、后验、解析锚点、列表候选和训练预算。", "",
              "## 预注册判定", "", markdown_table(decisions), "",
              "## 主要配对比较", "", markdown_table(tests), "",
              "## 主摘要", "", markdown_table(summary), "",
              "候选边际价值通过删除候选动作并对完整联盟与后验动作重新归一化计算。"]
    (RESULTS / "i3_1_report.md").write_text("\n".join(report), encoding="utf-8")
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
